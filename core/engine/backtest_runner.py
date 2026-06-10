"""
PyBroker 回测运行器 — 主回测、Walkforward、Bootstrap。

位置: core/engine/backtest_runner.py

提供:
  - PyBrokerResult: 回测结果封装
  - WalkforwardResult: Walkforward 结果封装
  - PyBrokerBacktestRunner: 主回测运行器
  - _WindowRunner: Walkforward 窗口独立运行器
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from core.config import BacktestConfig
from core.config.strategy_profiles import StrategyLibrary
from core.engine.switch_engine import FactorScoringEngine
from core.engine.pybroker_data_source import PyBrokerDataSource
from core.engine.pybroker_executor import PyBrokerExecutorBuilder  # P0-1 整改
from core.portfolio import PortfolioManager  # P0-2 整改
from core.risk_controller import RiskController, RiskConfig  # P0-3 整改
from utils.indicators import compute_atr  # P0-1 整改：风险估计公共函数

logger = logging.getLogger(__name__)

try:
    import pybroker

    PYBROKER_AVAILABLE = True
except ImportError:
    PYBROKER_AVAILABLE = False
    logger.warning("PyBroker 未安装。请运行: pip install pybroker>=1.0.0")


@dataclass
class PyBrokerResult:
    """PyBroker 回测结果封装。"""

    metrics: Dict[str, float]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    switch_log: pd.DataFrame
    bootstrap_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class WalkforwardResult:
    """Walkforward 向前滚动分析结果。"""

    windows: List[Dict[str, Any]]
    overall_metrics: Dict[str, float]
    equity_curves: List[pd.DataFrame]

    def plot_equity_curves(self):
        """绘制各窗口净值曲线（需 plotly）。"""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            for i, eq in enumerate(self.equity_curves):
                fig.add_trace(
                    go.Scatter(
                        x=eq["date"],
                        y=eq["equity"],
                        mode="lines",
                        name=f"Window {i + 1}",
                    )
                )
            fig.update_layout(
                title="Walkforward Equity Curves",
                xaxis_title="Date",
                yaxis_title="Equity",
            )
            fig.show()
        except ImportError:
            logger.warning("plotly 未安装，无法绘图。请运行: pip install plotly")
            for i, eq in enumerate(self.equity_curves):
                logger.debug(
                    "Window %d: final equity = %.2f",
                    i + 1,
                    eq["equity"].iloc[-1],
                )


class PyBrokerBacktestRunner:
    """
    PyBroker 主回测运行器。

    功能：
      - run: PyBroker 主回测（PyBroker 不可用直接报错，不再回退）
      - walkforward: 向前滚动分析
      - bootstrap_metrics: 绩效指标置信区间
    """

    def __init__(
        self,
        data_source: PyBrokerDataSource,
        config: Optional[BacktestConfig] = None,
        target_symbols: Optional[List[str]] = None,
    ):
        self.data_source = data_source
        self.target_symbols = target_symbols or data_source.symbols
        self.config = config or BacktestConfig()
        self.library = StrategyLibrary()

        # 构建ScoringConfig，从BacktestConfig同步信号层参数
        # P1-D1 整改：stop_loss_cooldown/commission_rate/slippage_rate 不属于信号层
        # 已迁移到 RiskConfig（规则4 风险控制统一），不再透传到 ScoringConfig
        from core.engine.switch_engine import ScoringConfig

        scoring_config = ScoringConfig(
            rebalance_days=self.config.rebalance_days,
            factor_weights=self.config.factor_weights,
            entry_threshold=self.config.entry_threshold,
            use_cross_section=self.config.use_cross_section,
            use_rank_score=self.config.use_rank_score,
            use_rolling_ic=self.config.use_rolling_ic,
            use_trend_filter=self.config.use_trend_filter,
        )
        self.switch_engine = FactorScoringEngine(self.library, scoring_config)
        # P0-1整改：不再使用 StrategyExecutorFactory，改为蓝图模式
        # PortfolioManager / RiskController 由 _run_pybroker 内部按蓝图组装
        self._portfolio = PortfolioManager(
            total_allocation=min(
                self.config.max_total_position_pct,
                0.8,
            ),
        )
        self._risk_controller = RiskController(
            RiskConfig(
                stop_loss_pct=self.config.stop_loss_pct,
                max_position_pct=self.config.max_position_pct,
                max_total_position_pct=self.config.max_total_position_pct,
                # P0-2 整改（2026-06-10）：注入复合止损参数，让 stop_optimization_config
                # 的 trailing_pct / max_holding_days 真正传递到 CompositeStopManager。
                # 之前这些字段保持默认值，追踪止损/时间止损实际从未触发。
                use_composite_stop=self.config.stop_optimization_config.enabled,
                fixed_stop_pct=self.config.stop_optimization_config.composite_fixed_stop_pct,
                trailing_mode=self.config.stop_optimization_config.trailing_mode,
                trailing_pct=self.config.stop_optimization_config.trailing_pct,
                trailing_atr_mult=self.config.stop_optimization_config.trailing_atr_multiplier,
                max_holding_days=self.config.stop_optimization_config.time_stop_max_holding_days,
                time_target_return=self.config.stop_optimization_config.time_stop_target_return,
            )
        )

        # 注入滚动IC引擎
        if self.config.use_rolling_ic:
            from core.engine.rolling_ic import RollingICWeightEngine, RollingICConfig

            ic_config = RollingICConfig(
                window=60, forward_period=5, ema_alpha=0.1, min_observations=30
            )
            self._rolling_ic_engine = RollingICWeightEngine(ic_config)
        else:
            self._rolling_ic_engine = None

        self._registered_strategies: List[str] = []
        self._last_result: Optional[PyBrokerResult] = None

        # 子策略适配器（新因子和子策略体系）
        # 规则1+规则17整改：完整集成新因子/子策略体系，不允许静默降级
        self._sub_strategy_adapter = None
        if self.config.use_sub_strategies:
            from core.engine.sub_strategy_adapter import SubStrategyAdapter

            merge_method = self.config.signal_merge_method
            self._sub_strategy_adapter = SubStrategyAdapter(
                config=self.config,
                use_new_factors=self.config.use_new_factors,
                use_sub_strategies=True,
                merge_method=merge_method,
            )
            logger.debug(
                "子策略适配器初始化成功（use_new_factors=%s）",
                self.config.use_new_factors,
            )

    def register_strategies(self, strategy_names: List[str]):
        """注册策略名称列表。"""
        self._registered_strategies = list(strategy_names)
        logger.debug("已注册策略: %s", strategy_names)

    def set_custom_params(self, custom_params: Dict[str, Dict[str, Any]]):
        """设置策略 custom_params（在 run() 时覆盖 default_params）。"""
        self._custom_params = custom_params
        logger.debug("已设置 custom_params: %s", list(custom_params.keys()))

    @staticmethod
    def _compute_rsi(close_ser, period):
        """RSI 计算（Wilder 平滑）。"""
        delta = close_ser.diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _compute_atr(high, low, close, period):
        """ATR 计算（Wilder 平滑）。"""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        if len(tr) > 0:
            tr.iloc[0] = tr1.iloc[0]
        return tr.rolling(period, min_periods=period).mean()

    def run(
        self,
        start_date: str,
        end_date: str,
        initial_cash: Optional[float] = None,
        custom_params: Optional[Dict[str, Dict[str, any]]] = None,
        use_execute_fusion: bool = False,
    ) -> PyBrokerResult:
        """执行回测（PyBroker 主引擎，不再回退到自研引擎）。"""
        if not self._registered_strategies:
            raise RuntimeError("请先调用 register_strategies() 注册策略")

        cash = initial_cash or self.config.initial_cash
        # 修复 best_params 注入 bug：合并 set_custom_params 传入的参数
        # （之前调用 set_custom_params 设的值会被这里的 None 覆盖）
        if custom_params is None and getattr(self, "_custom_params", None):
            custom_params = self._custom_params
        self._custom_params = custom_params
        self._use_execute_fusion = use_execute_fusion

        if not PYBROKER_AVAILABLE:
            raise RuntimeError(
                "PyBroker 未安装，无法执行回测。请运行: pip install pybroker>=1.0.0"
            )

        result = self._run_pybroker(start_date, end_date, cash)
        self._last_result = result
        return result

    def _run_pybroker(
        self, start_date: str, end_date: str, initial_cash: float
    ) -> PyBrokerResult:
        """使用 PyBroker 原生 API 执行回测。"""
        if not PYBROKER_AVAILABLE:
            raise RuntimeError("PyBroker 不可用")

        df = self.data_source.to_pybroker_df()

        df = df[
            (df["date"] >= pd.Timestamp(start_date))
            & (df["date"] <= pd.Timestamp(end_date))
        ]

        if self.target_symbols:
            available = set(df["symbol"].unique())
            target_set = set(self.target_symbols)
            matched = target_set & available
            if not matched:
                raise RuntimeError(
                    f"目标品种 {self.target_symbols} 不在数据中。"
                    f"可用品种: {sorted(available)[:10]}..."
                )
            df = df[df["symbol"].isin(matched)].copy()
            logger.debug(
                "过滤到目标品种: %s (%d 行)",
                sorted(matched),
                len(df),
            )

        if "is_dominant" in df.columns and df["is_dominant"].any():
            df = df[df["is_dominant"]].copy()
            if "product" in df.columns:
                df["symbol"] = df["product"]
            logger.debug(
                "已过滤到主力合约: %d 行, %d 品种",
                len(df),
                df["symbol"].nunique(),
            )
        elif "is_dominant" not in df.columns:
            logger.debug("没有主力合约信息，使用全部数据 (%d 行)", len(df))

        # 计算新因子（如果启用）
        if self._sub_strategy_adapter is not None and self.config.use_new_factors:
            logger.debug("计算新因子...")
            df = self._sub_strategy_adapter.compute_factors(df)

        pb_config = pybroker.StrategyConfig(
            initial_cash=initial_cash,
            buy_delay=self.config.pybroker_buy_delay,
            sell_delay=self.config.pybroker_sell_delay,
            bootstrap_samples=self.config.pybroker_bootstrap_samples,
        )
        for col in ["open", "high", "low", "close", "volume", "open_interest"]:
            if col in df.columns:
                df[col] = df[col].astype(float).to_numpy()
        from pybroker.scope import StaticScope

        scope = StaticScope.instance()
        custom_cols_to_register = [
            col
            for col in df.columns
            if col not in scope.default_data_cols and col not in scope.custom_data_cols
        ]
        if custom_cols_to_register:
            scope.register_custom_cols(custom_cols_to_register)
        strategy = pybroker.Strategy(df, start_date, end_date, config=pb_config)

        custom_params = getattr(self, "_custom_params", None) or {}

        # ── 5子策略指标注册 ──
        # 使用 StrategyIndicatorRegistry 替代硬编码指标构建
        from core.engine.strategy_indicators import StrategyIndicatorRegistry

        # P1-任务7整改：显式注册默认指标（不再依赖 import 时副作用）
        from core.engine.sub_strategy_indicators import register_default_indicators

        register_default_indicators()

        # 从策略注册表动态获取各子策略的参数
        # 修复 register_strategies 失效 bug：尊重 self._registered_strategies，
        # 未注册时回退到默认 5 子策略全集。
        _DEFAULT_SUBS = [
            "trend",
            "term_structure",
            "mean_reversion",
            "vol_breakout",
            "composite_resonance",
        ]
        all_strategy_names = self._registered_strategies or _DEFAULT_SUBS
        # 把激活的子策略集合同步给打分引擎，extract_factor_scores 据此过滤
        if hasattr(self, "switch_engine") and self.switch_engine is not None:
            self.switch_engine.set_active_strategies(all_strategy_names)
        sub_params = {}
        for sname in all_strategy_names:
            sp = self.library.get_profile(sname)
            sub_params[sname] = dict(sp.default_params) if sp else {}
            sub_params[sname].update(custom_params.get(sname, {}))

        # 通过注册表构建指标（解耦：不再硬编码任何指标计算逻辑）
        _indicators = [
            pybroker.indicator(name, fn)
            for name, fn in StrategyIndicatorRegistry.build_all(sub_params)
        ]
        # 添加通用指标（非策略特定）
        _indicators.append(
            pybroker.indicator(
                "sma_20", lambda d: pd.Series(d.close).rolling(20).mean().values
            )
        )

        symbols = sorted(df["symbol"].unique().tolist())

        # ──────────────────────────────────────────────────────
        # P0-1/P0-2/P0-3 整改：蓝图模式执行器
        # ──────────────────────────────────────────────────────
        blueprint_builder = PyBrokerExecutorBuilder(
            scoring_engine=self.switch_engine,
            portfolio_manager=self._portfolio,
            risk_controller=self._risk_controller,
            config=self.config,
            total_symbols=len(symbols),
            weight_method=getattr(self.config, "weight_method", "risk_parity"),
            risk_estimates_provider=self._estimate_symbol_risk,
        )
        blueprint_executor = blueprint_builder.build(strategy_params=sub_params)
        strategy.add_execution(
            blueprint_executor, symbols=symbols, indicators=_indicators
        )

        pb_result = strategy.backtest(
            start_date=start_date,
            end_date=end_date,
            lookahead=self.config.pybroker_buy_delay,
            calc_bootstrap=True,
        )
        self._last_pb_result = pb_result

        if hasattr(pb_result, "portfolio") and isinstance(
            pb_result.portfolio, pd.DataFrame
        ):
            pf = pb_result.portfolio.copy()
            equity_df = pf.reset_index()
            if "market_value" in equity_df.columns:
                equity_df = equity_df[["date", "market_value"]].rename(
                    columns={"market_value": "equity"}
                )
            elif "equity" in equity_df.columns:
                equity_df = equity_df[["date", "equity"]]
            else:
                equity_df = pd.DataFrame(columns=["date", "equity"])
        else:
            equity_df = pd.DataFrame(columns=["date", "equity"])

        if hasattr(pb_result, "trades") and isinstance(pb_result.trades, pd.DataFrame):
            trades = pb_result.trades.copy()
        else:
            trades = pd.DataFrame()

        if hasattr(pb_result, "metrics_df") and isinstance(
            pb_result.metrics_df, pd.DataFrame
        ):
            mdf = pb_result.metrics_df
            if "name" in mdf.columns and "value" in mdf.columns:
                metrics = dict(zip(mdf["name"], mdf["value"]))
            else:
                metrics = mdf.to_dict(orient="records")[0] if len(mdf) > 0 else {}
        elif hasattr(pb_result, "metrics"):
            m = pb_result.metrics
            metrics = {
                "sharpe": getattr(m, "sharpe", 0.0),
                "total_return_pct": getattr(m, "total_return_pct", 0.0),
                "max_drawdown_pct": getattr(m, "max_drawdown_pct", 0.0),
                "win_rate": getattr(m, "win_rate", 0.0),
                "profit_factor": getattr(m, "profit_factor", 0.0),
                "calmar": getattr(m, "calmar", 0.0),
                "trade_count": getattr(m, "trade_count", 0),
                "total_pnl": getattr(m, "total_pnl", 0.0),
            }
        else:
            metrics = {}

        # 不再需要市场环境检测
        # regime_df = self._run_regime_detection(df)

        return PyBrokerResult(
            metrics=metrics,
            equity_curve=equity_df,
            trades=trades,
            switch_log=self.switch_engine.get_decision_summary(),
        )

    # -----------------------------------------------------------------------
    # P0-1整改：蓝图辅助方法
    # -----------------------------------------------------------------------
    def _estimate_symbol_risk(self, symbol: str) -> Optional[float]:
        """
        估计品种风险（用于 risk_parity 权重分配）。

        规则17整改：使用 utils.indicators.compute_atr 公共函数，避免重复造轮。

        计算逻辑：
          1) 优先返回 ATR(14) / close，作为波动率代理
          2) ATR 不可用时使用 60 日日收益率年化波动率
          3) 数据不足时返回 None（risk_parity 会回退到等权）

        Args:
            symbol: 品种代码（如 'SHFE.RB'）

        Returns:
            风险估计值（>0），或 None（无法估计）
        """
        try:
            sym_df = self.data_source.to_pybroker_df()
            sym_df = sym_df[sym_df["symbol"] == symbol].sort_values("date")
            if len(sym_df) < 60:
                return None

            high = pd.Series(sym_df["high"].values, dtype=float)
            low = pd.Series(sym_df["low"].values, dtype=float)
            close = pd.Series(sym_df["close"].values, dtype=float)

            # 1) ATR(14) / close 路径
            atr = compute_atr(high, low, close, period=14, method="simple")
            last_close = float(close.iloc[-1])
            last_atr = float(atr.iloc[-1])
            if last_close > 0 and np.isfinite(last_atr) and last_atr > 0:
                return last_atr / last_close

            # 2) 回退：年化历史波动率（基于 60 日日收益率）
            daily_ret = close.pct_change().dropna().tail(60)
            if len(daily_ret) >= 20:
                vol = float(daily_ret.std(ddof=0)) * float(np.sqrt(252))
                if vol > 0 and np.isfinite(vol):
                    return vol

            return None
        except Exception as e:  # noqa: BLE001
            logger.debug("估计品种风险失败 %s: %s", symbol, e)
            return None

    def _get_default_sub_params(self) -> Dict[str, Dict[str, Any]]:
        """获取默认子策略参数（蓝图模式共用）。"""
        all_strategy_names = [
            "trend",
            "term_structure",
            "mean_reversion",
            "vol_breakout",
            "composite_resonance",
        ]
        sub_params: Dict[str, Dict[str, Any]] = {}
        for sname in all_strategy_names:
            sp = self.library.get_profile(sname)
            sub_params[sname] = dict(sp.default_params) if sp else {}
        return sub_params

    def walkforward(
        self,
        start_date: str,
        end_date: str,
        train_ratio: Optional[float] = None,
        step_ratio: Optional[float] = None,
        train_bars: Optional[int] = None,
        test_bars: Optional[int] = None,
        step_bars: Optional[int] = None,
    ) -> WalkforwardResult:
        """向前滚动分析。"""
        _train_bars = (
            train_bars if train_bars is not None else self.config.wf_train_bars
        )
        _test_bars = test_bars if test_bars is not None else self.config.wf_test_bars
        _step_bars = step_bars if step_bars is not None else self.config.wf_step_bars
        _train_ratio = train_ratio or self.config.wf_train_ratio
        _step_ratio = step_ratio or self.config.wf_step_ratio

        return self._walkforward_custom(
            start_date,
            end_date,
            train_ratio=_train_ratio,
            step_ratio=_step_ratio,
            train_bars=_train_bars,
            test_bars=_test_bars,
            step_bars=_step_bars,
        )

    def _walkforward_custom(
        self,
        start_date: str,
        end_date: str,
        train_ratio: float,
        step_ratio: float,
        train_bars: int = 0,
        test_bars: int = 0,
        step_bars: int = 0,
    ) -> WalkforwardResult:
        """自定义向前滚动分析。"""
        df = self.data_source.to_pybroker_df()
        df = df[
            (df["date"] >= pd.Timestamp(start_date))
            & (df["date"] <= pd.Timestamp(end_date))
        ]
        dates = sorted(df["date"].unique())
        total = len(dates)

        if train_bars > 0:
            _train_size = max(20, train_bars)
            _test_size = max(5, test_bars) if test_bars > 0 else max(5, train_bars // 4)
            _step_size = max(5, step_bars) if step_bars > 0 else max(5, _test_size)
        else:
            _train_size = max(20, int(total * train_ratio))
            _test_size = max(5, int(total * step_ratio))
            _step_size = _test_size

        windows = []
        equity_curves = []

        for test_start_idx in range(_train_size, total, _step_size):
            test_end_idx = min(test_start_idx + _test_size, total)
            if test_end_idx <= test_start_idx:
                continue

            train_start_idx = test_start_idx - _train_size
            train_dates = dates[train_start_idx:test_start_idx]
            test_dates = dates[test_start_idx:test_end_idx]

            if len(train_dates) < 10 or len(test_dates) < 5:
                continue

            test_df = df[df["date"].isin(test_dates)]

            wf_params = {}
            for sname in self._registered_strategies or ["trend"]:
                sp = self.library.get_profile(sname)
                wf_params[sname] = dict(sp.default_params) if sp else {}

            window_runner = _WindowRunner(
                symbols=self.data_source.symbols,
                strategies=self._registered_strategies or ["trend"],
                config=self.config,
                strategy_params=wf_params,
            )
            test_result = window_runner.run(test_df)

            windows.append(
                {
                    "train_start": str(train_dates[0].date()),
                    "train_end": str(train_dates[-1].date()),
                    "test_start": str(test_dates[0].date()),
                    "test_end": str(test_dates[-1].date()),
                    "metrics": test_result.metrics,
                }
            )
            equity_curves.append(test_result.equity_curve)

        overall = {}
        if windows:
            # 修复 2026-06-10：跨窗口 metrics 字段可能不一致（insufficient_data 时
            # 返回 {"error": ...}），用集合交集保证只计算所有窗口都有的字段。
            valid_metric_sets = [
                set(
                    k
                    for k in (w.get("metrics") or {})
                    if isinstance((w.get("metrics") or {}).get(k), (int, float))
                )
                for w in windows
            ]
            common_keys = (
                set.intersection(*valid_metric_sets) if valid_metric_sets else set()
            )
            for key in common_keys:
                vals = [(w.get("metrics") or {}).get(key) for w in windows]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                overall[key] = round(float(np.mean(vals)), 4)

        return WalkforwardResult(
            windows=windows,
            overall_metrics=overall,
            equity_curves=equity_curves,
        )

    def bootstrap_metrics(self, n_samples: Optional[int] = None) -> Dict:
        """绩效指标 bootstrap 重采样。"""
        if self._last_result is None:
            raise RuntimeError("请先调用 run()")

        n_samples = n_samples or self.config.pybroker_bootstrap_samples

        if PYBROKER_AVAILABLE:
            try:
                return self._bootstrap_pybroker(n_samples)
            except Exception as e:
                logger.warning("PyBroker bootstrap 失败 (%s)，回退到 numpy 实现。", e)

        return self._bootstrap_numpy(n_samples)

    def _bootstrap_pybroker(self, _n_samples: int) -> Dict:
        """使用 PyBroker 内置 bootstrap。"""
        if not hasattr(self, "_last_pb_result") or self._last_pb_result is None:
            raise RuntimeError("没有可用的 PyBroker 回测结果")

        pb_result = self._last_pb_result
        if not hasattr(pb_result, "bootstrap") or pb_result.bootstrap is None:
            raise RuntimeError(
                "回测结果中无 bootstrap 数据，请使用 calc_bootstrap=True"
            )

        bs = pb_result.bootstrap
        result = {}

        if hasattr(bs, "conf_intervals") and isinstance(
            bs.conf_intervals, pd.DataFrame
        ):
            ci = bs.conf_intervals
            for idx_val in ci.index:
                if isinstance(idx_val, tuple):
                    metric_name, conf_level = idx_val
                else:
                    metric_name, conf_level = str(idx_val), "value"
                row = ci.loc[idx_val]
                key = f"{metric_name} ({conf_level})"
                result[key] = {
                    "ci_lower": round(float(row.get("lower", 0)), 4),
                    "ci_upper": round(float(row.get("upper", 0)), 4),
                }
            return result

        for attr in ("sharpe", "drawdown", "profit_factor"):
            if hasattr(bs, attr):
                arr = getattr(bs, attr)
                if arr is not None and hasattr(arr, "__len__") and len(arr) > 0:
                    arr_np = np.asarray(arr)
                    result[attr] = {
                        "mean": round(float(np.mean(arr_np)), 4),
                        "std": round(float(np.std(arr_np)), 4),
                        "ci_lower": round(float(np.percentile(arr_np, 2.5)), 4),
                        "ci_upper": round(float(np.percentile(arr_np, 97.5)), 4),
                    }
        return result

    def _bootstrap_numpy(self, n_samples: int) -> Dict:
        """numpy 自实现 bootstrap。"""
        equity = self._last_result.equity_curve["equity"]
        daily_returns = equity.pct_change().dropna()

        if len(daily_returns) < 10:
            return {"error": "样本太少，无法 bootstrap"}

        n = len(daily_returns)
        rng = np.random.default_rng(42)

        metrics_samples: Dict[str, List[float]] = {
            "sharpe": [],
            "total_return": [],
            "max_drawdown": [],
            "calmar": [],
            "win_rate": [],
        }

        actual_samples = n_samples
        if actual_samples > 50000:
            logger.info(
                "bootstrap n_samples=%d 较大，可能需要较长时间。", actual_samples
            )

        for _ in range(actual_samples):
            idx = rng.integers(0, n, size=n)
            ret_sample = daily_returns.iloc[idx].values
            eq_sample = equity.iloc[0] * np.cumprod(1 + np.insert(ret_sample, 0, 0))

            ann_factor = np.sqrt(252)
            sharpe = (np.mean(ret_sample) / max(np.std(ret_sample), 1e-8)) * ann_factor
            total_ret = (eq_sample[-1] - eq_sample[0]) / eq_sample[0]
            peak = np.maximum.accumulate(eq_sample)
            dd = np.min((eq_sample - peak) / peak)
            calmar = total_ret / abs(dd) if abs(dd) > 1e-10 else 0.0
            win_rate = np.mean(ret_sample > 0)

            metrics_samples["sharpe"].append(float(sharpe))
            metrics_samples["total_return"].append(float(total_ret))
            metrics_samples["max_drawdown"].append(float(dd))
            metrics_samples["calmar"].append(float(calmar))
            metrics_samples["win_rate"].append(float(win_rate))

        result = {}
        for key, vals in metrics_samples.items():
            arr = np.array(vals)
            result[key] = {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "ci_lower": round(float(np.percentile(arr, 2.5)), 4),
                "ci_upper": round(float(np.percentile(arr, 97.5)), 4),
            }

        self._last_result.bootstrap_metrics = result
        return result

    @staticmethod
    def _generate_simple_signal(
        df: pd.DataFrame, idx: int, strategy_name: str, params: dict = None
    ) -> int:
        """简化信号生成（WalkForward fallback 引擎使用）。5子策略版本。"""
        if params is None:
            params = {}
        close = df["close"]
        i = idx

        # 5子策略信号映射
        if strategy_name == "trend":
            window = params.get("window", 20)
            if i < window:
                return 0
            ret = close.iloc[i] / close.iloc[max(0, i - window)] - 1
            signal = np.tanh(ret * 5)
            return 1 if signal > 0.2 else (-1 if signal < -0.2 else 0)

        elif strategy_name == "term_structure":
            lookback = params.get("lookback", 20)
            if i < lookback:
                return 0
            ma = close.iloc[max(0, i - lookback) : i + 1].mean()
            if ma <= 0:
                return 0
            spread_pct = (close.iloc[i] - ma) / ma * 100
            signal = np.tanh(-spread_pct / 3.0)
            return 1 if signal > 0.2 else (-1 if signal < -0.2 else 0)

        elif strategy_name == "mean_reversion":
            short_window = params.get("short_window", 7)
            if i < short_window:
                return 0
            delta_n = close.iloc[i] - close.iloc[max(0, i - short_window)]
            sign_val = -1 if delta_n > 0 else 1
            return sign_val if abs(delta_n / close.iloc[i]) > 0.01 else 0

        elif strategy_name == "vol_breakout":
            ma_window = params.get("ma_window", 7)
            if i < ma_window:
                return 0
            ma = close.iloc[max(0, i - ma_window) : i + 1].mean()
            deviation = ma - close.iloc[i]
            signal = np.tanh(deviation * 0.1)
            return 1 if signal > 0.2 else (-1 if signal < -0.2 else 0)

        elif strategy_name == "composite_resonance":
            # 复合共振：趋势+均值回归等权叠加
            window = params.get("window", 20)
            short_window = params.get("short_window", 7)
            if i < max(window, short_window):
                return 0
            ret = close.iloc[i] / close.iloc[max(0, i - window)] - 1
            trend_signal = np.tanh(ret * 5)
            delta_n = close.iloc[i] - close.iloc[max(0, i - short_window)]
            mr_signal = -1 if delta_n > 0 else 1
            composite = (trend_signal + mr_signal * 0.5) / 2.0
            return 1 if composite > 0.2 else (-1 if composite < -0.2 else 0)

        return 0

    @staticmethod
    def _compute_simple_metrics(
        equity: pd.Series, daily_returns: pd.Series
    ) -> Dict[str, float]:
        """计算简化绩效指标。"""
        if len(daily_returns) < 2 or len(equity) < 2:
            return {"error": "insufficient_data"}

        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        ann_factor = np.sqrt(252)
        sharpe = (daily_returns.mean() / max(daily_returns.std(), 1e-8)) * ann_factor
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        max_dd = dd.min()
        calmar = total_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0
        win_rate = (daily_returns > 0).mean()

        return {
            "total_return": round(total_return, 4),
            "total_return_pct": round(total_return * 100, 2),
            "sharpe": round(float(sharpe), 3),
            "max_drawdown": round(float(max_dd), 4),
            "max_drawdown_pct": round(float(max_dd) * 100, 2),
            "calmar": round(float(calmar), 3),
            "win_rate": round(float(win_rate), 4),
            "n_days": len(daily_returns),
            "final_equity": round(float(equity.iloc[-1]), 2),
        }

    def get_last_result(self) -> Optional[PyBrokerResult]:
        """获取最近一次回测结果。"""
        return self._last_result


class _WindowRunner:
    """
    Walkforward 每轮窗口的独立回测运行器。

    与主 PyBrokerBacktestRunner 隔离，避免状态污染。
    使用简化引擎（避免 PyBroker 全局状态问题）。
    """

    def __init__(
        self,
        symbols: List[str],
        strategies: List[str],
        config: BacktestConfig,
        strategy_params: dict = None,
    ):
        self.symbols = symbols
        self.strategies = strategies
        self.config = config
        self.strategy_params = strategy_params or {}

    def run(self, df: pd.DataFrame) -> PyBrokerResult:
        """对单窗口执行简化回测。"""
        cfg = self.config
        cost_rate = cfg.commission_rate + cfg.slippage_rate
        position_size = cfg.max_position_pct

        symbols = (
            sorted(df["symbol"].unique()) if "symbol" in df.columns else self.symbols
        )
        strategy_name = self.strategies[0] if self.strategies else "trend"

        all_equities = []
        all_trades = []

        per_symbol_cash = cfg.initial_cash / max(len(symbols), 1)

        for symbol in symbols:
            sym_df = df[df["symbol"] == symbol] if "symbol" in df.columns else df.copy()
            sym_df = sym_df.sort_values("date").reset_index(drop=True)
            if len(sym_df) < 20:
                continue

            cash = per_symbol_cash
            position = 0
            entry_price = 0.0
            shares = 0
            last_signal_dir = 0
            equity_list = []
            trade_records = []

            for i in range(len(sym_df)):
                row = sym_df.iloc[i]
                close = row["close"]
                date = row["date"]

                if position == 1:
                    equity = cash + shares * close
                elif position == -1:
                    equity = cash + shares * (entry_price - close)
                else:
                    equity = cash

                stop_pct = cfg.stop_loss_pct
                if i >= 14:
                    hw = sym_df["high"].iloc[i - 13 : i + 1]
                    lw = sym_df["low"].iloc[i - 13 : i + 1]
                    cs = sym_df["close"].shift(1).iloc[i - 13 : i + 1]
                    tr = pd.concat(
                        [(hw - lw).astype(float), (hw - cs).abs(), (lw - cs).abs()],
                        axis=1,
                    ).max(axis=1)
                    atr_val = tr.mean()
                    atr_stop = 2.0 * atr_val / close if close > 0 else stop_pct
                    effective_stop = max(stop_pct, atr_stop)
                else:
                    effective_stop = stop_pct

                if position == 1 and close < entry_price * (1 - effective_stop):
                    trade_records.append(
                        {
                            "date": date,
                            "side": "stop_loss_long",
                            "price": close,
                            "shares": shares,
                        }
                    )
                    cash += shares * close * (1 - cost_rate)
                    position = 0
                    shares = 0

                elif position == -1 and close > entry_price * (1 + effective_stop):
                    trade_records.append(
                        {
                            "date": date,
                            "side": "stop_loss_short",
                            "price": close,
                            "shares": shares,
                        }
                    )
                    cash -= shares * close * (1 + cost_rate)
                    position = 0
                    shares = 0

                strategy_params = self.strategy_params.get(strategy_name, {})

                signal = PyBrokerBacktestRunner._generate_simple_signal(
                    sym_df, i, strategy_name, strategy_params
                )

                if signal != 0:
                    if last_signal_dir != signal:
                        last_signal_dir = signal
                        signal = 0

                if signal == 1 and position != 1:
                    if position == -1:
                        cash -= shares * close * (1 + cost_rate)
                        position = 0
                        shares = 0
                    alloc = equity * position_size
                    shares = int(alloc / close) if close > 0 else 0
                    if shares > 0:
                        cash -= shares * close * (1 + cost_rate)
                        entry_price = close
                        position = 1

                elif signal == -1 and position != -1:
                    if position == 1:
                        cash += shares * close * (1 - cost_rate)
                        position = 0
                        shares = 0
                    alloc = equity * position_size
                    shares = int(alloc / close) if close > 0 else 0
                    if shares > 0:
                        cash += shares * close * (1 - cost_rate)
                        entry_price = close
                        position = -1

                if position == 1:
                    equity = cash + shares * close
                elif position == -1:
                    equity = cash + shares * (entry_price - close)
                else:
                    equity = cash

                equity_list.append({"date": date, "equity": equity})

            all_equities.append(pd.DataFrame(equity_list))
            tdf = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()
            if not tdf.empty:
                tdf["symbol"] = symbol
            all_trades.append(tdf)

        if not all_equities:
            return PyBrokerResult(
                metrics={"error": "no_data"},
                equity_curve=pd.DataFrame(columns=["date", "equity"]),
                trades=pd.DataFrame(),
                switch_log=pd.DataFrame(),
            )

        if len(all_equities) > 1:
            combined = all_equities[0][["date", "equity"]].copy()
            combined = combined.rename(columns={"equity": "eq_0"})
            for j, eq_df in enumerate(all_equities[1:], 1):
                merged = eq_df[["date", "equity"]].rename(columns={"equity": f"eq_{j}"})
                combined = pd.merge(combined, merged, on="date", how="outer")
            eq_cols = [c for c in combined.columns if c.startswith("eq_")]
            combined["equity"] = combined[eq_cols].ffill().sum(axis=1)
            combined_eq = combined[["date", "equity"]].ffill()
        else:
            combined_eq = all_equities[0][["date", "equity"]]

        trades_df = (
            pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        )

        if len(combined_eq) > 1:
            daily_ret = combined_eq["equity"].pct_change().dropna()
            metrics = PyBrokerBacktestRunner._compute_simple_metrics(
                combined_eq["equity"], daily_ret
            )
        else:
            metrics = {"error": "insufficient_data"}

        return PyBrokerResult(
            metrics=metrics,
            equity_curve=combined_eq,
            trades=trades_df,
            switch_log=pd.DataFrame(),
        )
