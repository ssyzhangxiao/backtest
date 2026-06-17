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
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import BacktestConfig
from core.config.strategy_profiles import StrategyLibrary
from core.engine.switch_engine import FactorScoringEngine
from core.engine.pybroker_data_source import PyBrokerDataSource
from core.execution.pybroker_executor import PyBrokerExecutorBuilder
from core.execution.factor_pool import UnifiedFactorPool
from core.execution.signal_abstraction import SignalAbstractionLayer
from core.portfolio import PortfolioManager
from core.risk_controller import RiskController, RiskConfig
from utils.indicators import compute_atr

from core.execution._result_types import PyBrokerResult, WalkforwardResult
from core.execution._walkforward import walkforward as _walkforward, _WindowRunner
from core.execution._bootstrap import (
    bootstrap_metrics as _bootstrap_metrics,
    generate_simple_signal,
    compute_simple_metrics,
)

logger = logging.getLogger(__name__)

try:
    import pybroker

    PYBROKER_AVAILABLE = True
except ImportError:
    PYBROKER_AVAILABLE = False
    logger.warning("PyBroker 未安装。请运行: pip install pybroker>=1.0.0")


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
                use_composite_stop=self.config.stop_optimization_config.enabled,
                fixed_stop_pct=self.config.stop_optimization_config.composite_fixed_stop_pct,
                trailing_mode=self.config.stop_optimization_config.trailing_mode,
                trailing_pct=self.config.stop_optimization_config.trailing_pct,
                trailing_atr_mult=self.config.stop_optimization_config.trailing_atr_multiplier,
                max_holding_days=self.config.stop_optimization_config.time_stop_max_holding_days,
                time_target_return=self.config.stop_optimization_config.time_stop_target_return,
            )
        )

        if self.config.use_rolling_ic:
            from core.ext.factors.evaluator import FactorEvaluator

            ic_config = dict(
                window=60, forward_period=5, ema_alpha=0.1, min_observations=30
            )
            self._rolling_ic_engine = FactorEvaluator(
                forward_period=ic_config["forward_period"],
                ic_window=ic_config["window"],
                min_observations=ic_config["min_observations"],
            )
        else:
            self._rolling_ic_engine = None

        self._registered_strategies: List[str] = []
        self._last_result: Optional[PyBrokerResult] = None

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

    # ── 委托到 _bootstrap 模块的静态方法 ──

    @staticmethod
    def _generate_simple_signal(df, idx, strategy_name, params=None):
        """委托到 _bootstrap.generate_simple_signal。"""
        return generate_simple_signal(df, idx, strategy_name, params)

    @staticmethod
    def _compute_simple_metrics(equity, daily_returns):
        """委托到 _bootstrap.compute_simple_metrics。"""
        return compute_simple_metrics(equity, daily_returns)

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
            logger.debug("过滤到目标品种: %s (%d 行)", sorted(matched), len(df))

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

        from core.engine.strategy_indicators import StrategyIndicatorRegistry
        from core.engine.sub_strategy_indicators import register_default_indicators

        register_default_indicators()

        _DEFAULT_SUBS = [
            "trend",
            "term_structure",
            "mean_reversion",
            "vol_breakout",
            "composite_resonance",
        ]
        all_strategy_names = self._registered_strategies or _DEFAULT_SUBS
        if hasattr(self, "switch_engine") and self.switch_engine is not None:
            self.switch_engine.set_active_strategies(all_strategy_names)
        sub_params = {}
        for sname in all_strategy_names:
            sp = self.library.get_profile(sname)
            sub_params[sname] = dict(sp.default_params) if sp else {}
            sub_params[sname].update(custom_params.get(sname, {}))

        _indicators = [
            pybroker.indicator(name, fn)
            for name, fn in StrategyIndicatorRegistry.build_all(sub_params)
        ]
        _indicators.append(
            pybroker.indicator(
                "sma_20", lambda d: pd.Series(d.close).rolling(20).mean().values
            )
        )

        symbols = sorted(df["symbol"].unique().tolist())

        # ── 统一因子池注入（2026-06-13） ──
        factor_pool = UnifiedFactorPool()
        signal_layer = SignalAbstractionLayer(
            factor_pool,
            default_mode=self.config.signal_mode,
            cta_weight=self.config.cta_hybrid_weight,
            xs_position_base=self.config.xs_position_base,
            xs_position_ceiling=self.config.xs_position_ceiling,
            xs_opposite_penalty=self.config.xs_opposite_penalty,
        ) if self.config.use_signal_abstraction else None
        # 同步 hybrid_blend_method（方向二 2026-06-15）
        if signal_layer is not None:
            signal_layer.hybrid_blend_method = self.config.hybrid_blend_method

        blueprint_builder = PyBrokerExecutorBuilder(
            scoring_engine=self.switch_engine,
            portfolio_manager=self._portfolio,
            risk_controller=self._risk_controller,
            config=self.config,
            total_symbols=len(symbols),
            weight_method=getattr(self.config, "weight_method", "risk_parity"),
            risk_estimates_provider=self._estimate_symbol_risk,
            signal_abstraction=signal_layer,
        )

        # ── 预计算信号（2026-06-14：替代运行时 per-bar 计算） ──
        blueprint_builder.precompute_signals(df, sub_params)

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

        return PyBrokerResult(
            metrics=metrics,
            equity_curve=equity_df,
            trades=trades,
            switch_log=self.switch_engine.get_decision_summary(),
        )

    def _estimate_symbol_risk(self, symbol: str) -> Optional[float]:
        """估计品种风险（用于 risk_parity 权重分配）。"""
        try:
            sym_df = self.data_source.to_pybroker_df()
            sym_df = sym_df[sym_df["symbol"] == symbol].sort_values("date")
            if len(sym_df) < 60:
                return None

            high = pd.Series(sym_df["high"].values, dtype=float)
            low = pd.Series(sym_df["low"].values, dtype=float)
            close = pd.Series(sym_df["close"].values, dtype=float)

            atr = compute_atr(high, low, close, period=14, method="simple")
            last_close = float(close.iloc[-1])
            last_atr = float(atr.iloc[-1])
            if last_close > 0 and np.isfinite(last_atr) and last_atr > 0:
                return last_atr / last_close

            daily_ret = close.pct_change().dropna().tail(60)
            if len(daily_ret) >= 20:
                vol = float(daily_ret.std(ddof=0)) * float(np.sqrt(252))
                if vol > 0 and np.isfinite(vol):
                    return vol

            return None
        except Exception as e:
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
        """向前滚动分析（委托到 _walkforward 模块）。"""
        return _walkforward(
            self, start_date, end_date,
            train_ratio=train_ratio,
            step_ratio=step_ratio,
            train_bars=train_bars,
            test_bars=test_bars,
            step_bars=step_bars,
        )

    def bootstrap_metrics(self, n_samples: Optional[int] = None) -> Dict:
        """绩效指标 bootstrap 重采样（委托到 _bootstrap 模块）。"""
        return _bootstrap_metrics(self, n_samples)

    def get_last_result(self) -> Optional[PyBrokerResult]:
        """获取最近一次回测结果。"""
        return self._last_result
