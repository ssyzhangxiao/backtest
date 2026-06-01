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
from core.market_regime import MarketRegimeDetector
from core.strategy_registry import StrategyLibrary
from core.engine.switch_engine import FactorScoringEngine
from core.engine.pybroker_data_source import PyBrokerDataSource
from core.engine.regime_indicator import RegimeIndicator
from core.engine.strategy_executor import StrategyExecutorFactory

logger = logging.getLogger(__name__)

try:
    import pybroker
    from pybroker import ExecContext
    PYBROKER_AVAILABLE = True
except ImportError:
    PYBROKER_AVAILABLE = False
    logger.warning("PyBroker 未安装。请运行: pip install pybroker>=1.0.0")
    ExecContext = Any


@dataclass
class PyBrokerResult:
    """PyBroker 回测结果封装。"""

    metrics: Dict[str, float]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    regime_history: pd.DataFrame
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
                logger.info(
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
      - _run_simplified: 自研简化引擎，仅用于交叉验证（并行运行对比结果）
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

        # 构建ScoringConfig，从BacktestConfig同步新参数
        from core.engine.switch_engine import ScoringConfig
        scoring_config = ScoringConfig(
            rebalance_days=self.config.rebalance_days,
            factor_weights=self.config.factor_weights,
            entry_threshold=self.config.entry_threshold,
            stop_loss_cooldown=self.config.stop_loss_cooldown,
            commission_rate=self.config.commission_rate,
            slippage_rate=self.config.slippage_rate,
            use_cross_section=self.config.use_cross_section,
            use_rank_score=self.config.use_rank_score,
            use_rolling_ic=self.config.use_rolling_ic,
            top_n_symbols=self.config.top_n_symbols,
        )
        self.switch_engine = FactorScoringEngine(self.library, scoring_config)
        self.regime_indicator = RegimeIndicator()
        self.executor_factory = StrategyExecutorFactory(
            self.library, self.switch_engine, self.config
        )
        self.executor_factory._total_symbols = len(self.target_symbols)

        # 注入滚动IC引擎
        if self.config.use_rolling_ic:
            from core.engine.rolling_ic import RollingICWeightEngine, RollingICConfig
            ic_config = RollingICConfig(
                window=60, forward_period=5, ema_alpha=0.1, min_observations=30
            )
            self._rolling_ic_engine = RollingICWeightEngine(ic_config)
            self.executor_factory._rolling_ic_engine = self._rolling_ic_engine
        else:
            self._rolling_ic_engine = None

        self._registered_strategies: List[str] = []
        self._last_result: Optional[PyBrokerResult] = None

    def register_strategies(self, strategy_names: List[str]):
        """注册策略名称列表。"""
        self._registered_strategies = list(strategy_names)
        logger.info("已注册策略: %s", strategy_names)

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

        strategies = self._registered_strategies
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
            logger.info(
                "过滤到目标品种: %s (%d 行)",
                sorted(matched),
                len(df),
            )

        if "is_dominant" in df.columns and df["is_dominant"].any():
            df = df[df["is_dominant"]].copy()
            if "product" in df.columns:
                df["symbol"] = df["product"]
            logger.info(
                "已过滤到主力合约: %d 行, %d 品种",
                len(df),
                df["symbol"].nunique(),
            )
        elif "is_dominant" not in df.columns:
            logger.info("没有主力合约信息，使用全部数据 (%d 行)", len(df))

        self.regime_indicator.fit(df)
        regime_fn, regime_conf_fn, regime_stab_fn = (
            self.regime_indicator.create_pybroker_regime_fn()
        )

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
            col for col in df.columns
            if col not in scope.default_data_cols and col not in scope.custom_data_cols
        ]
        if custom_cols_to_register:
            scope.register_custom_cols(custom_cols_to_register)
        strategy = pybroker.Strategy(df, start_date, end_date, config=pb_config)

        custom_params = getattr(self, "_custom_params", None) or {}

        ts_momentum_params = (
            dict(self.library.get_profile("ts_momentum").default_params)
            if self.library.get_profile("ts_momentum")
            else {}
        )
        ts_momentum_params.update(custom_params.get("ts_momentum", {}))
        _mom_window = int(ts_momentum_params.get("window", 20))

        roll_yield_params = (
            dict(self.library.get_profile("roll_yield").default_params)
            if self.library.get_profile("roll_yield")
            else {}
        )
        roll_yield_params.update(custom_params.get("roll_yield", {}))
        _ry_lookback = int(roll_yield_params.get("lookback", 20))
        _ry_entry_threshold = float(roll_yield_params.get("entry_threshold", 2.0))

        alpha019_params = (
            dict(self.library.get_profile("alpha019").default_params)
            if self.library.get_profile("alpha019")
            else {}
        )
        alpha019_params.update(custom_params.get("alpha019", {}))
        _a019_short_window = int(alpha019_params.get("short_window", 7))
        _a019_long_window = int(alpha019_params.get("long_window", 250))

        alpha032_params = (
            dict(self.library.get_profile("alpha032").default_params)
            if self.library.get_profile("alpha032")
            else {}
        )
        alpha032_params.update(custom_params.get("alpha032", {}))
        _a032_ma_window = int(alpha032_params.get("ma_window", 7))
        _a032_corr_window = int(alpha032_params.get("corr_window", 230))

        def _mom_ret(bar_data):
            return pd.Series(bar_data.close).pct_change(periods=_mom_window).to_numpy()

        def _roll_yield_ma(bar_data):
            return (
                pd.Series(bar_data.close)
                .rolling(_ry_lookback, min_periods=_ry_lookback)
                .mean()
                .to_numpy()
            )

        def _regime(bar_data):
            return regime_fn(bar_data)

        def _regime_conf(bar_data):
            return regime_conf_fn(bar_data)

        def _regime_stab(bar_data):
            return regime_stab_fn(bar_data)

        def _alpha019(bar_data):
            close = pd.Series(bar_data.close)
            close_7d_ago = close.shift(_a019_short_window)
            delta_7d = close.diff(_a019_short_window)
            short_term = close - close_7d_ago + delta_7d
            sign_component = -np.sign(short_term)
            returns = close.pct_change()
            cum_returns = pd.Series(np.nan, index=close.index)
            for i in range(_a019_long_window, len(close) + 1):
                window_returns = returns.iloc[i - _a019_long_window:i]
                cum_returns.iloc[i - 1] = np.prod(1 + window_returns) - 1
            cum_rank = cum_returns.rank(pct=True)
            result = sign_component * (1 + cum_rank)
            return result.values

        def _alpha032(bar_data):
            close = pd.Series(bar_data.close)
            ma_7 = close.rolling(window=_a032_ma_window, min_periods=_a032_ma_window).mean()
            price_deviation = ma_7 - close
            vwap = close
            if hasattr(bar_data, 'vwap') and bar_data.vwap is not None:
                vwap = pd.Series(bar_data.vwap)
            close_5d_ago = close.shift(5)
            rolling_corr = vwap.rolling(
                window=_a032_corr_window, min_periods=_a032_corr_window // 2
            ).corr(close_5d_ago)
            result = price_deviation + 20 * rolling_corr
            return result.values

        _indicators = [
            pybroker.indicator("mom_ret", _mom_ret),
            pybroker.indicator("roll_yield_ma", _roll_yield_ma),
            pybroker.indicator("alpha019_val", _alpha019),
            pybroker.indicator("alpha032_val", _alpha032),
            pybroker.indicator("regime", _regime),
            pybroker.indicator("regime_confidence", _regime_conf),
            pybroker.indicator("regime_stability", _regime_stab),
        ]

        symbols = sorted(df["symbol"].unique().tolist())
        primary_strategy = strategies[0] if strategies else "ts_momentum"
        use_exec_fusion = getattr(self, "_use_execute_fusion", False)

        if use_exec_fusion and len(strategies) > 1:
            from core.strategies import create_strategy
            strategy_instances = {}
            for sname in strategies:
                try:
                    profile = self.library.get_profile(sname)
                    params = dict(profile.default_params) if profile else {}
                    params.update(custom_params.get(sname, {}))
                    strategy_instances[sname] = create_strategy(sname, **params)
                except Exception as e:
                    logger.warning("创建策略实例 %s 失败: %s", sname, e)
            if strategy_instances:
                executor, fusion_indicators = self.executor_factory.create_fusion_executor(
                    strategy_instances,
                    use_weighted_fusion=True,
                    use_regime_filter=True,
                )
                all_indicators = _indicators + fusion_indicators
                strategy.add_execution(executor, symbols=symbols, indicators=all_indicators)
            else:
                executor = self.executor_factory.create_executor(
                    primary_strategy,
                    enable_switching=False,
                    all_strategy_names=strategies,
                    custom_params=custom_params,
                )
                strategy.add_execution(executor, symbols=symbols, indicators=_indicators)
        else:
            executor = self.executor_factory.create_executor(
                primary_strategy,
                enable_switching=False,
                all_strategy_names=strategies if len(strategies) > 1 else None,
                custom_params=custom_params,
            )
            strategy.add_execution(executor, symbols=symbols, indicators=_indicators)

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

        regime_df = self._run_regime_detection(df)

        return PyBrokerResult(
            metrics=metrics,
            equity_curve=equity_df,
            trades=trades,
            regime_history=regime_df,
            switch_log=self.switch_engine.get_decision_summary(),
        )

    def _run_fallback(
        self, start_date: str, end_date: str, initial_cash: float
    ) -> PyBrokerResult:
        """自研简化引擎回测（PyBroker 不可用时的 fallback）。"""
        return self._run_simplified(start_date, end_date, initial_cash)

    def _run_simplified(
        self, start_date: str, end_date: str, initial_cash: float
    ) -> PyBrokerResult:
        """简化回测引擎（fallback / 快速验证用）。"""
        cfg = self.config
        cost_rate = cfg.commission_rate + cfg.slippage_rate
        position_size = cfg.max_position_pct
        strategies = self._registered_strategies or ["ts_momentum"]

        strategy_params = {}
        for sname in strategies:
            sp = self.library.get_profile(sname)
            strategy_params[sname] = dict(sp.default_params) if sp else {}

        df = self.data_source.to_pybroker_df()
        df = df[
            (df["date"] >= pd.Timestamp(start_date))
            & (df["date"] <= pd.Timestamp(end_date))
        ]

        regime_result = self._run_regime_detection(df)

        symbols = self.data_source.symbols
        all_equities = []
        all_trades = []

        for symbol in symbols:
            sym_df = (
                df[df["symbol"] == symbol].sort_values("date").reset_index(drop=True)
            )
            if len(sym_df) < 50:
                continue

            cash = initial_cash / len(symbols)
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
                    equity = cash - shares * close
                else:
                    equity = cash

                stop_pct = cfg.stop_loss_pct
                if i >= 14:
                    h_window = sym_df["high"].iloc[i-13:i+1]
                    l_window = sym_df["low"].iloc[i-13:i+1]
                    c_shift = sym_df["close"].shift(1).iloc[i-13:i+1]
                    tr_vals = pd.concat([
                        (h_window - l_window).astype(float),
                        (h_window - c_shift).abs(),
                        (l_window - c_shift).abs()
                    ], axis=1).max(axis=1)
                    atr_val = tr_vals.mean()
                    atr_stop_pct = 2.0 * atr_val / close if close > 0 else stop_pct
                    effective_stop = max(stop_pct, atr_stop_pct)
                else:
                    effective_stop = stop_pct

                if position == 1 and close < entry_price * (1 - effective_stop):
                    base_pnl = shares * (close - entry_price)
                    exit_cost = shares * close * cost_rate
                    trade_records.append(
                        {"date": date, "symbol": symbol, "side": "stop_loss_long",
                         "price": close, "shares": shares, "pnl": base_pnl - exit_cost}
                    )
                    cash += shares * close * (1 - cost_rate)
                    position = 0
                    shares = 0

                elif position == -1 and close > entry_price * (1 + effective_stop):
                    base_pnl = shares * (entry_price - close)
                    exit_cost = shares * close * cost_rate
                    trade_records.append(
                        {"date": date, "symbol": symbol, "side": "stop_loss_short",
                         "price": close, "shares": shares, "pnl": base_pnl - exit_cost}
                    )
                    cash -= shares * close * (1 + cost_rate)
                    position = 0
                    shares = 0

                signal = self._generate_simple_signal(sym_df, i, strategies[0], strategy_params)

                if signal != 0:
                    if last_signal_dir != signal:
                        last_signal_dir = signal
                        signal = 0

                if signal == 1 and position != 1:
                    if position == -1:
                        base_pnl = shares * (entry_price - close)
                        exit_cost = shares * close * cost_rate
                        trade_records.append(
                            {"date": date, "symbol": symbol, "side": "short_close",
                             "price": close, "shares": shares, "pnl": base_pnl - exit_cost}
                        )
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
                        base_pnl = shares * (close - entry_price)
                        exit_cost = shares * close * cost_rate
                        trade_records.append(
                            {"date": date, "symbol": symbol, "side": "long_close",
                             "price": close, "shares": shares, "pnl": base_pnl - exit_cost}
                        )
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
                    equity = cash - shares * close
                else:
                    equity = cash

                equity_list.append({"date": date, "symbol": symbol, "equity": equity})

            all_equities.append(pd.DataFrame(equity_list))
            all_trades.append(
                pd.DataFrame(trade_records)
                if trade_records
                else pd.DataFrame(columns=["date", "symbol", "side", "price", "shares", "pnl"])
            )

        if not all_equities:
            empty_metrics = {"error": "no_data"}
            return PyBrokerResult(
                metrics=empty_metrics,
                equity_curve=pd.DataFrame(columns=["date", "equity"]),
                trades=pd.DataFrame(),
                regime_history=regime_result,
                switch_log=self.switch_engine.get_decision_summary(),
            )

        if len(all_equities) > 1:
            eq_curves: Dict[str, pd.Series] = {}
            for eq_df in all_equities:
                sym = eq_df["symbol"].iloc[0]
                eq_curves[sym] = pd.Series(eq_df["equity"].values, index=eq_df["date"])

            all_dates = sorted(set().union(*(e["date"] for e in all_equities)))
            portfolio_data = []
            for date in all_dates:
                day_eq = 0.0
                for eq_ser in eq_curves.values():
                    mask = eq_ser.index <= date
                    if mask.any():
                        day_eq += eq_ser.loc[mask].iloc[-1]
                portfolio_data.append({"date": date, "equity": day_eq})

            combined_eq = pd.DataFrame(portfolio_data)
        else:
            combined_eq = all_equities[0][["date", "equity"]]

        if len(combined_eq) > 1:
            daily_ret = combined_eq["equity"].pct_change().dropna()
            metrics = self._compute_simple_metrics(combined_eq["equity"], daily_ret)
        else:
            metrics = {"error": "insufficient_data"}

        trades_df = (
            pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        )

        return PyBrokerResult(
            metrics=metrics,
            equity_curve=combined_eq,
            trades=trades_df,
            regime_history=regime_result,
            switch_log=self.switch_engine.get_decision_summary(),
        )

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
        _train_bars = train_bars if train_bars is not None else self.config.wf_train_bars
        _test_bars = test_bars if test_bars is not None else self.config.wf_test_bars
        _step_bars = step_bars if step_bars is not None else self.config.wf_step_bars
        _train_ratio = train_ratio or self.config.wf_train_ratio
        _step_ratio = step_ratio or self.config.wf_step_ratio

        return self._walkforward_custom(
            start_date, end_date,
            train_ratio=_train_ratio, step_ratio=_step_ratio,
            train_bars=_train_bars, test_bars=_test_bars, step_bars=_step_bars,
        )

    def _walkforward_pybroker(
        self, start_date: str, end_date: str, train_ratio: float
    ) -> WalkforwardResult:
        """使用 PyBroker 内置 walkforward 方法。"""
        df = self.data_source.to_pybroker_df()
        df = df[
            (df["date"] >= pd.Timestamp(start_date))
            & (df["date"] <= pd.Timestamp(end_date))
        ]

        if "is_dominant" in df.columns and df["is_dominant"].any():
            df = df[df["is_dominant"]].copy()
            if "product" in df.columns:
                df["symbol"] = df["product"]

        pb_config = pybroker.StrategyConfig(
            initial_cash=self.config.initial_cash,
            buy_delay=self.config.pybroker_buy_delay,
            sell_delay=self.config.pybroker_sell_delay,
        )
        from pybroker.scope import StaticScope as _WFScope
        _wf_scope = _WFScope.instance()
        _wf_custom_cols = [
            col for col in df.columns
            if col not in _wf_scope.default_data_cols and col not in _wf_scope.custom_data_cols
        ]
        if _wf_custom_cols:
            _wf_scope.register_custom_cols(_wf_custom_cols)
        strategy = pybroker.Strategy(df, start_date, end_date, config=pb_config)

        def _wf_sma_5(bar_data):
            return pd.Series(bar_data.close).rolling(5).mean().to_numpy()

        def _wf_sma_20(bar_data):
            return pd.Series(bar_data.close).rolling(20).mean().to_numpy()

        def _wf_rsi_14(bar_data):
            close_ser = pd.Series(bar_data.close)
            rsi = PyBrokerBacktestRunner._compute_rsi(close_ser, 14)
            return rsi.fillna(50.0).to_numpy()

        def _wf_bb_upper(bar_data):
            close = pd.Series(bar_data.close)
            atr = PyBrokerBacktestRunner._compute_atr(pd.Series(bar_data.high), pd.Series(bar_data.low), close, 14)
            center = close.rolling(20, min_periods=1).mean()
            return (center + 1.5 * atr).to_numpy()

        def _wf_bb_lower(bar_data):
            close = pd.Series(bar_data.close)
            atr = PyBrokerBacktestRunner._compute_atr(pd.Series(bar_data.high), pd.Series(bar_data.low), close, 14)
            center = close.rolling(20, min_periods=1).mean()
            return (center - 1.5 * atr).to_numpy()

        regime_fn, regime_conf_fn, _ = self.regime_indicator.create_pybroker_regime_fn()

        def _wf_regime(bar_data):
            return regime_fn(bar_data)

        def _wf_regime_conf(bar_data):
            return regime_conf_fn(bar_data)

        _wf_indicators = [
            pybroker.indicator("sma_5", _wf_sma_5),
            pybroker.indicator("sma_20", _wf_sma_20),
            pybroker.indicator("rsi_14", _wf_rsi_14),
            pybroker.indicator("bb_upper", _wf_bb_upper),
            pybroker.indicator("bb_lower", _wf_bb_lower),
            pybroker.indicator("regime", _wf_regime),
            pybroker.indicator("regime_confidence", _wf_regime_conf),
        ]

        symbols = sorted(df["symbol"].unique().tolist())
        executor = self.executor_factory.create_executor("ts_momentum")
        strategy.add_execution(executor, symbols=symbols, indicators=_wf_indicators)

        n_windows = max(2, int(1.0 / (1.0 - train_ratio)))

        wf_result = strategy.walkforward(
            windows=n_windows,
            train_size=train_ratio,
            lookahead=self.config.pybroker_buy_delay,
        )

        windows = []
        equity_curves = []

        if hasattr(wf_result, "metrics_df") and isinstance(
            wf_result.metrics_df, pd.DataFrame
        ):
            mdf = wf_result.metrics_df
            for _, row in mdf.iterrows():
                window_metrics = {
                    k: v
                    for k, v in row.items()
                    if isinstance(v, (int, float)) and not pd.isna(v)
                }
                windows.append(
                    {
                        "train_start": str(row.get("train_start_date", "")),
                        "train_end": str(row.get("train_end_date", "")),
                        "test_start": str(row.get("test_start_date", "")),
                        "test_end": str(row.get("test_end_date", "")),
                        "metrics": window_metrics,
                    }
                )

        if hasattr(wf_result, "portfolio") and isinstance(
            wf_result.portfolio, pd.DataFrame
        ):
            pf = wf_result.portfolio.copy()
            if "market_value" in pf.columns:
                pf = pf.rename(columns={"market_value": "equity"})
            equity_curves.append(pf)

        overall = {}
        if windows:
            metric_keys = [
                k
                for k in windows[0]["metrics"]
                if isinstance(windows[0]["metrics"][k], (int, float))
            ]
            for key in metric_keys:
                vals = [w["metrics"][key] for w in windows]
                overall[key] = round(float(np.mean(vals)), 4)

        return WalkforwardResult(
            windows=windows,
            overall_metrics=overall,
            equity_curves=equity_curves,
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

            train_df = df[df["date"].isin(train_dates)]
            test_df = df[df["date"].isin(test_dates)]

            window_regime = RegimeIndicator(MarketRegimeDetector())
            window_regime.fit(train_df)
            regime_test = window_regime.detect(test_df)

            wf_params = {}
            for sname in (self._registered_strategies or ["ts_momentum"]):
                sp = self.library.get_profile(sname)
                wf_params[sname] = dict(sp.default_params) if sp else {}

            window_runner = _WindowRunner(
                symbols=self.data_source.symbols,
                strategies=self._registered_strategies or ["ts_momentum"],
                config=self.config,
                strategy_params=wf_params,
            )
            test_result = window_runner.run(test_df, regime_test)

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
            metric_keys = [
                k
                for k in windows[0]["metrics"]
                if isinstance(windows[0]["metrics"][k], (int, float))
            ]
            for key in metric_keys:
                vals = [w["metrics"][key] for w in windows]
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

    def _run_regime_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行环境检测并返回结果 DataFrame。"""
        return self.regime_indicator.detect(df)

    @staticmethod
    def _generate_simple_signal(df: pd.DataFrame, idx: int, strategy_name: str, params: dict = None) -> int:
        """简化信号生成（WalkForward fallback 引擎使用）。"""
        if params is None:
            params = {}
        close = df["close"]
        i = idx

        if strategy_name == "ts_momentum":
            window = params.get("window", 20)
            if i < window:
                return 0
            ret = close.iloc[i] / close.iloc[max(0, i - window)] - 1
            return 1 if ret > 0 else (-1 if ret < 0 else 0)

        elif strategy_name == "roll_yield":
            lookback = params.get("lookback", 20)
            entry_threshold = params.get("entry_threshold", 2.0)
            if i < lookback:
                return 0
            ma = close.iloc[max(0, i - lookback) : i + 1].mean()
            if ma <= 0:
                return 0
            spread_pct = (close.iloc[i] - ma) / ma * 100
            if spread_pct > entry_threshold:
                return -1
            elif spread_pct < -entry_threshold:
                return 1
            return 0

        elif strategy_name == "alpha019":
            if i < 7:
                return 0
            sign_val = -1 if close.iloc[i] < close.iloc[max(0, i - 7)] else 1
            return sign_val if abs(close.iloc[i] / close.iloc[max(0, i - 7)] - 1) > 0.01 else 0

        elif strategy_name == "alpha032":
            if i < 7:
                return 0
            ma_7 = close.iloc[max(0, i - 7) : i + 1].mean()
            deviation = ma_7 - close.iloc[i]
            return 1 if deviation > 0 else (-1 if deviation < 0 else 0)

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
        self, symbols: List[str], strategies: List[str], config: BacktestConfig,
        strategy_params: dict = None,
    ):
        self.symbols = symbols
        self.strategies = strategies
        self.config = config
        self.strategy_params = strategy_params or {}

    def run(self, df: pd.DataFrame, regime_df: pd.DataFrame) -> PyBrokerResult:
        """对单窗口执行简化回测。"""
        cfg = self.config
        cost_rate = cfg.commission_rate + cfg.slippage_rate
        position_size = cfg.max_position_pct

        symbols = (
            sorted(df["symbol"].unique()) if "symbol" in df.columns else self.symbols
        )
        strategy_name = self.strategies[0] if self.strategies else "ts_momentum"

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
                    hw = sym_df["high"].iloc[i-13:i+1]
                    lw = sym_df["low"].iloc[i-13:i+1]
                    cs = sym_df["close"].shift(1).iloc[i-13:i+1]
                    tr = pd.concat([(hw-lw).astype(float), (hw-cs).abs(), (lw-cs).abs()], axis=1).max(axis=1)
                    atr_val = tr.mean()
                    atr_stop = 2.0 * atr_val / close if close > 0 else stop_pct
                    effective_stop = max(stop_pct, atr_stop)
                else:
                    effective_stop = stop_pct

                if position == 1 and close < entry_price * (1 - effective_stop):
                    trade_records.append(
                        {"date": date, "side": "stop_loss_long", "price": close, "shares": shares}
                    )
                    cash += shares * close * (1 - cost_rate)
                    position = 0
                    shares = 0

                elif position == -1 and close > entry_price * (1 + effective_stop):
                    trade_records.append(
                        {"date": date, "side": "stop_loss_short", "price": close, "shares": shares}
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
                regime_history=regime_df,
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
            regime_history=regime_df,
            switch_log=pd.DataFrame(),
        )


def run_pybroker_backtest(
    df,
    start_date: str,
    end_date: str,
    strategy_names=None,
    initial_cash: float = 1_000_000,
    config=None,
):
    """
    便捷函数：一行代码执行 PyBroker 回测。

    Args:
        df: 行情 DataFrame（需含 date, symbol, open, high, low, close, volume）
        start_date: 回测开始日期
        end_date: 回测结束日期
        strategy_names: 策略名称列表，默认 ["ts_momentum"]
        initial_cash: 初始资金
        config: BacktestConfig 实例

    Returns:
        PyBrokerResult
    """
    from core.config import BacktestConfig
    from core.engine.pybroker_data_source import PyBrokerDataSource

    cfg = config or BacktestConfig()
    ds = PyBrokerDataSource(df)
    runner = PyBrokerBacktestRunner(ds, cfg)
    runner.register_strategies(strategy_names or ["ts_momentum"])
    return runner.run(start_date, end_date, initial_cash=initial_cash)
