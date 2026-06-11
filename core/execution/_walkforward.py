"""回测运行器 — Walkforward 向前滚动分析。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import BacktestConfig
from core.config.strategy_profiles import StrategyLibrary
from core.execution._result_types import PyBrokerResult, WalkforwardResult

logger = logging.getLogger(__name__)


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
        from core.execution.backtest_runner import PyBrokerBacktestRunner

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


def walkforward(
    runner,
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
        train_bars if train_bars is not None else runner.config.wf_train_bars
    )
    _test_bars = test_bars if test_bars is not None else runner.config.wf_test_bars
    _step_bars = step_bars if step_bars is not None else runner.config.wf_step_bars
    _train_ratio = train_ratio or runner.config.wf_train_ratio
    _step_ratio = step_ratio or runner.config.wf_step_ratio

    return _walkforward_custom(
        runner,
        start_date,
        end_date,
        train_ratio=_train_ratio,
        step_ratio=_step_ratio,
        train_bars=_train_bars,
        test_bars=_test_bars,
        step_bars=_step_bars,
    )


def _walkforward_custom(
    runner,
    start_date: str,
    end_date: str,
    train_ratio: float,
    step_ratio: float,
    train_bars: int = 0,
    test_bars: int = 0,
    step_bars: int = 0,
) -> WalkforwardResult:
    """自定义向前滚动分析。"""
    df = runner.data_source.to_pybroker_df()
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
        for sname in runner._registered_strategies or ["trend"]:
            sp = runner.library.get_profile(sname)
            wf_params[sname] = dict(sp.default_params) if sp else {}

        window_runner = _WindowRunner(
            symbols=runner.data_source.symbols,
            strategies=runner._registered_strategies or ["trend"],
            config=runner.config,
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
