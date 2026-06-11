"""参数优化 — 滚动优化。"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import pandas as pd

from core.optimizer._grid import grid_search, run_single_backtest

logger = logging.getLogger(__name__)


def rolling_optimize(
    param_grid: Dict[str, List],
    metric: str,
    maximize: bool,
    strategy_class,
    data: pd.DataFrame,
    symbols: List[str],
    train_months: int = 6,
    test_months: int = 1,
    config=None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """执行滚动优化。

    每次使用 train_months 个月的数据优化参数，
    在接下来的 test_months 个月使用最佳参数回测。
    """
    if config is None:
        from pybroker import StrategyConfig as _SC
        config = _SC(initial_cash=1_000_000)

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    all_dates = sorted(df["date"].unique())

    if len(all_dates) < 2:
        logger.warning("数据不足，无法执行滚动优化")
        return pd.DataFrame()

    start_date = all_dates[0]
    end_date = all_dates[-1]
    rolling_results: List[Dict] = []
    current_start = start_date

    while current_start < end_date:
        train_end = current_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)

        if train_end > end_date:
            break

        train_dates = df[(df["date"] >= current_start) & (df["date"] < train_end)]["date"]
        test_dates = df[(df["date"] >= train_end) & (df["date"] < test_end)]["date"]

        if train_dates.empty or test_dates.empty:
            current_start = train_end
            continue

        train_data = df[df["date"].isin(train_dates)]
        test_data = df[df["date"].isin(test_dates)]

        grid_results_list, grid_df = grid_search(
            param_grid, metric, maximize, strategy_class,
            train_data, symbols, config,
        )

        if grid_df.empty:
            current_start = train_end
            continue

        best_params = grid_df.iloc[0]
        param_cols = list(param_grid.keys())
        best_param_dict = {k: best_params[k] for k in param_cols if k in best_params.index}

        try:
            strategy_instance = strategy_class(**best_param_dict)
            indicators = []
            if hasattr(strategy_instance, "register_indicators"):
                indicators = strategy_instance.register_indicators()

            test_metrics = run_single_backtest(
                strategy_fn=strategy_instance.execute,
                symbols=symbols,
                data=test_data,
                config=config,
                indicators=indicators,
            )

            if test_metrics:
                result = {
                    "train_start": str(current_start.date()),
                    "train_end": str(train_end.date()),
                    "test_start": str(train_end.date()),
                    "test_end": str(test_end.date()),
                    **best_param_dict,
                    **test_metrics,
                }
                rolling_results.append(result)
        except Exception as e:
            logger.error("滚动优化测试失败: %s", e)

        current_start = train_end

    if progress_callback and rolling_results:
        progress_callback(len(rolling_results), len(rolling_results))

    return pd.DataFrame(rolling_results)
