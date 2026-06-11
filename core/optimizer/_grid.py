"""参数优化 — 网格搜索与单次回测。"""

from __future__ import annotations

import logging
from dataclasses import asdict
from itertools import product
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pybroker import StrategyConfig
else:
    try:
        from pybroker import StrategyConfig
    except ImportError:
        StrategyConfig = None  # type: ignore

logger = logging.getLogger(__name__)


def generate_param_combinations(param_grid: Dict[str, List]) -> List[Dict]:
    """生成参数组合列表。"""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = []
    for combo in product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def run_single_backtest(
    strategy_fn,
    symbols: List[str],
    data: pd.DataFrame,
    config,
    indicators: Optional[List] = None,
) -> Optional[Dict]:
    """运行单次回测并返回指标字典。"""
    from pybroker import Strategy

    try:
        start_date = str(data["date"].min().date())
        end_date = str(data["date"].max().date())
        strat = Strategy(data, start_date, end_date, config)
        strat.add_execution(fn=strategy_fn, symbols=symbols, indicators=indicators)
        result = strat.backtest()
        metrics_dict = asdict(result.metrics)
        return metrics_dict
    except Exception as e:
        logger.error("回测失败: %s", e)
        return None


def grid_search(
    param_grid: Dict[str, List],
    metric: str,
    maximize: bool,
    strategy_class,
    data: pd.DataFrame,
    symbols: List[str],
    config=None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple:
    """执行网格搜索。

    Returns:
        (results_list, results_df) 元组
    """
    if config is None:
        from pybroker import StrategyConfig as _SC
        config = _SC(initial_cash=1_000_000)

    combos = generate_param_combinations(param_grid)
    total = len(combos)
    results: List[Dict] = []

    for i, params in enumerate(combos):
        try:
            strategy_instance = strategy_class(**params)
            indicators = []
            if hasattr(strategy_instance, "register_indicators"):
                indicators = strategy_instance.register_indicators()

            metrics = run_single_backtest(
                strategy_fn=strategy_instance.execute,
                symbols=symbols,
                data=data,
                config=config,
                indicators=indicators,
            )
            if metrics is not None:
                result = {**params, **metrics}
                results.append(result)
        except Exception as e:
            logger.warning("参数组合 %s 失败: %s", params, e)

        if progress_callback:
            progress_callback(i + 1, total)

    if not results:
        return results, pd.DataFrame()

    results_df = pd.DataFrame(results)
    if metric in results_df.columns:
        results_df = results_df.sort_values(
            metric, ascending=not maximize
        ).reset_index(drop=True)

    return results, results_df
