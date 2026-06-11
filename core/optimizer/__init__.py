"""参数优化模块。

基于 PyBroker 的回测能力实现参数网格搜索和滚动优化。

⚠️ 调用约定（规则17 - 不重复造轮子）：
  ParameterOptimizer 属于底层基础设施，必须通过官方入口调用：
    - 根目录脚本 run_optimize.py
    - runner/optimization/ 下的优化子模块

子模块：
  - _grid: 网格搜索与单次回测
  - _rolling: 滚动优化
  - _walkforward: Walk-Forward 优化与稳健性评分
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from core.optimizer._grid import (
    generate_param_combinations,
    grid_search,
    run_single_backtest,
)
from core.optimizer._rolling import rolling_optimize
from core.optimizer._walkforward import (
    compute_robustness,
    walkforward_optimize,
    select_robust_params,
)

if TYPE_CHECKING:
    from pybroker import StrategyConfig
else:
    try:
        from pybroker import StrategyConfig
    except ImportError:
        StrategyConfig = None  # type: ignore

logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """参数优化器。

    支持网格搜索、滚动优化和 Walk-Forward 优化三种模式。
    """

    SUPPORTED_METRICS = [
        "sharpe",
        "total_return_pct",
        "profit_factor",
        "max_drawdown_pct",
        "win_rate",
        "sortino",
    ]

    def __init__(
        self,
        param_grid: Dict[str, List],
        metric: str = "sharpe",
        maximize: bool = True,
    ):
        self.param_grid = param_grid
        self.metric = metric
        self.maximize = maximize
        self.results: List[Dict] = []

    def _generate_param_combinations(self) -> List[Dict]:
        return generate_param_combinations(self.param_grid)

    @staticmethod
    def _run_single_backtest(
        strategy_fn,
        symbols: List[str],
        data: pd.DataFrame,
        config,
        indicators: Optional[List] = None,
    ) -> Optional[Dict]:
        return run_single_backtest(strategy_fn, symbols, data, config, indicators)

    def grid_search(
        self,
        strategy_class,
        data: pd.DataFrame,
        symbols: List[str],
        config=None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        results, results_df = grid_search(
            self.param_grid, self.metric, self.maximize,
            strategy_class, data, symbols, config, progress_callback,
        )
        self.results = results
        return results_df

    def rolling_optimize(
        self,
        strategy_class,
        data: pd.DataFrame,
        symbols: List[str],
        train_months: int = 6,
        test_months: int = 1,
        config=None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        return rolling_optimize(
            self.param_grid, self.metric, self.maximize,
            strategy_class, data, symbols,
            train_months, test_months, config, progress_callback,
        )

    def walkforward_optimize(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        runner_factory: Callable,
        train_years: int = 2,
        test_years: int = 1,
        metric: str = "sharpe",
    ) -> pd.DataFrame:
        return walkforward_optimize(
            self.param_grid, self.metric, self.maximize,
            data, symbols, runner_factory,
            train_years, test_years,
        )

    def _compute_robustness(self, results_df: pd.DataFrame, metric: str) -> Dict[int, float]:
        return compute_robustness(self.param_grid, results_df, metric)

    @staticmethod
    def select_robust_params(wf_df: pd.DataFrame, param_cols: List[str]) -> Dict:
        return select_robust_params(wf_df, param_cols)

    def get_best_params(self) -> Optional[Dict]:
        if not self.results:
            return None
        results_df = pd.DataFrame(self.results)
        if self.metric not in results_df.columns:
            return None
        best_idx = (
            results_df[self.metric].idxmax()
            if self.maximize
            else results_df[self.metric].idxmin()
        )
        param_cols = list(self.param_grid.keys())
        best = results_df.loc[best_idx]
        return {k: best[k] for k in param_cols if k in best.index}

    def save_results(self, filepath: str):
        if not self.results:
            return
        serializable = []
        for r in self.results:
            item = {}
            for k, v in r.items():
                if isinstance(v, np.integer):
                    item[k] = int(v)
                elif isinstance(v, np.floating):
                    item[k] = float(v)
                elif isinstance(v, np.ndarray):
                    item[k] = v.tolist()
                else:
                    item[k] = v
            serializable.append(item)
        dir_path = os.path.dirname(filepath) or "."
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def load_results(filepath: str) -> List[Dict]:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"优化结果文件不存在: {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"优化结果文件格式错误: {filepath}", e.doc, e.pos
            ) from e


__all__ = [
    "ParameterOptimizer",
    "generate_param_combinations",
    "grid_search",
    "run_single_backtest",
    "rolling_optimize",
    "walkforward_optimize",
    "compute_robustness",
    "select_robust_params",
]
