"""
参数优化模块。

基于 PyBroker 的回测能力实现参数网格搜索和滚动优化。
使用自定义的网格搜索 + 滚动窗口优化策略。

优化流程：
1. 定义参数搜索空间
2. 对每组参数运行回测
3. 按目标指标（如 Sharpe）排序
4. 返回最佳参数组合

滚动优化：
- 每月使用最近N月数据优化参数
- 在下一个月使用优化后的参数进行回测
- 避免过拟合

策略类接口要求：
- 必须提供 execute(ctx) 方法，签名为 def execute(self, ctx: ExecContext)
- 可选提供 register_indicators() 方法，返回 pybroker.indicator 列表
"""

import logging
import os
from dataclasses import asdict

import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Optional, Callable
import json

from pybroker import Strategy, StrategyConfig

logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """
    参数优化器。

    支持网格搜索和滚动优化两种模式。

    Attributes:
        param_grid: 参数搜索空间 {参数名: [候选值列表]}
        metric: 优化目标指标，如 'sharpe', 'total_return_pct'
        data: 回测数据 DataFrame
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
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = []
        for combo in product(*values):
            combos.append(dict(zip(keys, combo)))
        return combos

    @staticmethod
    def _run_single_backtest(
        strategy_fn,
        symbols: List[str],
        data: pd.DataFrame,
        config: StrategyConfig,
        indicators: Optional[List] = None,
    ) -> Optional[Dict]:
        """
        运行单次回测并返回指标字典。

        Args:
            strategy_fn: 策略执行函数 (ctx -> None)
            symbols: 交易合约代码列表
            data: 回测数据
            config: 策略配置
            indicators: PyBroker 指标列表，传递给 add_execution

        Returns:
            回测指标字典，或 None（回测失败时）
        """
        try:
            start_date = str(data["date"].min().date())
            end_date = str(data["date"].max().date())
            strat = Strategy(data, start_date, end_date, config)
            strat.add_execution(fn=strategy_fn, symbols=symbols, indicators=indicators)
            result = strat.backtest()
            metrics_dict = asdict(result.metrics)
            return metrics_dict
        except Exception as e:
            logger.error(f"回测失败: {e}")
            return None

    def grid_search(
        self,
        strategy_class,
        data: pd.DataFrame,
        symbols: List[str],
        config: Optional[StrategyConfig] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        """
        执行网格搜索。

        对参数空间中的每组参数，创建策略实例并运行回测，
        收集所有结果并按目标指标排序。

        策略类接口要求：
          - __init__(**params): 接受参数网格中的参数作为关键字参数
          - execute(ctx): 回测执行函数
          - register_indicators() [可选]: 返回 pybroker.indicator 列表

        Args:
            strategy_class: 策略类
            data: 回测数据
            symbols: 交易合约代码列表
            config: 策略配置，为 None 时使用默认配置
            progress_callback: 进度回调 callback(current, total)

        Returns:
            优化结果 DataFrame
        """
        if config is None:
            config = StrategyConfig(initial_cash=1_000_000)

        combos = self._generate_param_combinations()
        total = len(combos)
        self.results = []

        for i, params in enumerate(combos):
            try:
                strategy_instance = strategy_class(**params)

                indicators = []
                if hasattr(strategy_instance, "register_indicators"):
                    indicators = strategy_instance.register_indicators()

                metrics = self._run_single_backtest(
                    strategy_fn=strategy_instance.execute,
                    symbols=symbols,
                    data=data,
                    config=config,
                    indicators=indicators,
                )

                if metrics is not None:
                    result = {**params, **metrics}
                    self.results.append(result)

            except Exception as e:
                logger.warning(f"参数组合 {params} 失败: {e}")

            if progress_callback:
                progress_callback(i + 1, total)

        if not self.results:
            return pd.DataFrame()

        results_df = pd.DataFrame(self.results)

        if self.metric in results_df.columns:
            results_df = results_df.sort_values(
                self.metric, ascending=not self.maximize
            ).reset_index(drop=True)

        return results_df

    def rolling_optimize(
        self,
        strategy_class,
        data: pd.DataFrame,
        symbols: List[str],
        train_months: int = 6,
        test_months: int = 1,
        config: Optional[StrategyConfig] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        """
        执行滚动优化。

        每次使用 train_months 个月的数据优化参数，
        在接下来的 test_months 个月使用最佳参数回测。
        使用交易日历确保切分点落在有效交易日期上。

        Args:
            strategy_class: 策略类
            data: 回测数据
            symbols: 交易合约代码列表
            train_months: 训练窗口月数
            test_months: 测试窗口月数
            config: 策略配置
            progress_callback: 进度回调 callback(current, total_windows)

        Returns:
            滚动优化结果 DataFrame
        """
        if config is None:
            config = StrategyConfig(initial_cash=1_000_000)

        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        all_dates = sorted(df["date"].unique())

        if len(all_dates) < 2:
            logger.warning("数据不足，无法执行滚动优化")
            return pd.DataFrame()

        start_date = all_dates[0]
        end_date = all_dates[-1]

        rolling_results = []
        current_start = start_date

        while current_start < end_date:
            train_end = current_start + pd.DateOffset(months=train_months)
            test_end = train_end + pd.DateOffset(months=test_months)

            if train_end > end_date:
                break

            train_dates = df[(df["date"] >= current_start) & (df["date"] < train_end)][
                "date"
            ]
            test_dates = df[(df["date"] >= train_end) & (df["date"] < test_end)]["date"]

            if train_dates.empty or test_dates.empty:
                logger.debug(f"跳过空窗口: train [{current_start}, {train_end})")
                current_start = train_end
                continue

            train_data = df[df["date"].isin(train_dates)]
            test_data = df[df["date"].isin(test_dates)]

            grid_results = self.grid_search(strategy_class, train_data, symbols, config)

            if grid_results.empty:
                current_start = train_end
                continue

            best_params = grid_results.iloc[0]
            param_cols = list(self.param_grid.keys())
            best_param_dict = {
                k: best_params[k] for k in param_cols if k in best_params.index
            }

            try:
                strategy_instance = strategy_class(**best_param_dict)
                indicators = []
                if hasattr(strategy_instance, "register_indicators"):
                    indicators = strategy_instance.register_indicators()

                test_metrics = self._run_single_backtest(
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
                logger.error(f"滚动优化测试失败: {e}")

            current_start = train_end

        total_windows = len(rolling_results)
        if progress_callback and total_windows > 0:
            progress_callback(total_windows, total_windows)

        return pd.DataFrame(rolling_results)

    def get_best_params(self) -> Optional[Dict]:
        """
        获取最佳参数组合。

        Returns:
            最佳参数字典，或 None
        """
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
        """
        保存优化结果到 JSON 文件。

        Args:
            filepath: 保存路径
        """
        if not self.results:
            return

        serializable = []
        for r in self.results:
            item = {}
            for k, v in r.items():
                if isinstance(v, (np.integer,)):
                    item[k] = int(v)
                elif isinstance(v, (np.floating,)):
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
        """
        从 JSON 文件加载优化结果。

        Args:
            filepath: 文件路径

        Returns:
            优化结果列表

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: 文件格式错误
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"优化结果文件不存在: {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"优化结果文件格式错误: {filepath}", e.doc, e.pos
            ) from e
