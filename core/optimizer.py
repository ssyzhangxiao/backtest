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
from typing import Dict, List, Optional, Callable, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from pybroker import Strategy, StrategyConfig
else:
    try:
        from pybroker import Strategy, StrategyConfig
    except ImportError:
        Strategy = None  # type: ignore
        StrategyConfig = None  # type: ignore

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
                    self.results.append(result)

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

    def walkforward_optimize(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        runner_factory: Callable,
        train_years: int = 2,
        test_years: int = 1,
        metric: str = "sharpe",
    ) -> pd.DataFrame:
        """
        Walk-Forward 参数优化。

        将数据分为 N 个连续窗口（train_years 年训练 + test_years 年验证），
        每个窗口内做网格搜索，选择最优参数，在下一个窗口验证。

        最终参数选择：投票法 + 稳健性评分。

        Args:
            data: 回测数据
            symbols: 交易合约代码列表
            runner_factory: 创建 runner 的工厂函数 runner_factory(strategies) -> runner
            train_years: 训练窗口年数
            test_years: 验证窗口年数
            metric: 优化目标指标

        Returns:
            Walk-Forward 结果 DataFrame
        """
        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        all_dates = sorted(df["date"].unique())

        if len(all_dates) < 2:
            logger.warning("数据不足，无法执行 Walk-Forward 优化")
            return pd.DataFrame()

        start_date = all_dates[0]
        end_date = all_dates[-1]

        wf_results = []
        current_start = start_date

        while current_start < end_date:
            train_end = current_start + pd.DateOffset(years=train_years)
            test_end = train_end + pd.DateOffset(years=test_years)

            if train_end > end_date:
                break

            train_data = df[(df["date"] >= current_start) & (df["date"] < train_end)]
            test_data = df[(df["date"] >= train_end) & (df["date"] < test_end)]

            if train_data.empty or test_data.empty:
                current_start = train_end
                continue

            combos = self._generate_param_combinations()
            train_results = []

            for params in combos:
                try:
                    runner = runner_factory(list(self.param_grid.keys()))
                    train_start_str = str(current_start.date())
                    train_end_str = str(train_end.date())
                    param_dict = {list(self.param_grid.keys())[0]: params}
                    result = runner.run(
                        start_date=train_start_str,
                        end_date=train_end_str,
                        custom_params=param_dict,
                    )
                    m = result.metrics
                    m = {
                        k: float(v)
                        if isinstance(v, (int, float, np.integer, np.floating))
                        else v
                        for k, v in m.items()
                    }
                    m.update(params)
                    train_results.append(m)
                except Exception as e:
                    logger.debug("参数组合 %s 回测失败: %s", params, e)
                    continue

            if not train_results:
                current_start = train_end
                continue

            train_df = pd.DataFrame(train_results)
            if metric not in train_df.columns:
                current_start = train_end
                continue

            robust_score = self._compute_robustness(train_df, metric)
            train_df["robust_score"] = train_df.index.map(robust_score)
            best_idx = train_df["robust_score"].idxmax()
            best_params = train_df.loc[best_idx]
            param_cols = list(self.param_grid.keys())
            best_param_dict = {
                k: best_params[k] for k in param_cols if k in best_params.index
            }

            try:
                runner = runner_factory(list(self.param_grid.keys()))
                test_start_str = str(train_end.date())
                test_end_str = str(test_end.date())
                param_dict = {list(self.param_grid.keys())[0]: best_param_dict}
                test_result = runner.run(
                    start_date=test_start_str,
                    end_date=test_end_str,
                    custom_params=param_dict,
                )
                test_m = test_result.metrics
                test_m = {
                    k: float(v)
                    if isinstance(v, (int, float, np.integer, np.floating))
                    else v
                    for k, v in test_m.items()
                }
                record = {
                    "train_start": str(current_start.date()),
                    "train_end": str(train_end.date()),
                    "test_start": str(train_end.date()),
                    "test_end": str(test_end.date()),
                    **best_param_dict,
                    "train_sharpe": best_params.get(metric, 0),
                    "train_robust_score": best_params.get("robust_score", 0),
                    **test_m,
                }
                wf_results.append(record)
            except Exception as e:
                logger.debug("滚动窗口回测失败: %s", e)

        if not wf_results:
            return pd.DataFrame()

        wf_df = pd.DataFrame(wf_results)
        return wf_df

    def _compute_robustness(
        self, results_df: pd.DataFrame, metric: str
    ) -> Dict[int, float]:
        """
        计算每个参数组合的稳健性评分。

        robust_score = 0.7 * sharpe + 0.3 * neighborhood_avg_sharpe

        邻域定义：参数空间中相邻的参数组合（仅一个参数变化一步）。

        Args:
            results_df: 网格搜索结果 DataFrame
            metric: 目标指标

        Returns:
            {index: robust_score} 字典
        """
        param_cols = list(self.param_grid.keys())
        scores = {}

        if metric not in results_df.columns:
            return {i: 0.0 for i in results_df.index}

        metric_vals = results_df[metric].values

        for idx in results_df.index:
            own_sharpe = metric_vals[idx] if not np.isnan(metric_vals[idx]) else 0.0
            neighbor_sharpes = []

            row = results_df.loc[idx]
            for col in param_cols:
                if col not in results_df.columns:
                    continue
                current_val = row[col]
                for direction in [-1, 1]:
                    param_values = self.param_grid.get(col, [])
                    if not param_values:
                        continue
                    try:
                        current_idx_pos = param_values.index(current_val)
                    except (ValueError, AttributeError):
                        continue
                    neighbor_idx = current_idx_pos + direction
                    if 0 <= neighbor_idx < len(param_values):
                        neighbor_val = param_values[neighbor_idx]
                        neighbor_rows = results_df[results_df[col] == neighbor_val]
                        if not neighbor_rows.empty:
                            neighbor_mean = neighbor_rows[metric].mean()
                            if not np.isnan(neighbor_mean):
                                neighbor_sharpes.append(neighbor_mean)

            neighbor_avg = np.mean(neighbor_sharpes) if neighbor_sharpes else own_sharpe
            robust = 0.7 * own_sharpe + 0.3 * neighbor_avg
            scores[idx] = robust

        return scores

    @staticmethod
    def select_robust_params(wf_df: pd.DataFrame, param_cols: List[str]) -> Dict:
        """
        从 Walk-Forward 结果中选择稳健参数（投票法）。

        选择所有窗口中出现频率最高的参数组合，
        若无重复则选择平均 robust_score 最高的参数。

        Args:
            wf_df: Walk-Forward 结果 DataFrame
            param_cols: 参数列名列表

        Returns:
            最佳参数字典
        """
        if wf_df.empty:
            return {}

        param_combos = wf_df[param_cols].apply(lambda row: tuple(row.values), axis=1)
        combo_counts = param_combos.value_counts()

        if combo_counts.iloc[0] > 1:
            best_combo = combo_counts.index[0]
            return dict(zip(param_cols, best_combo))

        if "train_robust_score" in wf_df.columns:
            best_idx = wf_df["train_robust_score"].idxmax()
        elif "train_sharpe" in wf_df.columns:
            best_idx = wf_df["train_sharpe"].idxmax()
        else:
            best_idx = 0

        return {col: wf_df.loc[best_idx, col] for col in param_cols}
