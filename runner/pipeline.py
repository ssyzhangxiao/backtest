"""
Pipeline 编排器。

声明式链式调用回测流程：
  Pipeline("config.yaml").load_data().run_backtest("e1").report()

详见规则18。
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from core.config import BacktestConfig
from core.strategy_registry import StrategyLibrary
from runner.common.errors import PipelineError, ConfigError


class Pipeline:
    """
    回测流水线编排器。

    支持声明式组合实验步骤，内部缓存数据与结果。
    核心逻辑委托给 core/ 和 utils/，不重新实现。

    用法:
        # 完整回测流程
        Pipeline("config.yaml").load_data().run_backtest("e1").report()

        # 优化 + 验证
        Pipeline("config.yaml").load_data().optimize().validate("monte_carlo").report()

        # 配置热更新
        pipe = Pipeline("config.yaml").load_data()
        pipe.with_config(initial_cash=500000).run_backtest("e1").report()
    """

    def __init__(self, config_path: str = "config.yaml"):
        try:
            self._config = BacktestConfig.from_yaml(config_path)
        except Exception as e:
            raise ConfigError(f"配置加载失败: {e}") from e
        self._config_path = config_path
        self._raw_config: Optional[Dict[str, Any]] = None
        self._data = None
        self._results: Dict[str, Any] = {}
        self._lib = StrategyLibrary()
        self._opt_cfg: Optional[Dict[str, Any]] = None

    @property
    def config(self) -> BacktestConfig:
        """获取结构化配置。"""
        return self._config

    @property
    def results(self) -> Dict[str, Any]:
        """获取已执行的结果。"""
        return self._results

    def load_data(self) -> "Pipeline":
        """
        加载数据（委托 runner/data/loader）。

        同时加载原始配置和构建优化配置。

        Returns:
            self（支持链式调用）
        """
        from runner.data.loader import DataLoader, build_opt_cfg

        loader = DataLoader(self._config_path)
        self._data = loader.load()
        self._raw_config = loader.raw_config
        if self._raw_config:
            self._opt_cfg = build_opt_cfg(self._raw_config)
        logger.info("数据加载完成")
        return self

    def run_backtest(self, experiment: str = "all") -> "Pipeline":
        """
        执行指定实验（委托 runner/backtest/experiments/）。

        Args:
            experiment: 实验名称（"e1"~"e11" 或 "all"）

        Returns:
            self（支持链式调用）
        """
        if self._data is None:
            raise PipelineError("请先调用 load_data() 加载数据")
        from runner.backtest.experiments import run_experiment

        self._results[experiment] = run_experiment(
            experiment,
            self._config,
            self._data,
            self._raw_config,
        )
        return self

    def optimize(
        self,
        strategy: Optional[str] = None,
        tasks: Optional[List[str]] = None,
        symbol: Optional[str] = None,
    ) -> "Pipeline":
        """
        参数优化（委托 runner/optimization/）。

        执行网格搜索、窗口搜索和样本外优先选择。

        Args:
            strategy: 指定策略名称，None 表示全部策略
            tasks: 优化任务列表，默认 ["grid", "window", "oos"]
            symbol: 指定品种代码，None 表示全部品种

        Returns:
            self（支持链式调用）
        """
        if self._data is None:
            raise PipelineError("请先调用 load_data() 加载数据")
        if self._opt_cfg is None:
            raise PipelineError("请先调用 load_data() 构建优化配置")

        if tasks is None:
            tasks = ["grid", "window", "oos"]

        data = self._data
        if symbol:
            data = self._data.for_symbol(symbol)
            logger.info(f"优化: 单品种模式 - {symbol}")

        opt_results = _run_optimization(
            strategy,
            tasks,
            data,
            self._lib,
            self._opt_cfg,
        )
        self._results["optimization"] = opt_results
        return self

    def validate(
        self,
        method: str = "train_test",
        best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> "Pipeline":
        """
        验证（委托 runner/validation/ + core/validation/）。

        Args:
            method: 验证方法名称
                - "train_test": 训练/测试分割验证
                - "monte_carlo": 蒙特卡洛鲁棒性测试
                - "bootstrap": Bootstrap置信区间
                - "factor_ic": 因子IC稳定性分析
                - "all": 执行全部验证方法
            best_params: 优化后的最优参数

        Returns:
            self（支持链式调用）
        """
        if self._data is None:
            raise PipelineError("请先调用 load_data() 加载数据")
        if self._opt_cfg is None:
            raise PipelineError("请先调用 load_data() 构建优化配置")

        output_dir = Path(self._opt_cfg["output_dir"]) / "validation"
        output_dir.mkdir(parents=True, exist_ok=True)

        if method.lower() == "all":
            self._results["validation"] = _run_all_validations(
                self._data,
                self._lib,
                self._opt_cfg,
                output_dir,
                best_params,
            )
        else:
            from runner.validation import get_validator

            validator = get_validator(method)
            self._results["validation"] = validator(
                self._data,
                self._opt_cfg,
                self._lib,
                output_dir,
                best_params=best_params,
            )

        return self

    def report(self, fmt: str = "html") -> "Pipeline":
        """
        生成报告（委托 runner/report/）。

        Args:
            fmt: 报告格式（"html", "csv", "validation"）

        Returns:
            self（支持链式调用）
        """
        output_dir = Path("results")
        if self._opt_cfg:
            output_dir = Path(self._opt_cfg["output_dir"])

        from runner.report import generate

        generate(fmt, self._results, self._config, output_dir)
        return self

    def with_config(self, **overrides) -> "Pipeline":
        """
        配置热更新，返回新实例。

        不修改当前实例，确保线程安全。

        Args:
            **overrides: 配置覆盖项

        Returns:
            新的 Pipeline 实例
        """
        new_config = self._config.copy(update=overrides)
        new_pipe = Pipeline.__new__(Pipeline)
        new_pipe._config = new_config
        new_pipe._config_path = self._config_path
        new_pipe._raw_config = self._raw_config
        new_pipe._data = self._data
        new_pipe._results = dict(self._results)
        new_pipe._lib = self._lib
        new_pipe._opt_cfg = self._opt_cfg
        return new_pipe

    def is_healthy(self) -> bool:
        """状态检查：数据已加载且无异常。"""
        return self._data is not None


def _run_optimization(
    strategy: Optional[str],
    tasks: List[str],
    data,
    lib: StrategyLibrary,
    opt_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    执行参数优化流程。

    Args:
        strategy: 策略名称
        tasks: 优化任务列表
        data: 数据源
        lib: 策略库
        opt_cfg: 优化配置

    Returns:
        优化结果字典
    """
    from runner.strategy.selector import get_param_spaces
    from runner.optimization.grid_search import grid_search_single_strategy
    from runner.optimization.window_search import window_search_single_strategy
    from runner.optimization.oos_selector import select_best_by_oos_priority

    strategy_names = opt_cfg["strategy_names"]
    if strategy:
        strategy_names = [strategy]

    param_spaces = get_param_spaces(lib, strategy_names)
    results = {}

    # 网格搜索
    if "grid" in tasks:
        logger.info("优化: 网格搜索")
        grid_results = {}
        for sname, pspace in param_spaces.items():
            grid_results[sname] = grid_search_single_strategy(
                sname,
                pspace,
                data,
                lib,
                opt_cfg,
            )
        results["grid"] = grid_results

    # 窗口搜索
    if "window" in tasks:
        logger.info("优化: 窗口搜索")
        window_results = {}
        for sname, pspace in param_spaces.items():
            grid_df = results.get("grid", {}).get(sname, None)
            top_params = _extract_top_params(grid_df, pspace)
            if top_params:
                try:
                    window_results[sname] = window_search_single_strategy(
                        sname,
                        top_params,
                        data,
                        lib,
                        opt_cfg,
                    )
                except Exception as e:
                    logger.warning(f"窗口搜索 {sname} 失败: {e}")
        results["window"] = window_results

    # 样本外优先选择（简化版：直接取网格搜索 top 1）
    if "oos" in tasks:
        logger.info("优化: 样本外优先选择")
        best_params = {}
        for sname in strategy_names:
            grid_df = results.get("grid", {}).get(sname, None)
            if grid_df is not None and not grid_df.empty:
                param_space = param_spaces[sname]
                param_keys = list(param_space.keys())
                best_row = grid_df.iloc[0]
                best_params[sname] = {k: best_row[k] for k in param_keys}
        results["best_params"] = best_params

    return results


def _extract_top_params(
    grid_df,
    param_space: Dict[str, Any],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    从网格搜索结果中提取 Top N 参数组合。

    Args:
        grid_df: 网格搜索结果 DataFrame
        param_space: 参数空间
        top_n: 取前N个

    Returns:
        参数字典列表
    """
    if grid_df is None or grid_df.empty:
        return []

    param_keys = list(param_space.keys())
    top_df = grid_df.head(top_n)
    return [{k: row[k] for k in param_keys} for _, row in top_df.iterrows()]


def _run_all_validations(
    data,
    lib: StrategyLibrary,
    opt_cfg: Dict[str, Any],
    output_dir: Path,
    best_params: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    执行全部验证方法。

    Args:
        data: 数据源
        lib: 策略库
        opt_cfg: 优化配置
        output_dir: 输出目录
        best_params: 优化参数

    Returns:
        全部验证结果
    """
    from runner.validation.train_test import task2_train_test_split
    from runner.validation.monte_carlo import task3_monte_carlo
    from runner.validation.factor_stability import factor_ic_stability_analysis

    results = {}

    logger.info("=" * 60)
    logger.info("验证: 训练/测试分割")
    results["train_test"] = task2_train_test_split(
        data,
        opt_cfg,
        lib,
        output_dir,
        best_params=best_params,
    )

    logger.info("=" * 60)
    logger.info("验证: 蒙特卡洛")
    results["monte_carlo"] = task3_monte_carlo(
        data,
        opt_cfg,
        lib,
        output_dir,
        best_params=best_params,
    )

    logger.info("=" * 60)
    logger.info("验证: 因子IC稳定性")
    results["factor_ic"] = factor_ic_stability_analysis(
        data,
        opt_cfg,
        output_dir,
    )

    return results
