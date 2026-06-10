"""
Pipeline 编排器。

声明式链式调用回测流程：
  Pipeline("config.yaml").load_data().run_backtest("e1").report()

详见规则18。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from core.config import BacktestConfig
from core.config.strategy_profiles import StrategyLibrary
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

    @property
    def config(self) -> BacktestConfig:
        """获取结构化配置。"""
        return self._config

    @property
    def results(self) -> Dict[str, Any]:
        """获取已执行的结果。"""
        return self._results

    def reload_config(self) -> "Pipeline":
        """
        从磁盘重新加载配置文件（热重载）。

        注意：已加载的数据和结果不会被重置，只有配置会更新。

        Returns:
            self（支持链式调用）
        """
        logger.info(f"重新加载配置: {self._config_path}")
        self._config = BacktestConfig.from_yaml(self._config_path)
        # 同时重新加载原始配置
        if self._data is not None:
            from runner.data.loader import DataLoader
            loader = DataLoader(self._config_path)
            self._raw_config = loader.raw_config
        logger.info("配置热重载完成")
        return self



    def load_data(self) -> "Pipeline":
        """
        加载数据（委托 runner/data/loader）。

        同时加载原始配置。

        Returns:
            self（支持链式调用）
        """
        from runner.data.loader import DataLoader

        loader = DataLoader(self._config_path)
        self._data = loader.load()
        self._raw_config = loader.raw_config
        logger.info("数据加载完成")
        return self

    def run_backtest(
        self,
        experiment: str = "all",
        cross_sectional: bool = True,
        strategy: Optional[str] = None,
    ) -> "Pipeline":
        """
        执行指定实验（委托 runner/backtest/experiments/）。

        Args:
            experiment: 实验名称（"e1"~"e11" 或 "all"）
            cross_sectional: 是否启用多策略横截面打分模式
            strategy: 指定策略名称，None 表示自动选择

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
            cross_sectional=cross_sectional,
            strategy=strategy,
        )
        return self

    def run_experiments(
        self,
        experiments: List[str],
        cross_sectional: bool = False,
    ) -> "Pipeline":
        """
        批量执行多个实验（委托 runner/backtest/experiments.run_experiment）。

        用于在一次调用中按顺序跑 E1~E11 中指定子集，逐个打印指标摘要。

        Args:
            experiments: 实验名称列表，如 ["e1", "e2", "e11", "e9"]
            cross_sectional: 是否启用多策略横截面打分模式

        Returns:
            self（支持链式调用）
        """
        if self._data is None:
            raise PipelineError("请先调用 load_data() 加载数据")
        from runner.backtest.experiments import run_experiment

        for name in experiments:
            logger.info("[%s] 实验开始", name)
            self._results[name] = run_experiment(
                name,
                self._config,
                self._data,
                self._raw_config,
                cross_sectional=cross_sectional,
            )
            logger.info("[%s] 实验完成", name)
        return self

    def multi_oos(
        self,
        output_dir: Optional[Path] = None,
        strategies: Optional[List[str]] = None,
        windows: Optional[List[Tuple[str, str, str]]] = None,
        best_params: Optional[Dict[str, Dict[str, Any]]] = None,
        save_json: bool = True,
    ) -> "Pipeline":
        """
        多窗口 OOS 验证（委托 runner/validation/multi_oos.run_multi_oos）。

        对 5 子策略在多个 OOS 窗口内回测，提取 Sharpe/Return/MaxDD/Trades，
        并计算等权组合的 Sharpe 平均值。

        Args:
            output_dir: 输出目录，默认 config.yaml 中 output_dir/validation/multi_oos
            strategies: 子策略列表，默认 5 子策略
            windows: (窗口名, 起始日, 结束日) 元组列表
            best_params: 优化后的最优参数
            save_json: 是否保存 JSON 汇总

        Returns:
            self（支持链式调用），结果在 self._results["multi_oos"]
        """
        if self._data is None:
            raise PipelineError("请先调用 load_data() 加载数据")
        from runner.validation.multi_oos import run_multi_oos as _run_multi_oos

        if output_dir is None:
            output_dir = Path(self._config.output_dir) / "validation" / "multi_oos"
        self._results["multi_oos"] = _run_multi_oos(
            data_source=self._data,
            config=self._config,
            output_dir=Path(output_dir),
            strategies=strategies,
            windows=windows,
            best_params=best_params,
            save_json=save_json,
        )
        return self

    def full_validation(
        self,
        in_sample_start: str = "2020-01-01",
        in_sample_end: str = "2023-01-01",
        oos_start: str = "2023-01-01",
        oos_end: str = "2024-12-31",
        full_start: str = "2020-01-01",
        full_end: str = "2024-12-31",
        strategies: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
    ) -> "Pipeline":
        """
        全量验证 3 阶段流水线（委托 runner/validation/full_validation）。

        Phase 1: in_sample 调参（grid + OOS 优选）
        Phase 2: 6 品种 × 5 子策略 × {TRAIN, OOS} 窗口 EW 组合回测
        Phase 3: 全段蒙特卡洛 1000 次鲁棒性测试

        Args:
            in_sample_start: 训练区间起始
            in_sample_end: 训练区间结束
            oos_start: OOS 区间起始
            oos_end: OOS 区间结束
            full_start: 全段起始（蒙特卡洛用）
            full_end: 全段结束（蒙特卡洛用）
            strategies: 子策略列表，默认 5 子策略
            output_dir: 输出目录

        Returns:
            self（支持链式调用），结果在 self._results["full_validation"]
        """
        if self._data is None:
            raise PipelineError("请先调用 load_data() 加载数据")
        from runner.validation.full_validation import run_full_validation as _run_fv

        if output_dir is None:
            output_dir = Path("output_backtest_pybroker/full_validation")
        self._results["full_validation"] = _run_fv(
            pipe=self,
            in_sample_start=in_sample_start,
            in_sample_end=in_sample_end,
            oos_start=oos_start,
            oos_end=oos_end,
            full_start=full_start,
            full_end=full_end,
            strategies=strategies,
            output_dir=Path(output_dir),
        )
        return self

    def optimize(
        self,
        strategy: Optional[str] = None,
        tasks: Optional[List[str]] = None,
        symbol: Optional[str] = None,
        save_to_config: bool = True,
    ) -> "Pipeline":
        """
        参数优化（委托 runner/optimization/）。

        执行网格搜索、窗口搜索和样本外优先选择。

        Args:
            strategy: 指定策略名称，None 表示全部策略
            tasks: 优化任务列表，默认 ["grid", "window", "oos"]
            symbol: 指定品种代码，None 表示全部品种
            save_to_config: 是否自动保存优化后的参数到 config.yaml

        Returns:
            self（支持链式调用）
        """
        if self._data is None:
            raise PipelineError("请先调用 load_data() 加载数据")

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
            self._config,
        )
        self._results["optimization"] = opt_results

        # 自动保存优化后的参数到 config.yaml
        if save_to_config and "best_params" in opt_results:
            best_params = opt_results["best_params"]
            if best_params:
                logger.info("自动保存优化后的参数到 config.yaml")
                self._config.update_strategy_params(best_params, self._config_path)
                logger.info("参数保存成功！")

        return self

    def validate(
        self,
        method: str = "train_test",
        best_params: Optional[Dict[str, Dict[str, Any]]] = None,
        cross_sectional: bool = False,
    ) -> "Pipeline":
        """
        验证（委托 runner/validation/ + core/validation/）。

        Args:
            method: 验证方法名称
                - "train_test": 训练/测试分割验证
                - "monte_carlo": 蒙特卡洛鲁棒性测试
                - "bootstrap": Bootstrap置信区间
                - "factor_ic": 因子IC稳定性分析
                - "factor_alpha24": AlphaFutures24因子IC/IR验证
                - "factor_review": 因子6项复核
                - "cross_sectional": 多策略横截面打分验证
                - "all": 执行全部验证方法
            best_params: 优化后的最优参数
            cross_sectional: 是否启用多策略横截面打分模式

        Returns:
            self（支持链式调用）
        """
        if self._data is None:
            raise PipelineError("请先调用 load_data() 加载数据")

        output_dir = Path(self._config.output_dir) / "validation"
        output_dir.mkdir(parents=True, exist_ok=True)

        if method.lower() == "all":
            self._results["validation"] = _run_all_validations(
                self._data,
                self._lib,
                self._config,
                output_dir,
                best_params,
                cross_sectional=cross_sectional,
            )
        else:
            from runner.validation import get_validator

            validator = get_validator(method)
            self._results["validation"] = validator(
                self._data,
                self._config,
                self._lib,
                output_dir,
                best_params=best_params,
                cross_sectional=cross_sectional,
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
        output_dir = Path(self._config.output_dir)

        from runner.report import generate

        generate(fmt, self._results, self._config, output_dir)
        return self

    def screen_factors(
        self,
        symbols: Optional[List[str]] = None,
        do_winsorize: bool = True,
    ) -> "Pipeline":
        """
        因子筛选：对AlphaFutures全部30因子做IC/IR统计测试。

        委托 runner.validation.factor_alpha24_screening（其内部已用
        FactorEvaluator.evaluate_batch 批量评估），筛选出通过规则9
        （|IC|>0.03且|IR|>0.5）的有效因子。

        Args:
            symbols: 测试品种列表，默认使用配置中的品种
            do_winsorize: 是否对因子值做缩尾后处理

        Returns:
            self（支持链式调用）
        """
        if self._data is None:
            raise PipelineError("请先调用 load_data() 加载数据")

        if symbols is None:
            symbols = self._config.symbols

        logger.info("=" * 60)
        logger.info("因子筛选: AlphaFutures 30因子IC/IR测试")
        logger.info(f"  品种: {symbols}")
        logger.info("=" * 60)

        from runner.validation.factor_alpha24 import factor_alpha24_screening

        results = factor_alpha24_screening(
            data_source=self._data,
            config=self._config,
            lib=self._lib,
            output_dir=Path(self._config.output_dir),
            do_winsorize=do_winsorize,
        )
        self._results["factor_screening"] = results
        return self

    def review_factors(
        self,
        symbols: Optional[List[str]] = None,
    ) -> "Pipeline":
        """
        因子复核：对全部因子执行6项质量检查。

        委托 runner.validation.factor_review.factor_review_validation，
        执行数据存活率、缺失值占比、异常值抵抗、参数敏感性、
        因子正交性、时序稳定性共6项复核。

        Args:
            symbols: 复核品种列表，默认使用配置中的品种

        Returns:
            self（支持链式调用）
        """
        if self._data is None:
            raise PipelineError("请先调用 load_data() 加载数据")

        if symbols is None:
            symbols = self._config.symbols

        logger.info("=" * 60)
        logger.info("因子复核: 6项质量检查")
        logger.info(f"  品种: {symbols}")
        logger.info("=" * 60)

        from runner.validation.factor_review import factor_review_validation

        results = factor_review_validation(
            data_source=self._data,
            config=self._config,
            lib=self._lib,
            output_dir=Path(self._config.output_dir),
        )
        self._results["factor_review"] = results
        return self

    def with_config(self, **overrides: Any) -> "Pipeline":
        """
        配置热更新，返回新实例。

        不修改当前实例，确保线程安全。**overrides 必须是 BacktestConfig
        合法字段**（如 initial_cash、symbols、train_start 等），任意键会
        在 dataclasses.replace 时抛 TypeError，便于及时发现拼写错误。

        Args:
            **overrides: BacktestConfig 字段覆盖项

        Returns:
            新的 Pipeline 实例
        """
        from dataclasses import replace

        new_config = replace(self._config, **overrides)
        new_pipe = Pipeline.__new__(Pipeline)
        new_pipe._config = new_config
        new_pipe._config_path = self._config_path
        new_pipe._raw_config = self._raw_config
        new_pipe._data = self._data
        new_pipe._results = dict(self._results)
        new_pipe._lib = self._lib
        return new_pipe

    def is_healthy(self) -> bool:
        """状态检查：数据已加载且无异常。"""
        return self._data is not None

    def verify_chain(self) -> Dict[str, bool]:
        """
        P0-任务5整改：验证完整调用链是否正确连接。

        调用链（自下而上）:
          因子层 FactorEngine
            → 评估层 FactorEvaluator
            → 子策略合成层 SubStrategyAdapter / PortfolioManager
            → 横截面打分层 FactorScoringEngine
            → 执行层 PyBrokerExecutorBuilder（蓝图模式）
            → 风控层 RiskController

        Returns:
            {组件名: 是否就位}
        """
        chain_status: Dict[str, bool] = {
            "config_loaded": self._config is not None,
            "data_loaded": self._data is not None,
            "strategy_library": self._lib is not None,
        }

        # 验证 BacktestConfig 关键字段
        # P2 整改：删除 `backtest_config_has_factor_weights` 检查。
        # 原因：factor_weights 是 Dict[str, float]，合法状态包含空字典（用户未启用任何
        # 子策略时，default_factory=dict 即为空）。`bool({})` 与 `bool(None)` 均为 False，
        # 会产生"未配置"的误导性告警；改用 `bool(self._config.factor_weights)` 仍无法
        # 区分"空字典（合法）"与"字段缺失（异常）"。该字段的存在性已在 BacktestConfig
        # dataclass 定义处保证，无需在 health check 中重复验证。
        if self._config is not None:
            chain_status["backtest_config_has_symbols"] = bool(self._config.symbols)

        # 验证核心模块可导入且存在
        try:
            from core.engine.switch_engine import FactorScoringEngine
            chain_status["factor_scoring_engine"] = FactorScoringEngine is not None
        except ImportError:
            chain_status["factor_scoring_engine"] = False

        try:
            from core.portfolio import PortfolioManager
            chain_status["portfolio_manager"] = PortfolioManager is not None
        except ImportError:
            chain_status["portfolio_manager"] = False

        try:
            from core.engine.pybroker_data_source import create_hybrid_data_source
            chain_status["hybrid_data_source"] = create_hybrid_data_source is not None
        except ImportError:
            chain_status["hybrid_data_source"] = False

        try:
            from core.engine.backtest_runner import PyBrokerBacktestRunner
            chain_status["pybroker_runner"] = PyBrokerBacktestRunner is not None
        except ImportError:
            chain_status["pybroker_runner"] = False

        # 验证数据加载策略：TqSdk 优先，CSV 仅用于 spread
        try:
            import inspect
            from core.engine.pybroker_data_source import create_hybrid_data_source
            source = inspect.getsource(create_hybrid_data_source)
            chain_status["tqsdk_primary"] = "TqSdk" in source and "禁止静默回退" in source
            chain_status["csv_only_for_spread"] = "spread" in source
        except Exception:
            chain_status["tqsdk_primary"] = False
            chain_status["csv_only_for_spread"] = False

        healthy = all(chain_status.values())
        if not healthy:
            failed = [k for k, v in chain_status.items() if not v]
            logger.warning("调用链存在未就位组件: %s", failed)
        else:
            logger.info("完整调用链验证通过: %s", list(chain_status.keys()))

        return chain_status


def _run_optimization(
    strategy: Optional[str],
    tasks: List[str],
    data,
    lib: StrategyLibrary,
    config: BacktestConfig,
) -> Dict[str, Any]:
    """
    执行参数优化流程。

    Args:
        strategy: 策略名称
        tasks: 优化任务列表
        data: 数据源
        lib: 策略库
        config: 回测配置

    Returns:
        优化结果字典
    """
    from runner.strategy.selector import get_param_spaces
    from runner.optimization.grid_search import grid_search_single_strategy
    from runner.optimization.window_search import window_search_single_strategy

    strategy_names = config.strategy_names
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
                config,
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
                        config,
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
    config: BacktestConfig,
    output_dir: Path,
    best_params: Optional[Dict[str, Dict[str, Any]]],
    cross_sectional: bool = False,
) -> Dict[str, Any]:
    """
    执行全部验证方法（委托 runner.validation._VALIDATOR_MAP）。

    Args:
        data: 数据源
        lib: 策略库
        config: 回测配置
        output_dir: 输出目录
        best_params: 优化参数
        cross_sectional: 是否启用多策略横截面打分模式

    Returns:
        全部验证结果
    """
    from runner.validation import _VALIDATOR_MAP

    common_kwargs = {
        "best_params": best_params,
        "cross_sectional": cross_sectional,
    }
    results: Dict[str, Any] = {}
    for name, fn in _VALIDATOR_MAP.items():
        logger.info("=" * 60)
        logger.info(f"验证: {name}")
        results[name] = fn(data, config, lib, output_dir, **common_kwargs)
    return results

