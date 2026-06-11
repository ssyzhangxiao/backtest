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
from runner.pipeline_factor_ops import (
    _run_factor_review,
    _run_factor_screening,
)
from runner.pipeline_helpers import (
    _run_all_validations,
    _run_optimization,
    _verify_chain,
)


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

    def __init__(
        self,
        config_path: str = "config.yaml",
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化 Pipeline。

        Args:
            config_path: 配置文件路径
            overrides: 运行时覆盖（最高优先级），支持：
                - 顶层字段: {"rebalance_days": 5}
                - 段路径: {"backtest__rebalance_days": 5}
                - 嵌套 dict: {"backtest": {"rebalance_days": 5}}
                优先级: dataclass默认 < YAML < 环境变量QUANT_* < overrides
        """
        try:
            self._config = BacktestConfig.from_yaml(config_path, overrides=overrides)
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
        因子筛选（委托 runner.pipeline_factor_ops._run_factor_screening）。

        Returns:
            self（支持链式调用）
        """
        results = _run_factor_screening(
            config=self._config,
            data=self._data,
            lib=self._lib,
            symbols=symbols,
            do_winsorize=do_winsorize,
        )
        self._results["factor_screening"] = results
        return self

    def review_factors(
        self,
        symbols: Optional[List[str]] = None,
    ) -> "Pipeline":
        """
        因子复核（委托 runner.pipeline_factor_ops._run_factor_review）。

        Returns:
            self（支持链式调用）
        """
        results = _run_factor_review(
            config=self._config,
            data=self._data,
            lib=self._lib,
            symbols=symbols,
        )
        self._results["factor_review"] = results
        return self

    def with_config(self, **overrides: Any) -> "Pipeline":
        """
        配置热更新，返回新实例。

        不修改当前实例，确保线程安全。支持两种覆盖格式：
        1. BacktestConfig 合法字段（如 initial_cash、symbols）— 直接 replace
        2. 分层路径格式（如 backtest__rebalance_days）— 重新 from_yaml

        Args:
            **overrides: BacktestConfig 字段覆盖项或分层路径

        Returns:
            新的 Pipeline 实例
        """
        from dataclasses import replace

        # 检测是否有分层路径格式（含 __ 分隔符）
        has_layered_key = any("__" in k for k in overrides)

        if has_layered_key:
            # 分层覆盖：重新 from_yaml 并传入 overrides
            new_config = BacktestConfig.from_yaml(
                self._config_path, overrides=overrides
            )
        else:
            # 直接字段覆盖
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

        委托给 runner.pipeline_helpers._verify_chain，避免 Pipeline 主体膨胀。
        """
        return _verify_chain(
            config=self._config,
            data=self._data,
            lib=self._lib,
        )
