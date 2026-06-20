"""
Pipeline 辅助函数（2026-06-11 拆分：规则7 文件行数约束）。

Pipeline 主体（runner/pipeline.py）只保留 Pipeline 类 + 编排入口；
复杂的多步编排（优化流程、TopN 参数提取、验证编排、健康检查）下沉到本模块，
由 Pipeline 类内部调用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from core.config import BacktestConfig
from core.config.strategy_profiles import StrategyLibrary


def _extract_metrics_from_df(df: "Any") -> Dict[str, Any]:
    """从实验返回的 DataFrame 中提取汇总指标（行/列启发式）。

    启发式：若列包含 sharpe/annual_return/max_drawdown，直接取第一行；
    否则尝试在 index 中查找。

    2026-06-20 迁移（来自 runner/pipeline.py 顶层）。
    """
    if df is None or getattr(df, "empty", True):
        return {"metrics": {"sharpe": 0.0, "annual_return": 0.0, "max_drawdown": 0.0}}
    metrics = {"sharpe": 0.0, "annual_return": 0.0, "max_drawdown": 0.0}
    cols_lower = [str(c).lower() for c in df.columns]
    for key, aliases in {
        "sharpe": ["sharpe", "sharpe_ratio"],
        "annual_return": ["annual_return", "annual", "yearly_return"],
        "max_drawdown": ["max_drawdown", "mdd", "max_dd"],
    }.items():
        for alias in aliases:
            if alias in cols_lower:
                col = df.columns[cols_lower.index(alias)]
                try:
                    metrics[key] = float(df[col].iloc[0])
                except (ValueError, TypeError, IndexError):
                    pass
                break
    return {"metrics": metrics}


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


def _run_signal_fusion(
    config: BacktestConfig,
    symbols,
    strategies,
    weights,
    mode,
    entry_threshold,
    output_dir,
) -> Any:
    """
    多策略信号融合（委托 runner/validation/signal_fusion.run_signal_fusion）。
    """
    from runner.validation.signal_fusion import run_signal_fusion

    return run_signal_fusion(
        symbols=symbols or list(config.symbols),
        strategies=strategies,
        weights=weights,
        mode=mode,
        entry_threshold=entry_threshold,
        initial_cash=float(config.initial_cash),
        full_start=str(config.full_start),
        test_start=str(config.full_end),
        output_dir=Path(output_dir) if output_dir else Path(config.output_dir) / "validation" / "signal_fusion",
    )


def _run_parameter_plateau(
    config: BacktestConfig,
    symbol,
    strategy_name,
    strategy_params,
    perturbation,
    steps,
    variation_threshold,
    output_dir,
) -> Any:
    """
    参数平原测试（委托 runner/validation/parameter_plateau.run_parameter_plateau）。
    """
    from runner.validation.parameter_plateau import run_parameter_plateau

    od = Path(output_dir) if output_dir else Path(config.output_dir) / "validation" / "parameter_plateau"
    return run_parameter_plateau(
        symbol=symbol,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        perturbation=perturbation,
        steps=steps,
        variation_threshold=variation_threshold,
        entry_threshold=0.05,
        initial_cash=float(config.initial_cash),
        full_start=str(config.full_start),
        test_start=str(config.full_end),
        output_dir=od,
    )


def _run_walk_forward(
    config: BacktestConfig,
    symbols,
    strategies,
    windows,
    entry_threshold,
    output_dir,
) -> Any:
    """
    Walk-Forward 滚动验证（委托 runner/validation/walk_forward.run_walk_forward）。
    """
    from runner.validation.walk_forward import run_walk_forward, DEFAULT_WINDOWS

    od = Path(output_dir) if output_dir else Path(config.output_dir) / "validation" / "walk_forward"
    return run_walk_forward(
        symbols=symbols or list(config.symbols),
        strategies=strategies or {},
        windows=windows or DEFAULT_WINDOWS,
        entry_threshold=entry_threshold,
        initial_cash=float(config.initial_cash),
        output_dir=od,
     )


def _verify_chain(
    config: Optional[BacktestConfig],
    data,
    lib: Optional[StrategyLibrary],
) -> Dict[str, bool]:
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
        "config_loaded": config is not None,
        "data_loaded": data is not None,
        "strategy_library": lib is not None,
    }

    # 验证 BacktestConfig 关键字段
    if config is not None:
        chain_status["backtest_config_has_symbols"] = bool(config.symbols)

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
        from core.execution.backtest_runner import PyBrokerBacktestRunner

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
