"""
Pipeline 因子编排（2026-06-11 拆分：规则7 文件行数约束）。

Pipeline 主体（runner/pipeline.py）只保留 Pipeline 类 + 编排入口；
因子相关编排（screen_factors / review_factors）下沉到本模块。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from loguru import logger

from core.config import BacktestConfig
from core.config.strategy_profiles import StrategyLibrary
from runner.common.errors import PipelineError


def _run_factor_screening(
    config: BacktestConfig,
    data,
    lib: StrategyLibrary,
    symbols: Optional[List[str]],
    do_winsorize: bool,
) -> Any:
    """
    因子筛选：对AlphaFutures全部因子做IC/IR统计测试。

    委托 runner.validation.factor_alpha24_screening（其内部已用
    FactorEvaluator.evaluate_batch 批量评估），筛选出通过规则9
    （|IC|>0.03且|IR|>0.5）的有效因子。
    """
    if data is None:
        raise PipelineError("请先调用 load_data() 加载数据")

    if symbols is None:
        symbols = config.symbols

    logger.info("=" * 60)
    logger.info("因子筛选: AlphaFutures 因子 IC/IR 测试")
    logger.info(f"  品种: {symbols}")
    logger.info("=" * 60)

    from runner.validation.factor_alpha24 import factor_alpha24_screening

    return factor_alpha24_screening(
        data_source=data,
        config=config,
        lib=lib,
        output_dir=Path(config.output_dir),
        do_winsorize=do_winsorize,
    )


def _run_factor_review(
    config: BacktestConfig,
    data,
    lib: StrategyLibrary,
    symbols: Optional[List[str]],
) -> Any:
    """
    因子复核：对全部因子执行6项质量检查。

    委托 runner.validation.factor_review.factor_review_validation，
    执行数据存活率、缺失值占比、异常值抵抗、参数敏感性、
    因子正交性、时序稳定性共6项复核。
    """
    if data is None:
        raise PipelineError("请先调用 load_data() 加载数据")

    if symbols is None:
        symbols = config.symbols

    logger.info("=" * 60)
    logger.info("因子复核: 6 项质量检查")
    logger.info(f"  品种: {symbols}")
    logger.info("=" * 60)

    from runner.validation.factor_review import factor_review_validation

    return factor_review_validation(
        data_source=data,
        config=config,
        lib=lib,
        output_dir=Path(config.output_dir),
    )
