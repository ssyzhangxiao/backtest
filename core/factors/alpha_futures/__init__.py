"""
商品期货 Alpha 因子库 — 工程化重构版本。

基于抽象基类的独立因子类 + 注册表 + 引擎调度架构。
提供因子计算、评估、变换、筛选、复核一站式 Pipeline。
"""
from .base_factor import BaseFactor
from .factor_registry import (
    register_factor,
    get_factor,
    list_available_factors,
    get_factor_registry,
    SUB_STRATEGY_FACTOR_GROUPS,
    get_sub_strategy_factors,
)
from .factor_engine import FactorEngine
from .factor_pipeline import FactorPipeline, PipelineResult
from .sub_strategy_aggregator import compute_sub_strategy_scores_from_ohlcv

# 导入所有因子（自动注册）
from .factors import T_01, R_01, H_05, CF_01, TS_01

__all__ = [
    "BaseFactor",
    "register_factor",
    "get_factor",
    "list_available_factors",
    "get_factor_registry",
    "SUB_STRATEGY_FACTOR_GROUPS",
    "get_sub_strategy_factors",
    "FactorEngine",
    "FactorPipeline",
    "PipelineResult",
    "compute_sub_strategy_scores_from_ohlcv",
    "T_01",
    "R_01",
    "H_05",
    "CF_01",
    "TS_01",
]
