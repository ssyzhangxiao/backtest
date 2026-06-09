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

# 导入所有因子（自动注册到 _FACTOR_REGISTRY）
# 通过 factors 子包的 __init__.py 一次性触发所有因子的 @register_factor 装饰器
from . import factors as _factors_pkg  # noqa: F401  (仅触发子包 import)

# 显式从子包导出全部因子类，保证 `from core.factors.alpha_futures import T_02` 可用
# 因子数量以 list_available_factors() 动态为准，但显式列出让 IDE 自动补全和静态检查可用
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
]

# 显式 re-export 所有已注册因子，避免下游需要 `from .alpha_futures.factors import T_02`
# 这里使用 `from .factors import <name>` 形式：factors 子包已在上面被导入，类对象直接可访问
for _name in list_available_factors():
    globals()[_name] = _factors_pkg.__dict__.get(_name)
    if _name not in globals():
        # 防御性 fallback：若 factors 子包未显式 export（不会发生，但保安全）
        exec(f"from .factors import {_name}", globals())  # noqa: S102
    __all__.append(_name)

# 去重 __all__（防御性）
__all__ = list(dict.fromkeys(__all__))
