"""
商品期货 Alpha 因子库 — 工程化重构版本。

基于抽象基类的独立因子类 + 注册表 + 引擎调度架构。
提供因子计算、评估、变换、筛选、复核一站式 Pipeline。
"""
import importlib
import pkgutil
from pathlib import Path

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

# ──────────────────────────────────────────────
# 动态扫描 factors/ 子包：自动导入 + 导出全部因子类
# ──────────────────────────────────────────────
# 1) 触发 factors 子包导入，副作用：所有 @register_factor 装饰器执行
_factors_pkg = importlib.import_module(".factors", __name__)

# 2) 扫描 factors/ 目录下的所有非 _ 前缀的子模块
#    （确保未来新增因子类无需手动维护导出列表，IDE 自动补全仍可用）
_factors_dir = Path(_factors_pkg.__file__).parent
_loaded_factor_classes: list = []
for _module_info in pkgutil.iter_modules([str(_factors_dir)]):
    _name = _module_info.name
    if _name.startswith("_"):
        continue
    _mod = importlib.import_module(f".factors.{_name}", __name__)
    for _attr_name in dir(_mod):
        _attr = getattr(_mod, _attr_name)
        # 仅收集 BaseFactor 子类（排除 helper / constant / 不相关对象）
        if isinstance(_attr, type) and issubclass(_attr, BaseFactor) and _attr is not BaseFactor:
            globals()[_attr_name] = _attr
            _loaded_factor_classes.append(_attr_name)

# 3) 防御性兜底：若 list_available_factors() 报告了但扫描未加载的
#    （例如因子定义在 factors/__init__.py 中而非独立子模块）
_registry_names = set(list_available_factors())
_scanned_names = set(_loaded_factor_classes)
_missing = _registry_names - _scanned_names
if _missing:
    for _name in _missing:
        _cls = getattr(_factors_pkg, _name, None)
        if _cls is not None and isinstance(_cls, type) and issubclass(_cls, BaseFactor):
            globals()[_name] = _cls
            _loaded_factor_classes.append(_name)

# 4) 静态导出：基础 API（不依赖因子数量，跨版本稳定）
#    这些是公开 API，必须显式赋值到 globals() 才能让 `from ... import X` 拿到
_API_EXPORTS = [
    ("BaseFactor", BaseFactor),
    ("register_factor", register_factor),
    ("get_factor", get_factor),
    ("list_available_factors", list_available_factors),
    ("get_factor_registry", get_factor_registry),
    ("SUB_STRATEGY_FACTOR_GROUPS", SUB_STRATEGY_FACTOR_GROUPS),
    ("get_sub_strategy_factors", get_sub_strategy_factors),
    ("FactorEngine", FactorEngine),
    ("FactorPipeline", FactorPipeline),
    ("PipelineResult", PipelineResult),
    ("compute_sub_strategy_scores_from_ohlcv", compute_sub_strategy_scores_from_ohlcv),
]
__all__ = [name for name, _ in _API_EXPORTS]
for _name, _obj in _API_EXPORTS:
    globals()[_name] = _obj
# 追加动态发现的因子类
__all__.extend(sorted(_loaded_factor_classes))
__all__ = list(dict.fromkeys(__all__))  # 去重保序
