"""
因子注册表。

使用装饰器自动注册因子类，提供查询接口。
"""
from typing import Dict, List, Type, Any

from .base_factor import BaseFactor

# 全局因子注册表
_FACTOR_REGISTRY: Dict[str, Type[BaseFactor]] = {}


def register_factor(cls: Type[BaseFactor]) -> Type[BaseFactor]:
    """
    装饰器：注册因子类。

    Args:
        cls: 因子类（继承自 BaseFactor）

    Returns:
        原因子类（支持链式调用）
    """
    if not hasattr(cls, "name") or not cls.name:
        raise ValueError(f"因子类 {cls.__name__} 必须定义 name 属性")
    _FACTOR_REGISTRY[cls.name] = cls
    return cls


def get_factor(name: str, config: Any) -> BaseFactor:
    """
    根据名称获取因子实例。

    Args:
        name: 因子编号，如 "T_01"
        config: 全局配置对象

    Returns:
        因子实例

    Raises:
        ValueError: 未知因子
    """
    if name not in _FACTOR_REGISTRY:
        raise ValueError(f"未知因子: {name}，可用因子: {list_available_factors()}")
    return _FACTOR_REGISTRY[name](config)


def list_available_factors() -> List[str]:
    """
    列出所有已注册因子。

    Returns:
        因子编号列表
    """
    return list(_FACTOR_REGISTRY.keys())


def get_factor_registry() -> Dict[str, Type[BaseFactor]]:
    """
    获取因子注册表（只读）。

    Returns:
        因子注册表副本
    """
    return _FACTOR_REGISTRY.copy()


# 子策略 → 因子名列表的固定映射（P0 整改：替换 basic_factors.compute_factor_scores_from_ohlcv）
SUB_STRATEGY_FACTOR_GROUPS: Dict[str, List[str]] = {
    "trend": ["T_01", "T_02", "T_03", "T_04", "T_05"],
    "term_structure": ["TS_01", "TS_02", "TS_03"],
    "mean_reversion": ["M_01", "M_02", "M_03", "M_04", "M_05"],
    "vol_breakout": ["V_01", "V_02", "V_03", "V_04", "H_01", "H_02", "H_03", "H_04", "H_05"],
    "composite_resonance": ["R_01", "R_02", "R_03", "R_04", "R_05", "CF_01", "CF_02", "CF_03"],
}


def get_sub_strategy_factors(strategy_name: str) -> List[str]:
    """
    获取指定子策略对应的因子编号列表。

    Args:
        strategy_name: 子策略名（trend / term_structure / mean_reversion / vol_breakout / composite_resonance）

    Returns:
        因子编号列表，未知子策略返回空列表
    """
    return list(SUB_STRATEGY_FACTOR_GROUPS.get(strategy_name, []))
