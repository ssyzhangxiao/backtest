"""
策略注册表模块。

包含所有可用策略的映射字典以及创建策略实例的工具函数。
"""
from typing import Dict

from .base import BaseStrategy
from .dual_ma import DualMAStrategy
from .rsi import RSIStrategy
from .spread import SpreadStrategy
from .term_structure import TermStructureStrategy
from .vol_breakout import VolatilityBreakoutStrategy

STRATEGY_REGISTRY: Dict[str, type] = {
    "dual_ma": DualMAStrategy,
    "rsi": RSIStrategy,
    "spread": SpreadStrategy,
    "term_structure": TermStructureStrategy,
    "vol_breakout": VolatilityBreakoutStrategy,
}


def get_strategy_class(name: str) -> type:
    """
    根据策略名称获取策略类。

    Args:
        name: 策略名称，如 'dual_ma', 'rsi', 'spread' 等

    Returns:
        对应的策略类

    Raises:
        ValueError: 当策略名称不存在时
    """
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"未知策略 '{name}'，可用策略: {available}")
    return STRATEGY_REGISTRY[name]


def create_strategy(name: str, **kwargs) -> BaseStrategy:
    """
    根据策略名称创建策略实例。

    Args:
        name: 策略名称
        **kwargs: 传递到策略构造函数的参数

    Returns:
        策略实例
    """
    strategy_class = get_strategy_class(name)
    return strategy_class(**kwargs)