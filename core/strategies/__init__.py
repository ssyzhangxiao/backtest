"""
策略包。

提供统一的策略接口，包含基类、全部策略实现及注册表。

使用示例：
    from core.strategies import TSMomentumStrategy, create_strategy
    strategy = create_strategy('ts_momentum', window=20, position_size=0.2)
"""
from .base import BaseStrategy
from .ts_momentum import TSMomentumStrategy
from .roll_yield import RollYieldStrategy
from .alpha019 import Alpha019Strategy
from .alpha032 import Alpha032Strategy
from .registry import STRATEGY_REGISTRY, get_strategy_class, create_strategy

__all__ = [
    "BaseStrategy",
    "TSMomentumStrategy",
    "RollYieldStrategy",
    "Alpha019Strategy",
    "Alpha032Strategy",
    "STRATEGY_REGISTRY",
    "get_strategy_class",
    "create_strategy",
]