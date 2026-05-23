"""
策略包。

提供统一的策略接口，包含基类、全部策略实现及注册表。

使用示例：
    from core.strategies import DualMAStrategy, create_strategy
    strategy = create_strategy('dual_ma', short_ma=5, long_ma=20)
"""
from .base import BaseStrategy
from .dual_ma import DualMAStrategy
from .rsi import RSIStrategy
from .spread import SpreadStrategy
from .term_structure import TermStructureStrategy
from .vol_breakout import VolatilityBreakoutStrategy
from .registry import STRATEGY_REGISTRY, get_strategy_class, create_strategy

__all__ = [
    "BaseStrategy",
    "DualMAStrategy",
    "RSIStrategy",
    "SpreadStrategy",
    "TermStructureStrategy",
    "VolatilityBreakoutStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy_class",
    "create_strategy",
]