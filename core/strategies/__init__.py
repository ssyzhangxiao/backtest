"""
策略包。

提供统一的策略接口，包含基类、全部策略实现及注册表。

使用示例：
    from core.strategies import TSMomentumStrategy, create_strategy
    strategy = create_strategy('ts_momentum', window=20, position_size=0.2)
"""

from .base import BaseStrategy
from .strategy_ts_momentum import TSMomentumStrategy
from .strategy_roll_yield import RollYieldStrategy
from .strategy_alpha019 import Alpha019Strategy
from .strategy_alpha032 import Alpha032Strategy
from .registry import STRATEGY_REGISTRY, get_strategy_class, create_strategy

# 向后兼容：支持旧命名导入
try:
    from .ts_momentum import TSMomentumStrategy as _TSMomentumStrategy_old
except ImportError:
    pass
try:
    from .roll_yield import RollYieldStrategy as _RollYieldStrategy_old
except ImportError:
    pass
try:
    from .alpha019 import Alpha019Strategy as _Alpha019Strategy_old
except ImportError:
    pass
try:
    from .alpha032 import Alpha032Strategy as _Alpha032Strategy_old
except ImportError:
    pass

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
