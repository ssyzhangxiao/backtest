"""
止损优化模块。

提供追踪止损、时间止损和复合止损管理。
"""

from .trailing_stop import TrailingStop
from .time_stop import TimeStop
from .composite_stop import CompositeStopManager

__all__ = [
    "TrailingStop",
    "TimeStop",
    "CompositeStopManager",
]
