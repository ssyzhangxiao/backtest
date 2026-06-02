"""
多时间框架模块。

提供周频/月频趋势判断、时间框架过滤和信号同步。
"""

from .trend_filter import MultiTFFilter, TrendDirection
from .signal_filter import SignalFilter, FilterResult

__all__ = [
    "MultiTFFilter",
    "TrendDirection",
    "SignalFilter",
    "FilterResult",
]
