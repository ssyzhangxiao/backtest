"""
兼容性模块：旧命名支持。

请使用:
    from core.strategies.strategy_alpha032 import Alpha032Strategy

或直接从包导入:
    from core.strategies import Alpha032Strategy
"""
from .strategy_alpha032 import Alpha032Strategy

__all__ = ["Alpha032Strategy"]
