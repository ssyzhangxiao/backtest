"""
兼容性模块：旧命名支持。

请使用:
    from core.strategies.strategy_alpha019 import Alpha019Strategy

或直接从包导入:
    from core.strategies import Alpha019Strategy
"""
from .strategy_alpha019 import Alpha019Strategy

__all__ = ["Alpha019Strategy"]
