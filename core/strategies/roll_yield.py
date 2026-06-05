"""
兼容性模块：旧命名支持。

请使用:
    from core.strategies.strategy_roll_yield import RollYieldStrategy

或直接从包导入:
    from core.strategies import RollYieldStrategy
"""
from .strategy_roll_yield import RollYieldStrategy

__all__ = ["RollYieldStrategy"]
