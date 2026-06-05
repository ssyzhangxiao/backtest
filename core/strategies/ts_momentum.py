"""
兼容性模块：旧命名支持。

请使用:
    from core.strategies.strategy_ts_momentum import TSMomentumStrategy

或直接从包导入:
    from core.strategies import TSMomentumStrategy
"""
from .strategy_ts_momentum import TSMomentumStrategy

__all__ = ["TSMomentumStrategy"]
