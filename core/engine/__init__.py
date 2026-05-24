"""
引擎模块。

提供核心回测引擎和策略切换引擎。
"""

from core.engine.switch_engine import (
    StrategySwitchEngine,
    SwitchConfig,
    SwitchDecision,
    SwitchReason,
)
from core.engine.runner import (
    BacktestRunner,
    BacktestConfig,
    StrategyResult,
    PortfolioResult,
)

__all__ = [
    "StrategySwitchEngine",
    "SwitchConfig",
    "SwitchDecision",
    "SwitchReason",
    "BacktestRunner",
    "BacktestConfig",
    "StrategyResult",
    "PortfolioResult",
]
