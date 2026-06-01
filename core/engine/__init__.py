"""
引擎模块。

提供核心回测引擎和因子打分调仓引擎。
"""

from core.engine.switch_engine import (
    FactorScoringEngine,
    StrategySwitchEngine,
    ScoringConfig,
    RebalanceDecision,
    RebalanceReason,
)
from core.engine.runner import (
    BacktestRunner,
    BacktestConfig,
    StrategyResult,
    PortfolioResult,
)
from core.engine.rolling_ic import RollingICWeightEngine, RollingICConfig
from core.engine.factor_decay import FactorDecayMonitor, FactorDecayConfig, DecayStatus, DecayAlert

__all__ = [
    "FactorScoringEngine",
    "StrategySwitchEngine",
    "ScoringConfig",
    "RebalanceDecision",
    "RebalanceReason",
    "BacktestRunner",
    "BacktestConfig",
    "StrategyResult",
    "PortfolioResult",
    "RollingICWeightEngine",
    "RollingICConfig",
    "FactorDecayMonitor",
    "FactorDecayConfig",
    "DecayStatus",
    "DecayAlert",
]
