"""
引擎模块。

提供核心回测引擎和因子打分调仓引擎。

⚠️ P0-5整改（2026-06-07）：
  - 自研回测引擎 BacktestRunner 已完全移除
  - 交叉验证功能提取到 CrossValidator（core/engine/cross_validator.py）
  - PyBroker 主引擎位于 core/engine/backtest_runner.py
  - 蓝图执行器位于 core/engine/pybroker_executor.py
"""

from core.engine.switch_engine import (
    FactorScoringEngine,
    ScoringConfig,
    RebalanceDecision,
    RebalanceReason,
)
from core.engine.cross_validator import CrossValidator
from core.engine.rolling_ic import RollingICWeightEngine, RollingICConfig
from core.engine.factor_decay import FactorDecayMonitor, FactorDecayConfig, DecayStatus, DecayAlert

__all__ = [
    "FactorScoringEngine",
    "ScoringConfig",
    "RebalanceDecision",
    "RebalanceReason",
    "CrossValidator",
    "RollingICWeightEngine",
    "RollingICConfig",
    "FactorDecayMonitor",
    "FactorDecayConfig",
    "DecayStatus",
    "DecayAlert",
]
