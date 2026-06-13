"""
引擎模块。

提供因子打分调仓引擎等核心功能。

⚠️ 迁移历史（2026-06-12 → 2026-06-13）：
  - backtest_runner / pybroker_executor / _bootstrap / _result_types / _walkforward
    已迁移到 core/execution/，重定向文件已删除（2026-06-13）
    请改用: from core.execution import ...
  - FactorDecayMonitor / RollingICWeightEngine 已删除（功能由 core.ext.factors.evaluator 提供）
  - cross_validator 已迁移到 core/validation/
"""

from core.engine.switch_engine import (
    FactorScoringEngine,
    ScoringConfig,
    RebalanceDecision,
    RebalanceReason,
)
from core.validation.cross_validator import CrossValidator

__all__ = [
    "FactorScoringEngine",
    "ScoringConfig",
    "RebalanceDecision",
    "RebalanceReason",
    "CrossValidator",
]
