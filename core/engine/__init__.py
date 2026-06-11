"""
引擎模块。

提供因子打分调仓引擎和向后兼容重导出。

⚠️ 迁移历史（2026-06-12）：
  - backtest_runner / pybroker_executor / _bootstrap / _result_types / _walkforward
    已迁移到 core/execution/，此处保留向后兼容重导出
  - factor_decay / rolling_ic 已物理删除，功能由 core.ext.factors.evaluator 提供
  - cross_validator 已迁移到 core/validation/
  - FactorDecayMonitor / RollingICWeightEngine 兼容别名已删除（2026-06-12）
"""

from core.engine.switch_engine import (
    FactorScoringEngine,
    ScoringConfig,
    RebalanceDecision,
    RebalanceReason,
)
from core.validation.cross_validator import CrossValidator

# ── 向后兼容：从 core.execution 重导出 ──
from core.execution.backtest_runner import PyBrokerBacktestRunner
from core.execution.pybroker_executor import PyBrokerExecutorBuilder
from core.execution._result_types import PyBrokerResult, WalkforwardResult

__all__ = [
    # 核心引擎
    "FactorScoringEngine",
    "ScoringConfig",
    "RebalanceDecision",
    "RebalanceReason",
    "CrossValidator",
    # 从 core.execution 重导出
    "PyBrokerBacktestRunner",
    "PyBrokerExecutorBuilder",
    "PyBrokerResult",
    "WalkforwardResult",
]
