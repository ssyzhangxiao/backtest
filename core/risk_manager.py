"""
风控模块 — 向后兼容层。

实际实现已迁移到:
  - core.risk_controller: 纯风控逻辑（RiskController）
  - core.engine.strategy_executor.RiskManagerAdapter: PyBroker 适配层

此模块仅保留导入重定向。
"""

from core.risk_controller import RiskController, RiskConfig
from core.engine.strategy_executor import RiskManagerAdapter

# 向后兼容别名
RiskManager = RiskManagerAdapter

__all__ = ["RiskController", "RiskConfig", "RiskManager", "RiskManagerAdapter"]
