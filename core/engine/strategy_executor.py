"""兼容层 — strategy_executor 已废弃，请使用 core.execution.pybroker_executor。"""

import warnings

warnings.warn(
    "core.engine.strategy_executor 已废弃，请使用 "
    "from core.execution.pybroker_executor import PyBrokerExecutorBuilder",
    DeprecationWarning,
    stacklevel=2,
)

# 从 deprecated/ 保留 RiskManagerAdapter 兼容
from core.deprecated.strategy_executor import RiskManagerAdapter  # noqa: F401

__all__ = ["RiskManagerAdapter"]
