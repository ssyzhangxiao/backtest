"""兼容层 — PyBrokerExecutorBuilder 已迁移到 core/execution/pybroker_executor.py。"""

import warnings
from core.execution.pybroker_executor import PyBrokerExecutorBuilder  # noqa: F401

warnings.warn(
    "请使用 from core.execution.pybroker_executor import PyBrokerExecutorBuilder",
    DeprecationWarning,
    stacklevel=2,
)
