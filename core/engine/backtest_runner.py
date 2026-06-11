"""兼容层 — PyBrokerBacktestRunner 已迁移到 core/execution/backtest_runner.py。"""

import warnings
from core.execution.backtest_runner import PyBrokerBacktestRunner  # noqa: F401

warnings.warn(
    "请使用 from core.execution.backtest_runner import PyBrokerBacktestRunner",
    DeprecationWarning,
    stacklevel=2,
)
