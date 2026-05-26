"""向后兼容配置模块（v2: 从 core.config 重导出）。

新代码请直接从 core.config 导入：
  from core.config import DATA_DIR, PYBROKER_EXTRA_COLUMNS, get_default_stress_events, BacktestConfig
"""

from core.config import (
    DATA_DIR,
    PYBROKER_EXTRA_COLUMNS,
    _PYBROKER_COLUMNS,
    INITIAL_CASH,
    get_default_stress_events,
    BacktestConfig,
)

# 向后兼容旧名称
PYBROKER_COLUMNS = PYBROKER_EXTRA_COLUMNS