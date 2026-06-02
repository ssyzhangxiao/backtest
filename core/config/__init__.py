"""
回测系统配置包。

规则2：config.yaml 是单一数据源，BacktestConfig 必须与 yaml 完全同步。
规则16：模块目录结构 — 职责单一，接口清晰。

拆分后的配置类按职责独立存放，通过本文件统一导出公共接口。
外部代码 `from core.config import BacktestConfig` 保持不变。
"""

from .constants import (
    DATA_DIR,
    PYBROKER_EXTRA_COLUMNS,
    INITIAL_CASH,
    DEFAULT_FACTOR_WEIGHTS,
    get_default_stress_events,
)
from .factors_config import FactorModuleConfig
from .adaptive_config import AdaptiveModuleConfig
from .multi_tf_config import MultiTFModuleConfig
from .position_config import PositionModuleConfig
from .stop_config import StopOptimizationConfig
from .instrument_config import InstrumentModuleConfig
from .validation_config import ValidationModuleConfig
from .backtest_config import BacktestConfig

__all__ = [
    "DATA_DIR",
    "PYBROKER_EXTRA_COLUMNS",
    "INITIAL_CASH",
    "DEFAULT_FACTOR_WEIGHTS",
    "get_default_stress_events",
    "FactorModuleConfig",
    "AdaptiveModuleConfig",
    "MultiTFModuleConfig",
    "PositionModuleConfig",
    "StopOptimizationConfig",
    "InstrumentModuleConfig",
    "ValidationModuleConfig",
    "BacktestConfig",
]
