"""
回测系统配置包。

规则2：config.yaml 是单一数据源，BacktestConfig 必须与 yaml 完全同步。
规则16：模块目录结构 — 职责单一，接口清晰。
规则23：分层配置 — defaults < YAML < env vars < runtime overrides。

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
from .stop_config import StopOptimizationConfig
from .validation_config import ValidationModuleConfig
from .backtest_config import BacktestConfig
from .strategy_profiles import (
    StrategyProfile,
    StrategyLibrary,
    STRATEGY_NAMES,
    SUB_STRATEGY_NAMES,
)
from .layered_config import (
    LayeredConfigLoader,
    ENV_PREFIX,
    ENV_SECTION_ALIAS,
    load_env_overrides,
    merge_overrides,
)

__all__ = [
    "DATA_DIR",
    "PYBROKER_EXTRA_COLUMNS",
    "INITIAL_CASH",
    "DEFAULT_FACTOR_WEIGHTS",
    "get_default_stress_events",
    "FactorModuleConfig",
    "StopOptimizationConfig",
    "ValidationModuleConfig",
    "BacktestConfig",
    "StrategyProfile",
    "StrategyLibrary",
    "STRATEGY_NAMES",
    "SUB_STRATEGY_NAMES",
    "LayeredConfigLoader",
    "ENV_PREFIX",
    "ENV_SECTION_ALIAS",
    "load_env_overrides",
    "merge_overrides",
]
