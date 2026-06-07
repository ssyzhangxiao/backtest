"""
Runner 公共工具包导出层（规则17）。

所有函数委托给子模块（utils / portfolio_utils / config_utils / errors），
本模块仅做 re-export，避免调用方感知子模块路径变化。
"""

from runner.common.utils import (
    format_metrics,
    handle_backtest_errors,
    is_valid_number,
    safe_div,
    safe_float,
    sanitize_filename,
    save_csv,
    save_equity_curve,
)
from runner.common.portfolio_utils import (
    calculate_risk_parity_fusion,
    calculate_risk_parity_weights,
    calculate_rolling_volatility,
)
from runner.common.config_utils import (
    get_backtest_config,
    get_factors_list,
    get_missing_data_method,
    get_montecarlo_config,
    get_walkforward_config,
)
from runner.common.errors import (
    BacktestError,
    ConfigError,
    DataError,
    OptimizationError,
    PipelineError,
    ValidationError,
)

__all__ = [
    # utils
    "format_metrics",
    "handle_backtest_errors",
    "is_valid_number",
    "safe_div",
    "safe_float",
    "sanitize_filename",
    "save_csv",
    "save_equity_curve",
    # portfolio_utils
    "calculate_risk_parity_fusion",
    "calculate_risk_parity_weights",
    "calculate_rolling_volatility",
    # config_utils
    "get_backtest_config",
    "get_factors_list",
    "get_missing_data_method",
    "get_montecarlo_config",
    "get_walkforward_config",
    # errors
    "BacktestError",
    "ConfigError",
    "DataError",
    "OptimizationError",
    "PipelineError",
    "ValidationError",
]

