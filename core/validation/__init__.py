"""
回测验证模块。

提供蒙特卡洛模拟、参数敏感性分析和样本外验证。

P1 整改（2026-06-07）：
  - SensitivityResult 扩展为三指标（Sharpe/MaxDrawdown/AnnualReturn）
  - analyze 加返回值校验（None 兜底，非 dict 抛错）

P2 整改（2026-06-07）：
  - MonteCarloSimulator 支持 trading_days_per_year 参数化
  - SensitivityAnalyzer 支持 n_jobs 并行扰动
"""

from .monte_carlo import (
    MonteCarloSimulator,
    MonteCarloResult,
    DEFAULT_N_SIMULATIONS,
    DEFAULT_TRADING_DAYS_PER_YEAR,
    QUANTILES,
)
from .sensitivity import (
    SensitivityAnalyzer,
    SensitivityResult,
    FullSensitivityResult,
    DEFAULT_PERTURBATION,
    HIGH_SENSITIVITY_THRESHOLD,
)

__all__ = [
    # Monte Carlo
    "MonteCarloSimulator",
    "MonteCarloResult",
    "DEFAULT_N_SIMULATIONS",
    "DEFAULT_TRADING_DAYS_PER_YEAR",
    "QUANTILES",
    # Sensitivity
    "SensitivityAnalyzer",
    "SensitivityResult",
    "FullSensitivityResult",
    "DEFAULT_PERTURBATION",
    "HIGH_SENSITIVITY_THRESHOLD",
]
