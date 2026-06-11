"""AlphaFutures24 因子IC/IR验证 — 兼容shim。

已拆分为：
  - _factor_cross_spread.py: 跨品种价差因子辅助函数
  - _factor_panel.py: 因子面板构建和IC计算
  - _factor_screening.py: 因子筛选（factor_alpha24_screening）
  - _factor_combo.py: 组合IC验证（factor_combo_ic_validation）

本文件仅做重导出，保持旧导入路径兼容。
"""

import warnings

warnings.warn(
    "runner.validation.factor_alpha24 已拆分为子模块，"
    "请使用 from runner.validation._factor_screening import factor_alpha24_screening "
    "或 from runner.validation._factor_combo import factor_combo_ic_validation",
    DeprecationWarning,
    stacklevel=2,
)

from runner.validation._factor_cross_spread import compute_pair_signal, build_cross_spread_panel
from runner.validation._factor_panel import build_factor_panel, compute_cross_sectional_ic
from runner.validation._factor_screening import factor_alpha24_screening
from runner.validation._factor_combo import factor_combo_ic_validation

__all__ = [
    "compute_pair_signal",
    "build_cross_spread_panel",
    "build_factor_panel",
    "compute_cross_sectional_ic",
    "factor_alpha24_screening",
    "factor_combo_ic_validation",
]
