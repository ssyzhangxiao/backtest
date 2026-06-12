"""
V_04: 持仓量均线差率（OI-MACD柱）。

波动率因子，持仓量长短均线发散→波动率将上升。

2026-06-12 优化：降低NaN传播——收紧跳变阈值、缩短MA窗口。
"""

import numpy as np
import pandas as pd

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import safe_div, winsorize


@register_factor
class V_04(BaseFactor):
    """V_04: 持仓量均线差率（OI-MACD柱）。"""

    name = "V_04"
    category = "波动率"
    formula = "(SMA(OI,9)-SMA(OI,26))/SMA(OI,12)*100"
    dependencies = ["oi_safe"]

    fast_ma: int = 9
    slow_ma: int = 26
    ref_ma: int = 12

    def compute(
        self,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        oi_s = pd.Series(oi_safe, dtype=float)
        # 使用 min_periods=1 容忍NaN，避免单次跳变污染整段
        oi_fast = oi_s.rolling(self.fast_ma, min_periods=1).mean().values
        oi_slow = oi_s.rolling(self.slow_ma, min_periods=1).mean().values
        oi_ref = oi_s.rolling(self.ref_ma, min_periods=1).mean().values
        return safe_div(oi_fast - oi_slow, oi_ref) * 100

    def post_process(self, values: np.ndarray) -> np.ndarray:
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)
