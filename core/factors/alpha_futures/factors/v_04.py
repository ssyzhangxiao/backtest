"""
V_04: 持仓量均线差率（OI-MACD柱）。

波动率因子，持仓量长短均线发散→波动率将上升。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import mean, safe_div


@register_factor
class V_04(BaseFactor):
    """V_04: 持仓量均线差率（OI-MACD柱）。"""

    name = "V_04"
    category = "波动率"
    formula = "(MEAN(OI_SAFE,9)-MEAN(OI_SAFE,26))/MEAN(OI_SAFE,12)*100"
    dependencies = ["oi_safe"]

    # P1整改：硬编码窗口改为类属性
    fast_ma: int = 9
    slow_ma: int = 26
    ref_ma: int = 12

    def compute(
        self,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        oi_fast = mean(oi_safe, self.fast_ma)
        oi_slow = mean(oi_safe, self.slow_ma)
        oi_ref = mean(oi_safe, self.ref_ma)
        return safe_div(oi_fast - oi_slow, oi_ref) * 100

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)