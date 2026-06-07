"""
M_04: 期限结构驱动的资金流。

资金流因子，近月升水时增仓=多头力量；远月升水时增仓=空头力量。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delta, sum_rolling


@register_factor
class M_04(BaseFactor):
    """M_04: 期限结构驱动的资金流。"""

    name = "M_04"
    category = "资金流"
    formula = "SUM(CARRY>0?DELTA(OI_SAFE,1):-DELTA(OI_SAFE,1), 10)"
    dependencies = ["carry", "oi_safe"]

    # P1整改：硬编码窗口改为类属性
    sum_window: int = 10

    def compute(
        self,
        carry: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        delta_oi = delta(oi_safe, 1)
        directional_oi = np.where(carry > 0, delta_oi, -delta_oi)
        return sum_rolling(directional_oi, self.sum_window)

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)