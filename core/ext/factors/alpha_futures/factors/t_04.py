"""
T_04: 价格Z-Score与增仓共振（优化版，替代原Carry依赖）。

趋势因子，价格标准化位置与持仓变化共振信号。
无需Carry数据结构，适配无期限结构数据的场景。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delta, zscore, sma, safe_div


@register_factor
class T_04(BaseFactor):
    """T_04: 价格Z-Score与增仓共振。"""

    name = "T_04"
    category = "趋势"
    formula = "ZSCORE(CLOSE,20) * DELTA(OI_SAFE,1)"
    dependencies = ["close", "oi_safe"]

    price_window: int = 20

    def compute(
        self,
        close: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        price_z = zscore(close, self.price_window)
        return price_z * delta(oi_safe, 1)

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)
