"""
V_03: 日内振幅与增仓幅度双滚动标准化乘积。

波动率因子，使用滚动标准化（窗口20），避免全序列前瞻性偏差。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delta, zscore


@register_factor
class V_03(BaseFactor):
    """V_03: 日内振幅与增仓幅度双滚动标准化乘积。"""

    name = "V_03"
    category = "波动率"
    formula = "ZSCORE(HIGH-LOW, 20) * ZSCORE(DELTA(OI_SAFE,1), 20)"
    dependencies = ["high", "low", "oi_safe"]

    # P1整改：硬编码窗口改为类属性
    zscore_window_param: int = 20

    def compute(
        self,
        high: np.ndarray,
        low: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        range_val = high - low
        w = self.zscore_window_param
        return zscore(range_val, window=w) * zscore(delta(oi_safe, 1), window=w)

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)