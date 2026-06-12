"""
H_05: 三重共振因子（优化版，替代原Carry依赖）。

高阶复合因子，价格位置、持仓量趋势、价格突破方向三者按权重相加。
使用价格Z-Score替代原Carry期限结构。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delta, sign, zscore, winsorize


@register_factor
class H_05(BaseFactor):
    """H_05: 三重共振因子（价格Z-Score替代Carry）。"""

    name = "H_05"
    category = "高阶复合"
    formula = "0.40*ZSCORE(CLOSE,20) + 0.30*Z(ΔOI) + 0.30*SIGN(ΔCLOSE)"
    dependencies = ["close", "oi_safe"]

    weight_price: float = 0.40
    weight_delta_oi: float = 0.30
    weight_price_dir: float = 0.30

    oi_delta_window: int = 5
    close_delta_window: int = 5
    price_z_window: int = 20

    def compute(
        self,
        close: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        zscore_window = kwargs.get("zscore_window", 0)
        if zscore_window is None or zscore_window <= 0:
            window_arg = None
        else:
            window_arg = zscore_window

        price_z = zscore(close, self.price_z_window)
        z_price = zscore(price_z, window=window_arg) if window_arg else price_z
        z_delta_oi = zscore(delta(oi_safe, self.oi_delta_window), window=window_arg)
        price_dir = sign(delta(close, self.close_delta_window))
        return (
            self.weight_price * z_price
            + self.weight_delta_oi * z_delta_oi
            + self.weight_price_dir * price_dir
        )

    def post_process(self, values: np.ndarray) -> np.ndarray:
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)
