"""
M_03: 20日条件增仓累积。

资金流因子，中期资金流向能量潮指标。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delay, delta, sum_rolling


@register_factor
class M_03(BaseFactor):
    """M_03: 20日条件增仓累积。"""

    name = "M_03"
    category = "资金流"
    formula = "SUM(CLOSE>DELAY(CLOSE,1)?DELTA(OI_SAFE,1):(CLOSE<DELAY(CLOSE,1)?-DELTA(OI_SAFE,1):0),20)"
    dependencies = ["close", "oi_safe"]

    # P1整改：硬编码窗口改为类属性
    sum_window: int = 20

    def compute(
        self,
        close: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        delta_oi = delta(oi_safe, 1)
        prev_close = delay(close, 1)
        conditional_oi = np.where(
            close > prev_close,
            delta_oi,
            np.where(close < prev_close, -delta_oi, 0.0),
        )
        return sum_rolling(conditional_oi, self.sum_window)

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)