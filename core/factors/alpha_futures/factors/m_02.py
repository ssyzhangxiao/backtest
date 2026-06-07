"""
M_02: 20日日内多空力量与增仓累积。

资金流因子，M_01的中期波段版本，过滤单日噪声。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delta, safe_div, sum_rolling


@register_factor
class M_02(BaseFactor):
    """M_02: 20日日内多空力量与增仓累积。"""

    name = "M_02"
    category = "资金流"
    formula = "SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*DELTA(OI_SAFE,1), 20)"
    dependencies = ["close", "high", "low", "oi_safe"]

    # P1整改：硬编码窗口改为类属性
    sum_window: int = 20

    def compute(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        delta_oi = delta(oi_safe, 1)
        safe_range = high - low
        power = safe_div((close - low) - (high - close), safe_range)
        return sum_rolling(power * delta_oi, self.sum_window)

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)