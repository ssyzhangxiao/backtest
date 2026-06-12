"""
TS_03: 收盘价位置因子（替代原基差变化率，适配无期限结构数据）。

测量收盘价在近期最高最低区间内的相对位置，
高位=多头主导，低位=空头主导。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import sma


@register_factor
class TS_03(BaseFactor):
    """TS_03: 收盘价位置因子。"""

    name = "TS_03"
    category = "期限结构"
    formula = "(CLOSE - MIN(LOW,10)) / (MAX(HIGH,10) - MIN(LOW,10))"
    dependencies = ["close", "high", "low"]

    window: int = 10

    def compute(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        import pandas as pd
        s_close = pd.Series(close)
        s_high = pd.Series(high)
        s_low = pd.Series(low)
        highest = s_high.rolling(self.window).max().values
        lowest = s_low.rolling(self.window).min().values
        denom = highest - lowest
        safe_denom = np.where(denom < 1e-8, np.nan, denom)
        return (close - lowest) / safe_denom

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)
