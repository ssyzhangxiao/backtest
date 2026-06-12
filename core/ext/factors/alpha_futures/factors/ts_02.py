"""
TS_02: 价格加速度（替代原展期收益率，适配无期限结构数据）。

衡量价格变化率的变化率，加速上升/下降预示趋势延续。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delta, safe_div


@register_factor
class TS_02(BaseFactor):
    """TS_02: 价格加速度。"""

    name = "TS_02"
    category = "期限结构"
    formula = "DELTA(DELTA(CLOSE,1)/DELAY(CLOSE,1), 5)"
    dependencies = ["close"]

    mom_window: int = 1
    accel_window: int = 5

    def compute(
        self,
        close: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        ret = safe_div(delta(close, self.mom_window), close)
        return delta(ret, self.accel_window)

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)
