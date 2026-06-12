"""
M_04: 价格方向驱动的资金流累积（优化版，替代原Carry依赖）。

资金流因子，每日价格涨跌方向驱动的持仓变化累积。
无需Carry数据结构。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delta, sum_rolling, sign


@register_factor
class M_04(BaseFactor):
    """M_04: 价格方向驱动的资金流累积。"""

    name = "M_04"
    category = "资金流"
    formula = "SUM(SIGN(DELTA(CLOSE,1))*DELTA(OI_SAFE,1), 10)"
    dependencies = ["close", "oi_safe"]

    cum_window: int = 10

    def compute(
        self,
        close: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        direction = sign(delta(close, 1))
        oi_flow = direction * delta(oi_safe, 1)
        return sum_rolling(oi_flow, self.cum_window)

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)
