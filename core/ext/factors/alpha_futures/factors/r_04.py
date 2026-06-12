"""
R_04: 价格动量均值回复（优化版，替代原Carry依赖）。

回归因子，动量过后的价格均值回复。
无需Carry数据结构。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import tsrank, delta


@register_factor
class R_04(BaseFactor):
    """R_04: 价格动量均值回复（替代原Carry Z-score）。"""

    name = "R_04"
    category = "回归"
    formula = "-1 * TSRANK(DELTA(CLOSE,20), 40)"
    dependencies = ["close"]

    momentum_window: int = 20
    rank_window: int = 40

    def compute(
        self,
        close: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        mom = delta(close, self.momentum_window)
        return -1.0 * tsrank(mom, self.rank_window)

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)
