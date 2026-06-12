"""
H_01: 条件性动量共振（优化版，替代原Carry依赖）。

高阶复合因子，增仓环境下捕价格动量共振；缩仓退守负持仓因子。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import abs_, delta, mean, tsrank, zscore


@register_factor
class H_01(BaseFactor):
    """H_01: 条件性动量共振。"""

    name = "H_01"
    category = "高阶复合"
    formula = "(OI_SAFE>MEAN(OI_SAFE,20)) ? ZSCORE(CLOSE,20)*TSRANK(ABS(DELTA(CLOSE,7)),60) : (-1*OI_SAFE)"
    dependencies = ["close", "oi_safe"]

    oi_ma_window: int = 20
    momentum_window: int = 7
    momentum_tsrank_window: int = 60
    price_z_window: int = 20

    def compute(
        self,
        close: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        oi_mean_20 = mean(oi_safe, self.oi_ma_window)
        is_accumulating = oi_safe > oi_mean_20
        price_momentum = tsrank(abs_(delta(close, self.momentum_window)), self.momentum_tsrank_window)
        price_z = zscore(close, self.price_z_window)
        return np.where(is_accumulating, price_z * price_momentum, -1 * oi_safe)

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)
