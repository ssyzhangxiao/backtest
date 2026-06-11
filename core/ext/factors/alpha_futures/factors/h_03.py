"""
H_03: 相对持仓时序排名与反转时序排名乘积。

高阶复合因子，持仓异常放大+价格短期超跌→反弹拐点。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delta, mean, safe_div, tsrank


@register_factor
class H_03(BaseFactor):
    """H_03: 相对持仓时序排名与反转时序排名乘积。"""

    name = "H_03"
    category = "高阶复合"
    formula = "TSRANK(OI_SAFE/MEAN(OI_SAFE,20), 20) * TSRANK(-1*DELTA(CLOSE,7), 8)"
    dependencies = ["close", "oi_safe"]

    # P1整改：硬编码窗口改为类属性
    oi_ma_window: int = 20
    oi_tsrank_window: int = 20
    close_delta_window: int = 7
    close_tsrank_window: int = 8

    def compute(
        self,
        close: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        # P0整改：内部计算 oi_mean_20
        oi_mean_20 = mean(oi_safe, self.oi_ma_window)
        rel_oi = safe_div(oi_safe, oi_mean_20)
        return tsrank(rel_oi, self.oi_tsrank_window) * tsrank(
            -1 * delta(close, self.close_delta_window), self.close_tsrank_window
        )

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)