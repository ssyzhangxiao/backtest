"""
T_05: 6日条件增仓累积（OBV-OI变形）。

趋势因子，捕捉趋势方向确认信号。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delay, delta, sum_rolling


@register_factor
class T_05(BaseFactor):
    """T_05: 6日条件增仓累积（OBV-OI变形）。"""

    name = "T_05"
    category = "趋势"
    formula = "SUM(CLOSE>DELAY(CLOSE,1)?DELTA(OI_SAFE,1):(CLOSE<DELAY(CLOSE,1)?-DELTA(OI_SAFE,1):0),6)"
    dependencies = ["close", "oi_safe"]

    # P1整改：硬编码窗口改为类属性
    sum_window: int = 6

    def compute(
        self,
        close: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        计算 T_05 因子。

        Args:
            close: 收盘价序列
            oi_safe: 安全持仓量序列
            **kwargs: 其他参数（未使用）

        Returns:
            T_05 因子值序列
        """
        delta_oi = delta(oi_safe, 1)
        prev_close = delay(close, 1)
        conditional_oi = np.where(
            close > prev_close,
            delta_oi,
            np.where(close < prev_close, -delta_oi, 0.0),
        )
        return sum_rolling(conditional_oi, self.sum_window)

    def post_process(self, values: np.ndarray) -> np.ndarray:
        """
        后处理：1%和99%缩尾去除极端值。

        Args:
            values: 原始因子值序列

        Returns:
            后处理后的因子值序列
        """
        from ...operators import winsorize

        return winsorize(values, lower_pct=0.01, upper_pct=0.99)
