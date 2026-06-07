"""
T_03: 日度收益率与日增仓乘积。

趋势因子，捕捉极短期资金入场方向确认信号。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delay, delta, safe_div


@register_factor
class T_03(BaseFactor):
    """T_03: 日度收益率与日增仓乘积。"""

    name = "T_03"
    category = "趋势"
    formula = "(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1) * DELTA(OI_SAFE,1)"
    dependencies = ["close", "oi_safe"]

    # P1整改：硬编码窗口改为类属性（delta=1 是 1-day 默认值）
    close_delta_window: int = 1
    oi_delta_window: int = 1

    def compute(
        self,
        close: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        计算 T_03 因子。

        Args:
            close: 收盘价序列
            oi_safe: 安全持仓量序列
            **kwargs: 其他参数（未使用）

        Returns:
            T_03 因子值序列
        """
        daily_ret = safe_div(
            delta(close, self.close_delta_window),
            delay(close, self.close_delta_window),
        )
        return daily_ret * delta(oi_safe, self.oi_delta_window)

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
