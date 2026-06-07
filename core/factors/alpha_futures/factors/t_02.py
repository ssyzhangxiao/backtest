"""
T_02: 12日动量与总持仓乘积。

趋势因子，捕捉中期价格动量与持仓量共振信号。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delay, delta, safe_div


@register_factor
class T_02(BaseFactor):
    """T_02: 12日动量与总持仓乘积。"""

    name = "T_02"
    category = "趋势"
    formula = "(CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12) * OI_SAFE"
    dependencies = ["close", "oi_safe"]

    # P1整改：硬编码窗口改为类属性
    momentum_window: int = 12

    def compute(
        self,
        close: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        计算 T_02 因子。

        Args:
            close: 收盘价序列
            oi_safe: 安全持仓量序列
            **kwargs: 其他参数（未使用）

        Returns:
            T_02 因子值序列
        """
        momentum = safe_div(delta(close, self.momentum_window), delay(close, self.momentum_window))
        return momentum * oi_safe

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
