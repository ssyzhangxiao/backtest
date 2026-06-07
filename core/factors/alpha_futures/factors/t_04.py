"""
T_04: 期限结构与增仓共振（顶级Alpha因子）。

趋势因子，期限结构陡峭化与增仓共振信号。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delta


@register_factor
class T_04(BaseFactor):
    """T_04: 期限结构与增仓共振（顶级Alpha因子）。"""

    name = "T_04"
    category = "趋势"
    formula = "CARRY * DELTA(OI_SAFE,1)"
    dependencies = ["carry", "oi_safe"]

    # P1整改：硬编码窗口改为类属性
    oi_delta_window: int = 1

    def compute(
        self,
        carry: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        计算 T_04 因子。

        Args:
            carry: Carry因子序列
            oi_safe: 安全持仓量序列
            **kwargs: 其他参数（未使用）

        Returns:
            T_04 因子值序列
        """
        return carry * delta(oi_safe, self.oi_delta_window)

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
