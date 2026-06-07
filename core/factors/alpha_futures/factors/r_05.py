"""
R_05: 负相对持仓量。

回归因子，捕捉持仓异常萎缩→蓄势反转节点信号。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import mean, safe_div


@register_factor
class R_05(BaseFactor):
    """R_05: 负相对持仓量。"""

    name = "R_05"
    category = "回归"
    formula = "-1 * OI_SAFE / MEAN(OI_SAFE, 20)"
    dependencies = ["oi_safe"]

    # P1整改：硬编码窗口改为类属性
    ma_window: int = 20

    def compute(
        self,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        计算 R_05 因子。

        Args:
            oi_safe: 安全持仓量序列
            **kwargs: 其他参数（未使用）

        Returns:
            R_05 因子值序列
        """
        return -1 * safe_div(oi_safe, mean(oi_safe, self.ma_window))

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
