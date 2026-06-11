"""
R_04: 期限结构均值回复。

回归因子，捕捉极端Back/Contango结构均值回复信号。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import zscore


@register_factor
class R_04(BaseFactor):
    """R_04: 期限结构均值回复。"""

    name = "R_04"
    category = "回归"
    formula = "-1 * ZSCORE(CARRY, w)"
    dependencies = ["carry"]

    def compute(
        self,
        carry: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        计算 R_04 因子。

        Args:
            carry: Carry因子序列
            **kwargs: zscore_window（引擎注入）

        Returns:
            R_04 因子值序列
        """
        # P0整改：zscore_window 从 kwargs 接收，>0 时为扩张窗口
        zscore_window = kwargs.get("zscore_window", 0)
        if zscore_window is None or zscore_window <= 0:
            window_arg = None
        else:
            window_arg = zscore_window
        return -1 * zscore(carry, window=window_arg)

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
