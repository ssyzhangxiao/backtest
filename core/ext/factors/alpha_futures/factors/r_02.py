"""
R_02: 最高价与增仓率滚动标准化5日相关性。

回归因子，捕捉"增仓滞涨"的顶部反转信号。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import corr, delta, zscore


@register_factor
class R_02(BaseFactor):
    """R_02: 最高价与增仓率滚动标准化5日相关性。"""

    name = "R_02"
    category = "回归"
    formula = "-1 * CORR(HIGH, ZSCORE(DELTA(OI_SAFE,1),20), 5)"
    dependencies = ["high", "oi_safe"]

    # P1整改：硬编码窗口改为类属性
    zscore_window_param: int = 20
    corr_window: int = 5

    def compute(
        self,
        high: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        计算 R_02 因子。

        Args:
            high: 最高价序列
            oi_safe: 安全持仓量序列
            **kwargs: 其他参数（未使用）

        Returns:
            R_02 因子值序列
        """
        delta_oi = delta(oi_safe, 1)
        z_delta_oi = zscore(delta_oi, window=self.zscore_window_param)
        return -1 * corr(high, z_delta_oi, self.corr_window)

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
