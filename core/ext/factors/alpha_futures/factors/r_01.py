"""
R_01: 平滑日内涨跌与增仓率背离。

回归因子，捕捉"增仓滞涨"的顶部反转信号。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import corr, delta, log, zscore


@register_factor
class R_01(BaseFactor):
    """R_01: 平滑日内涨跌与增仓率背离。"""

    name = "R_01"
    category = "回归"
    formula = "-1 * CORR(ZSCORE(DELTA(LOG(OI_SAFE),1)), ZSCORE(INTRADAY_RET), 6)"
    dependencies = ["oi_safe", "intraday_ret"]

    # P1整改：硬编码窗口改为类属性
    corr_window: int = 6

    def compute(
        self,
        oi_safe: np.ndarray,
        intraday_ret: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        计算 R_01 因子。

        Args:
            oi_safe: 安全持仓量序列
            intraday_ret: 平滑日内收益率序列
            **kwargs: zscore_window（引擎注入）

        Returns:
            R_01 因子值序列
        """
        # P0整改：zscore_window 从 kwargs 接收
        zscore_window = kwargs.get("zscore_window", 0)
        if zscore_window is None or zscore_window <= 0:
            window_arg = None
        else:
            window_arg = zscore_window
        log_oi = log(oi_safe)
        z_delta_oi = zscore(delta(log_oi, 1), window=window_arg)
        z_intraday = zscore(intraday_ret, window=window_arg)
        return -1 * corr(z_delta_oi, z_intraday, self.corr_window)

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
