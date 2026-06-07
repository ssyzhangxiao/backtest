"""
H_05: 三重共振因子。

高阶复合因子，期限结构、持仓量趋势、价格突破方向三者同向共振。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delta, sign, zscore


@register_factor
class H_05(BaseFactor):
    """H_05: 三重共振因子。"""

    name = "H_05"
    category = "高阶复合"
    formula = "ZSCORE(CARRY) * ZSCORE(DELTA(OI_SAFE,5)) * SIGN(DELTA(CLOSE,5))"
    dependencies = ["carry", "oi_safe", "close"]

    # P1整改：硬编码窗口改为类属性
    oi_delta_window: int = 5
    close_delta_window: int = 5

    def compute(
        self,
        carry: np.ndarray,
        oi_safe: np.ndarray,
        close: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        计算 H_05 因子。

        Args:
            carry: Carry 期限结构因子
            oi_safe: 安全持仓量序列
            close: 收盘价序列
            **kwargs: zscore_window（引擎注入），>0 时为扩张窗口

        Returns:
            H_05 因子值序列
        """
        # P0整改：zscore_window 从 kwargs 接收（引擎已注入）
        zscore_window = kwargs.get("zscore_window", 0)
        if zscore_window is None or zscore_window <= 0:
            window_arg = None
        else:
            window_arg = zscore_window
        z_carry = zscore(carry, window=window_arg)
        z_delta_oi = zscore(delta(oi_safe, self.oi_delta_window), window=window_arg)
        price_dir = sign(delta(close, self.close_delta_window))
        return z_carry * z_delta_oi * price_dir

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
