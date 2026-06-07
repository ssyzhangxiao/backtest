"""
V_01: 持仓量变化率（已叠加价格方向）。

波动率因子，捕捉"持仓异动 + 价格方向"的双重信号。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delay, delta, safe_div, sign


@register_factor
class V_01(BaseFactor):
    """V_01: 持仓量变化率（方向修正）。"""

    name = "V_01"
    category = "波动率"
    formula = "((OI_SAFE-DELAY(OI_SAFE,5))/DELAY(OI_SAFE,5)) * 100 * SIGN(DELTA(CLOSE,5))"
    dependencies = ["oi_safe", "close"]

    # P1整改：硬编码窗口改为类属性
    lookback_window: int = 5

    def compute(
        self,
        oi_safe: np.ndarray,
        close: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        计算 V_01 因子（持仓量变化率 × 价格方向）。

        原公式：safe_div(oi - delay(oi, n), delay(oi, n)) * 100
        改造：乘以 `sign(delta(close, n))`，使价涨OI增 → 正（多开），
              价涨OI减 → 负（空平/多平，被动），符合持仓量解释。

        Args:
            oi_safe: 安全持仓量序列
            close: 收盘价序列
            **kwargs: 其他参数（未使用）

        Returns:
            V_01 因子值序列
        """
        oi_change_pct = safe_div(
            oi_safe - delay(oi_safe, self.lookback_window),
            delay(oi_safe, self.lookback_window),
        ) * 100
        price_dir = sign(delta(close, self.lookback_window))
        return oi_change_pct * price_dir

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
