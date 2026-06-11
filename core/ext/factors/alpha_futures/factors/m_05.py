"""
M_05: 持仓量MACD指标。

资金流因子，经典量能指标的OI版，金叉/死叉提示资金面拐点。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import sma_ema


@register_factor
class M_05(BaseFactor):
    """M_05: 持仓量MACD指标。"""

    name = "M_05"
    category = "资金流"
    formula = "SMA(OI_SAFE,13,2)-SMA(OI_SAFE,27,2)-SMA(SMA(OI_SAFE,13,2)-SMA(OI_SAFE,27,2),10,2)"
    dependencies = ["oi_safe"]

    # P1整改：硬编码窗口改为类属性（MACD 标准参数：12/26/9，OI 适用 13/27/10）
    fast_period: int = 13
    slow_period: int = 27
    signal_period: int = 10
    ema_order: int = 2

    def compute(
        self,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        sma_fast = sma_ema(oi_safe, self.fast_period, self.ema_order)
        sma_slow = sma_ema(oi_safe, self.slow_period, self.ema_order)
        dif = sma_fast - sma_slow
        dea = sma_ema(dif, self.signal_period, self.ema_order)
        return dif - dea

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)