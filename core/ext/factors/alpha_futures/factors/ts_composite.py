"""
TS_composite: 波动率-动量合成因子（替代原期限结构合成）。

将TS_01(波动率比)、TS_02(价格加速度)、TS_03(位置因子)等权合成，
作为"期限结构"类别的综合信号。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import ema, zscore, delta, safe_div


@register_factor
class TS_composite(BaseFactor):
    """TS_composite: 波动率-动量合成因子。"""

    name = "TS_composite"
    category = "期限结构"
    formula = "EMA(3) of mean(zscore(TS_01), zscore(TS_02), zscore(TS_03))"
    dependencies = ["high", "low", "close"]

    smoothing_window: int = 3

    def compute(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        # TS_01: 日内波动率比
        hl = high.astype(float) - low.astype(float)
        ts01 = safe_div(hl, close)

        # TS_02: 价格加速度 (5日收益率差分)
        ret = safe_div(delta(close, 1), close)
        ts02 = delta(ret, 5)

        # TS_03: 10日位置因子
        import pandas as pd
        highest = pd.Series(high).rolling(10).max().values
        lowest = pd.Series(low).rolling(10).min().values
        denom = highest - lowest
        safe_denom = np.where(denom < 1e-8, np.nan, denom)
        ts03 = safe_div(close - lowest, safe_denom)

        composite = (zscore(ts01) + zscore(ts02) + zscore(ts03)) / 3.0
        return ema(composite, window=self.smoothing_window)

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)
