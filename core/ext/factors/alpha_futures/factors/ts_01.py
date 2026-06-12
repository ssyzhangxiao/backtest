"""
TS_01: 短期波动率比（替代原基差率，适配无期限结构数据）。

衡量日内波动幅度与价格水平的比值，高值表示市场情绪激烈。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor


@register_factor
class TS_01(BaseFactor):
    """TS_01: 短期波动率比。"""

    name = "TS_01"
    category = "期限结构"
    formula = "(HIGH - LOW) / CLOSE"
    dependencies = ["high", "low", "close"]

    def compute(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        hl = high.astype(float) - low.astype(float)
        safe_close = np.where(np.abs(close) < 1e-8, np.nan, close)
        return hl / safe_close

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)
