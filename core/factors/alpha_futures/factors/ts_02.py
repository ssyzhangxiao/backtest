"""
TS_02: 期限价差因子。

源自 term_structure.py 的 term_spread。
期限价差 = 近月价 - 远月价，直接反映期限结构。
"""
from typing import Optional

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor


@register_factor
class TS_02(BaseFactor):
    """TS_02: 期限价差。"""

    name = "TS_02"
    category = "期限结构"
    formula = "NEAR_PRICE - FAR_PRICE"
    dependencies = ["near_price", "far_price"]

    def compute(
        self,
        near_price: Optional[np.ndarray] = None,
        far_price: Optional[np.ndarray] = None,
        close: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        # 检查必需数据
        if near_price is None or far_price is None:
            # 如果没有近月/远月价格，返回全 NaN
            length = len(close) if close is not None else 100
            return np.full(length, np.nan, dtype=float)
        
        near = np.asarray(near_price, dtype=float)
        far = np.asarray(far_price, dtype=float)
        min_len = min(len(near), len(far))
        return near[-min_len:] - far[-min_len:]

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)