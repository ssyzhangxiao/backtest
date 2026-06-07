"""
TS_01: 基差率因子。

源自 term_structure.py 的 basis_rate。
基差率 = (近月价 - 远月价) / 远月价，正=近月升水。
"""
from typing import Optional

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor


@register_factor
class TS_01(BaseFactor):
    """TS_01: 基差率。"""

    name = "TS_01"
    category = "期限结构"
    formula = "(NEAR_PRICE - FAR_PRICE) / FAR_PRICE"
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
        near = near[-min_len:]
        far = far[-min_len:]
        safe_far = np.where(np.abs(far) < 1e-8, np.nan, far)
        return (near - far) / safe_far

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)