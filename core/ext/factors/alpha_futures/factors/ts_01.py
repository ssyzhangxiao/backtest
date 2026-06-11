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
            # 根治（P0 整改 2026-06-10）：必须有 close 才能确定返回长度；
            # 不再硬编码 fallback 100（历史上 100 是占位常量，会与实际数据长度不匹配
            # 触发 factor_engine 的 NaN right-align 兜底告警）。
            # 若调用方未传 close（理论不应发生），主动报错，避免静默错误。
            if close is None:
                raise ValueError(
                    "TS_01.compute 缺少 close 参数：必须传入 close 数组以确定返回长度"
                )
            return np.full(len(close), np.nan, dtype=float)
        
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