"""
TS_03: 展期收益因子。

源自 term_structure.py 的 roll_yield。
展期收益 = 远月价 - 近月价，正=远月升水(contango)，负=远月贴水(backwardation)。
"""
from typing import Optional

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import sma


@register_factor
class TS_03(BaseFactor):
    """TS_03: 展期收益（5日平滑）。"""

    name = "TS_03"
    category = "期限结构"
    formula = "SMA(FAR_PRICE - NEAR_PRICE, 5)"
    dependencies = ["near_price", "far_price"]

    # P1整改：硬编码窗口改为类属性，可被子类覆盖或从 config 注入
    smoothing_window: int = 5

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
        raw = far[-min_len:] - near[-min_len:]

        # P1整改：使用 sma 算子替代手写 np.convolve + 边缘循环
        # sma 等价于"扩张窗口边界+有效窗口中心"，无前瞻性
        return sma(raw, self.smoothing_window)

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)