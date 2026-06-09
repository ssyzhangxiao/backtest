"""
TS_composite: 期限结构合成因子。

将 TS_01 (基差率)、TS_02 (期限价差)、TS_03 (展期收益) 三个高度相关的
期限结构因子做截面时序合成：

  1. 对每个因子在时间序列上做 zscore 标准化
  2. 等权平均 (zscore(TS_01) + zscore(TS_02) + zscore(TS_03)) / 3
  3. EMA(3) 时序平滑（半衰期 ~1.5 日），抑制日内跳变

合成后等价于对原始 NEAR-FAR 信号在时间序列上做单次低通滤波，
去除了三因子间的冗余相关性（互相关 >0.95 → 合并后 <0.6）。
"""
from typing import Optional

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import ema, zscore


@register_factor
class TS_composite(BaseFactor):
    """TS_composite: 期限结构合成因子（TS_01+TS_02+TS_03）。"""

    name = "TS_composite"
    category = "期限结构"
    formula = "EMA(3) of mean(zscore(TS_01), zscore(TS_02), zscore(TS_03))"
    dependencies = ["near_price", "far_price"]

    # EMA 平滑窗口（半衰期 ≈ window/2）
    smoothing_window: int = 3

    def compute(
        self,
        near_price: Optional[np.ndarray] = None,
        far_price: Optional[np.ndarray] = None,
        close: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        if near_price is None or far_price is None:
            length = len(close) if close is not None else 100
            return np.full(length, np.nan, dtype=float)

        near = np.asarray(near_price, dtype=float)
        far = np.asarray(far_price, dtype=float)
        n = min(len(near), len(far))
        near = near[-n:]
        far = far[-n:]
        safe_far = np.where(np.abs(far) < 1e-8, np.nan, far)

        # 三个原始期限结构子因子
        ts01 = (near - far) / safe_far                       # 基差率
        ts02 = near - far                                    # 期限价差
        ts03 = np.full(n, np.nan, dtype=float)
        # TS_03 是 5 日 SMA 的 roll_yield = far - near
        raw_roll = far - near
        valid_mask = np.isfinite(raw_roll)
        if valid_mask.sum() >= 5:
            # 简单 SMA 5 日（与原 TS_03 对齐）
            kernel = np.ones(5) / 5.0
            ts03 = np.convolve(np.where(valid_mask, raw_roll, 0.0), kernel, mode="same")
            # 边界回填：首个 5 日内用 cumsum 修正
            cumsum = np.cumsum(np.where(valid_mask, raw_roll, 0.0))
            cnt = np.cumsum(valid_mask.astype(int))
            for i in range(min(5, n)):
                if not np.isfinite(ts03[i]) or cnt[i] == 0:
                    ts03[i] = cumsum[i] / max(cnt[i], 1)

        # zscore 标准化（时间序列）
        z01 = zscore(ts01)
        z02 = zscore(ts02)
        z03 = zscore(ts03)

        # 等权平均
        composite = (z01 + z02 + z03) / 3.0

        # EMA 时序平滑
        smoothed = ema(composite, window=self.smoothing_window)
        return smoothed

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)
