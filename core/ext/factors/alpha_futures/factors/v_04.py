"""
V_04: 持仓量均线差率（OI-MACD柱）。

波动率因子，持仓量长短均线发散→波动率将上升。

2026-06-11 重写（消除 OOS 过拟合）：
- OI 跳变检测：`abs(ΔOI / OI_prev) > 0.30` 处标 NaN
  - 触发场景：主力合约切换日 OI 跳变 -30%~-80%、交易所 OI 调整口径
- 跳变点由 pandas rolling.mean 跳过（min_periods=window 容忍 NaN）
- 公式保持不变：(MA9 - MA26) / MA12 * 100
"""

import numpy as np
import pandas as pd

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import mean, safe_div, winsorize


@register_factor
class V_04(BaseFactor):
    """V_04: 持仓量均线差率（OI-MACD柱）。"""

    name = "V_04"
    category = "波动率"
    formula = "(MEAN(OI_CLEANED,9)-MEAN(OI_CLEANED,26))/MEAN(OI_CLEANED,12)*100"
    dependencies = ["oi_safe"]

    # P1整改：硬编码窗口改为类属性
    fast_ma: int = 9
    slow_ma: int = 26
    ref_ma: int = 12

    # 2026-06-11 重写：跳变检测阈值（绝对值），0.30 = 30% 单日跳变视为换月/调整
    oi_jump_threshold: float = 0.30

    def _clean_oi_jumps(self, oi: np.ndarray) -> np.ndarray:
        """
        清理 OI 跳变：单日 |ΔOI/OI_prev| > threshold 处标 NaN。

        跳变点两侧各 +/-1 日也置 NaN（保守策略：单日跳变会污染前后窗口）。

        Returns:
            清洗后的 OI 序列（NaN 标记的位置由 rolling.mean 跳过）
        """
        oi_s = pd.Series(oi, dtype=float)
        # 计算 OI 变化率
        prev_oi = oi_s.shift(1)
        pct_change = (oi_s - prev_oi) / prev_oi.replace(0, np.nan)
        # 跳变点掩码
        jump_mask = pct_change.abs() > self.oi_jump_threshold
        # 跳变点及其前后一日一并标记
        neighbor_mask = jump_mask | jump_mask.shift(1) | jump_mask.shift(-1)
        # 替换为 NaN
        return oi_s.where(~neighbor_mask, np.nan).values

    def compute(
        self,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        oi_clean = self._clean_oi_jumps(oi_safe)
        oi_fast = mean(oi_clean, self.fast_ma)
        oi_slow = mean(oi_clean, self.slow_ma)
        oi_ref = mean(oi_clean, self.ref_ma)
        return safe_div(oi_fast - oi_slow, oi_ref) * 100

    def post_process(self, values: np.ndarray) -> np.ndarray:
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)
