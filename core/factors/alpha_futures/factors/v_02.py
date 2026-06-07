"""
V_02: 平滑日内波动率与5日增仓乘积。

波动率因子，高波动+持续增仓=确认趋势行情。

P 整改（2026-06-07）：std 使用 min_periods=10 兜底，避免涨跌停日
INTRADAY_RET 全零导致 std=0 误判为"无波动"。
"""

import numpy as np
import pandas as pd

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delta


@register_factor
class V_02(BaseFactor):
    """V_02: 平滑日内波动率与5日增仓乘积。"""

    name = "V_02"
    category = "波动率"
    formula = "STD(INTRADAY_RET, 20) * DELTA(OI_SAFE, 5)"
    dependencies = ["intraday_ret", "oi_safe"]

    # P1整改：硬编码窗口改为类属性
    std_window: int = 20
    oi_delta_window: int = 5
    # min_periods 兜底：避免涨跌停日窗口全零导致 std=0
    std_min_periods: int = 10

    def compute(
        self,
        intraday_ret: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        计算 V_02 因子。

        使用 min_periods=self.std_min_periods 而非全窗口，让涨跌停
        日（INTRADAY_RET=0）混入后仍能输出合理 std 值，而非 0。
        """
        vol = (
            pd.Series(np.asarray(intraday_ret, dtype=float))
            .rolling(window=self.std_window, min_periods=self.std_min_periods)
            .std()
            .values
        )
        return vol * delta(oi_safe, self.oi_delta_window)

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)