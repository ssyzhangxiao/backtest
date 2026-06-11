"""
CF_01: 持仓量变化率因子（已叠加价格方向）。

源自 capital_flow.py 的 position_change_rate。
计算持仓量相对于滚动均值的偏离度，再乘以同期价格方向：
  - 价涨 + 增仓 → 正（多头主动建仓）
  - 价涨 + 减仓 → 负（空头主动平仓，但价格未跌）
  - 价跌 + 减仓 → 正（多头主动平仓）
  - 价跌 + 增仓 → 负（空头主动建仓）
"""

import numpy as np
import pandas as pd

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delta, sign


@register_factor
class CF_01(BaseFactor):
    """CF_01: 持仓量相对均值的偏离度（方向修正）。"""

    name = "CF_01"
    category = "资金流"
    formula = "((OI_SAFE - MA(OI_SAFE, n)) / MA(OI_SAFE, n)) * SIGN(DELTA(CLOSE, n))"
    dependencies = ["oi_safe", "close"]

    # P1整改：硬编码窗口改为类属性
    ma_window: int = 5

    def compute(
        self,
        oi_safe: np.ndarray,
        close: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        计算 CF_01 因子（持仓量偏离度 × 价格方向）。

        Args:
            oi_safe: 安全持仓量序列
            close: 收盘价序列
            **kwargs: 其他参数（未使用）

        Returns:
            CF_01 因子值序列
        """
        oi = np.asarray(oi_safe, dtype=float)
        n = self.ma_window
        result = np.full_like(oi, np.nan, dtype=float)
        if len(oi) < n:
            return result
        oi_ma = pd.Series(oi).rolling(window=n, min_periods=n).mean().values
        valid = oi_ma > 0
        result[valid] = (oi[valid] - oi_ma[valid]) / oi_ma[valid]
        # 叠加价格方向：同期价格涨跌方向
        price_dir = sign(delta(close, n))
        return result * price_dir

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)