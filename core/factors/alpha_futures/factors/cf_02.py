"""
CF_02: 资金净流入/流出因子。

源自 capital_flow.py 的 capital_net_flow。
基于成交额方向加权，量化资金净流向。
"""

import numpy as np
import pandas as pd

from ..base_factor import BaseFactor
from ..factor_registry import register_factor


@register_factor
class CF_02(BaseFactor):
    """CF_02: 20日资金净流入/流出。"""

    name = "CF_02"
    category = "资金流"
    formula = "SUM(CLOSE * VOLUME * SIGN(RETURN), 20) / MEAN_ABS"
    dependencies = ["close", "volume"]

    # P1整改：硬编码窗口改为类属性
    rolling_window: int = 20

    def compute(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        c = np.asarray(close, dtype=float)
        v = np.asarray(volume, dtype=float)
        n = len(c)
        window = self.rolling_window
        result = np.full(n, np.nan, dtype=float)

        if n < window + 1:
            return result

        # P2整改：使用 np.concatenate 替代 np.insert（更清晰且零开销）
        # 日收益率方向（首日无收益，填 0）
        returns = np.diff(c) / c[:-1]
        returns = np.concatenate(([0.0], returns))

        # 方向加权成交额
        weighted_flow = c * v * np.sign(returns)

        # 滚动累计
        flow_series = pd.Series(weighted_flow)
        rolling_sum = flow_series.rolling(window=window, min_periods=window).sum()
        rolling_mean = flow_series.rolling(window=window, min_periods=window).mean()

        mean_abs = rolling_mean.abs()
        valid = mean_abs > 1e-10
        result_values = rolling_sum.values.copy()
        result_values[valid] = result_values[valid] / mean_abs[valid]
        result_values[~valid] = 0.0
        result_values[:window] = np.nan

        return result_values

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)