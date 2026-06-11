"""
CF_03: 持仓量-价格背离信号因子（动态阈值）。

源自 capital_flow.py 的 oi_price_divergence。
量增价涨=多头，量增价跌=空头，量减价涨=虚涨偏空，量减价跌=虚跌偏多。

阈值改为基于滚动分位数的动态值（默认取滚动窗口内 70% 分位），
让低波动品种阈值自动收紧、高波动品种阈值自动放宽，避免静态
硬编码导致低波动品种误触发、高波动品种漏触发。
"""

import numpy as np
import pandas as pd

from ..base_factor import BaseFactor
from ..factor_registry import register_factor


def _rolling_abs_quantile(arr: np.ndarray, window: int, q: float) -> np.ndarray:
    """
    滚动 |arr| 的分位数（用于动态阈值）。

    使用 min_periods=window//2 以保证早期也能有合理估计。
    """
    s = pd.Series(np.abs(np.asarray(arr, dtype=float)))
    return s.rolling(window=window, min_periods=max(1, window // 2)).quantile(q).values


@register_factor
class CF_03(BaseFactor):
    """CF_03: 持仓量-价格背离信号（动态阈值）。"""

    name = "CF_03"
    category = "资金流"
    formula = "量价背离连续信号：量增价涨=+1, 量增价跌=-1, 量减价涨=-0.5, 量减价跌=+0.5"
    dependencies = ["close", "oi_safe"]

    # P1整改：硬编码阈值/窗口改为类属性，可配置
    oi_ma_window: int = 5
    # 静态阈值（向后兼容，作为"分位数不可用时"的回落值）
    oi_threshold: float = 0.03
    px_threshold: float = 0.005
    # 动态阈值配置
    dynamic_threshold: bool = True
    threshold_window: int = 60
    oi_quantile: float = 0.7
    px_quantile: float = 0.7

    def compute(
        self,
        close: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        c = np.asarray(close, dtype=float)
        oi = np.asarray(oi_safe, dtype=float)
        n = len(c)
        window = self.oi_ma_window
        result = np.full(n, np.nan, dtype=float)

        if n < window + 1:
            return result

        # 价格变化率
        price_change = np.full(n, 0.0)
        price_change[1:] = (c[1:] - c[:-1]) / c[:-1]

        # 持仓量变化率
        oi_change = np.full(n, 0.0)
        oi_ma = pd.Series(oi).rolling(window=window, min_periods=window).mean().values
        valid_oi = oi_ma > 0
        oi_change[valid_oi] = (oi[valid_oi] - oi_ma[valid_oi]) / oi_ma[valid_oi]

        # 动态阈值：滚动 |oi_change| / |price_change| 的分位数
        if self.dynamic_threshold and n >= self.threshold_window:
            oi_thresh_arr = _rolling_abs_quantile(
                oi_change, self.threshold_window, self.oi_quantile,
            )
            px_thresh_arr = _rolling_abs_quantile(
                price_change, self.threshold_window, self.px_quantile,
            )
        else:
            oi_thresh_arr = np.full(n, self.oi_threshold, dtype=float)
            px_thresh_arr = np.full(n, self.px_threshold, dtype=float)

        # 兜底：动态阈值若为 NaN（前期），回落静态值
        oi_thresh_arr = np.where(
            np.isnan(oi_thresh_arr), self.oi_threshold, oi_thresh_arr,
        )
        px_thresh_arr = np.where(
            np.isnan(px_thresh_arr), self.px_threshold, px_thresh_arr,
        )
        # 防止分位数为 0 导致除零
        oi_thresh_arr = np.maximum(oi_thresh_arr, 1e-6)
        px_thresh_arr = np.maximum(px_thresh_arr, 1e-6)

        for i in range(window, n):
            oi_c = oi_change[i]
            px_c = price_change[i]
            oi_t = float(oi_thresh_arr[i])
            px_t = float(px_thresh_arr[i])

            if abs(oi_c) < oi_t and abs(px_c) < px_t:
                result[i] = 0.0
                continue

            if oi_c > oi_t and px_c > px_t:
                result[i] = min(1.0, (oi_c + px_c) / (oi_t + px_t))
            elif oi_c > oi_t and px_c < -px_t:
                result[i] = -min(1.0, (oi_c + abs(px_c)) / (oi_t + px_t))
            elif oi_c < -oi_t and px_c > px_t:
                result[i] = -min(1.0, (abs(oi_c) + px_c) / (oi_t + px_t)) * 0.5
            elif oi_c < -oi_t and px_c < -px_t:
                result[i] = min(1.0, (abs(oi_c) + abs(px_c)) / (oi_t + px_t)) * 0.5
            else:
                result[i] = 0.0

        return result

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)