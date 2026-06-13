"""
波动率均值回复 CTA 策略 — 纯信号生成器。

三层信号映射：
  第一层（线性滤波）：全量滚动波动率（年化）
  第二层（状态变换）：波动率 z-score + 效率比(ER)动态窗口
  第三层（复合）：方向由趋势决定，强度由 z-score 缩放

配置参数:
  vol_window: 波动率计算窗口（默认 20）
  lookback:   z-score 统计窗口（默认 252）
  entry_z:    入场阈值（默认 1.2）
  vol_percentile: 波动率百分位门槛（默认 0.7）
  direction_window: 基础趋势方向判断窗口（默认 20）
  er_period:  效率比计算周期（默认 10）
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from core.strategies.cta.base import CTABaseStrategy
from core.strategies.cta.registry import register_cta_strategy

logger = logging.getLogger(__name__)


class VolMeanReversionStrategy(CTABaseStrategy):
    """波动率均值回复 CTA 策略 — 纯信号生成器。

    全量滚动波动率（正确 Welford 语义） + ER 动态窗口 + 二次确认。

    配置参数:
        vol_window: 波动率计算窗口（默认 20）
        lookback:   z-score 窗口（默认 252）
        entry_z:    入场阈值（默认 1.2）
        vol_percentile: 波动率百分位门槛（默认 0.7）
        direction_window: 基础趋势判断窗口（默认 20）
        er_period:  效率比计算周期（默认 10）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {
            "vol_window": 20,
            "lookback": 252,
            "entry_z": 1.2,
            "vol_percentile": 0.7,
            "direction_window": 20,
            "er_period": 10,
            **(config or {}),
        }
        super().__init__(merged)

    def compute_signal(
        self,
        symbol: str,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray | None = None,
        ctx: Any = None,
    ) -> float:
        """计算纯信号。

        Returns:
            信号 [-1, 1]:
              >0 做多（波动率高 + 趋势向下），<0 做空（波动率高 + 趋势向上）
              0 无信号
        """
        min_len = self.config["vol_window"] + self.config["lookback"] + 5
        if not self._validate(close, min_len=min_len):
            return 0.0

        vol_window = self.config["vol_window"]
        lookback = self.config["lookback"]
        entry_z = self.config["entry_z"]

        ret = np.diff(close) / close[:-1]
        if len(ret) < lookback + vol_window:
            return 0.0

        # ── 全量滚动波动率（向量化，无增量 bug） ──
        # 对每个 i，取 ret[i-vol_window:i] 的标准差 × sqrt(252)
        n_vol = len(ret) - vol_window + 1
        rolling_vol = np.empty(n_vol)
        for i in range(n_vol):
            rolling_vol[i] = float(np.std(ret[i: i + vol_window])) * np.sqrt(252)

        if len(rolling_vol) < lookback + 1:
            return 0.0

        current_vol = float(rolling_vol[-1])
        hist_vol = rolling_vol[-(lookback + 1): -1]

        if len(hist_vol) < 20:
            return 0.0

        mean = float(np.mean(hist_vol))
        std = float(np.std(hist_vol))
        if std <= 1e-10:
            return 0.0

        z = (current_vol - mean) / std

        # 波动率百分位过滤
        vol_percentile = self.config.get("vol_percentile", 0.7)
        vol_rank = float(np.sum(hist_vol < current_vol)) / len(hist_vol)

        if z < entry_z or vol_rank < vol_percentile:
            return 0.0

        # ── 效率比(ER)动态方向窗口 ──
        er_period = self.config.get("er_period", 10)
        base_window = self.config.get("direction_window", 20)

        if len(ret) > er_period + 5:
            net_change = ret[-er_period:].sum()
            gross_change = np.sum(np.abs(ret[-er_period:]))
            er = abs(net_change) / (gross_change + 1e-10)
            dynamic_window = max(10, int(base_window * (1.0 - er * 0.5)))
        else:
            dynamic_window = base_window

        # ── 二次确认：趋势方向连续 2 根同向 ──
        if len(ret) >= dynamic_window + 2:
            short_term_trend_1 = float(np.mean(ret[-dynamic_window:]))
            short_term_trend_2 = float(np.mean(ret[-(dynamic_window - 1):-1]))
            if short_term_trend_1 * short_term_trend_2 <= 0:
                return 0.0
        else:
            return 0.0

        signal_strength = min(1.0, z / entry_z)

        if short_term_trend_1 > 0:
            signal = -signal_strength  # 做空
        else:
            signal = signal_strength   # 做多

        self.set_state(symbol, "market_state", "oscillation")
        return signal


register_cta_strategy("vol_mean_reversion", VolMeanReversionStrategy)

__all__ = ["VolMeanReversionStrategy"]
