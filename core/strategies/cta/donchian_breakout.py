"""
唐奇安通道突破策略 — 纯信号生成器。

三层信号映射：
  第一层（线性滤波）：N 日高低点通道（极值滤波）
  第二层（状态变换）：标准 ADX(+DM/-DM/TR) + 通道突破判定
  第三层（复合）：连续信号 × 动量因子，动态 ATR 乘数

（规则31管线）增强功能：
  - 标准 ADX（+DM/-DM/TR）替代简化版
  - 信号强度加入动量因子：1 + 0.2×(今日涨幅/ATR)
  - 动态 ATR 乘数：低波 0.3，高波 0.7

输出约定：
  - 纯信号 [-1, 1] + market_state（"trend"/"oscillation"）
  - 不维护持仓状态，不实现退出逻辑（执行器统一管理）

配置参数:
  entry_lookback: 入场通道周期（默认 20）
  atr_window:     ATR 计算窗口（默认 14）
  atr_entry_mult: ATR 确认倍数（默认 0.5，低波 0.3，高波 0.7）
  trend_filter_ma:趋势过滤 MA 窗口（默认 60，0=禁用）
  adx_window:     ADX 计算窗口（默认 14）
  adx_threshold:  ADX 入场阈值（默认 20）
  momentum_factor:动量因子权重（默认 0.2）
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from core.strategies.cta.base import CTABaseStrategy
from core.strategies.cta.registry import register_cta_strategy


class DonchianBreakoutStrategy(CTABaseStrategy):
    """唐奇安通道突破策略 — 纯信号生成器。

    标准 ADX 趋势过滤 + 动量因子增强信号 + 动态 ATR 确认。

    配置参数:
        entry_lookback: 入场通道周期（默认 20）
        atr_window:     ATR 计算窗口（默认 14）
        atr_entry_mult: ATR 确认倍数（默认 0.5）
        trend_filter_ma:趋势过滤 MA 窗口（默认 60，0=禁用）
        adx_window:     ADX 计算窗口（默认 14）
        adx_threshold:  ADX 入场阈值（默认 20）
        momentum_factor:动量因子权重（默认 0.2）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {
            "entry_lookback": 20,
            "atr_window": 14,
            "atr_entry_mult": 0.5,
            "trend_filter_ma": 60,
            "adx_window": 14,
            "adx_threshold": 20,
            "momentum_factor": 0.2,
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
              >0 做多，<0 做空，0 无信号
        """
        min_len = self.config["entry_lookback"] + 5
        if not self._validate(close, min_len=min_len):
            return 0.0

        # ── 标准 ADX 趋势过滤 ──
        trend_allowed, adx_val = self._check_trend_adx(close, high, low)

        # ── 动态 ATR 乘数（低波/高波自适应） ──
        atr_base = self._compute_atr(close, high, low, self.config["atr_window"])
        # 波动率状态：用 ATR 占价格的比例判断
        vol_regime = atr_base / (float(close[-1]) + 1e-10)
        # 低波：<1%, 高波：>3%
        if vol_regime < 0.01:
            atr_entry_mult = 0.3  # 低波收紧
        elif vol_regime > 0.03:
            atr_entry_mult = 0.7  # 高波放宽
        else:
            atr_entry_mult = self.config["atr_entry_mult"]

        # 通道极值（排除当前 bar）
        lookback = self.config["entry_lookback"]
        prev_highs = high[-(lookback + 1):-1]
        prev_lows = low[-(lookback + 1):-1]
        if len(prev_highs) == 0 or len(prev_lows) == 0:
            return 0.0
        highest_high = float(np.max(prev_highs))
        lowest_low = float(np.min(prev_lows))

        current_close = float(close[-1])
        prev_close = float(close[-2]) if len(close) >= 2 else current_close
        atr_entry = atr_entry_mult

        # ── 动量因子 ──
        mom_factor = self.config.get("momentum_factor", 0.2)
        # 今日涨幅 / ATR
        daily_ret = (current_close - prev_close) / (prev_close + 1e-10)
        ret_atr_ratio = daily_ret / (atr_base / (prev_close + 1e-10) + 1e-10)
        momentum_boost = 1.0 + mom_factor * np.clip(ret_atr_ratio, -3.0, 3.0)

        # ── 向上突破 ──
        if trend_allowed >= 0 and current_close > highest_high + atr_entry * atr_base:
            # 突破强度 × 动量因子
            break_dist = (current_close - highest_high) / (atr_entry * atr_base + 1e-10)
            signal = min(1.0, break_dist * momentum_boost)
            self.set_state(symbol, "market_state", "trend")
            # 存储 ADX 供执行器参考
            self.set_state(symbol, "adx", adx_val)
            return signal

        # ── 向下突破 ──
        if trend_allowed <= 0 and current_close < lowest_low - atr_entry * atr_base:
            break_dist = (lowest_low - current_close) / (atr_entry * atr_base + 1e-10)
            signal = -min(1.0, break_dist * momentum_boost)
            self.set_state(symbol, "market_state", "trend")
            self.set_state(symbol, "adx", adx_val)
            return signal

        return 0.0

    def _check_trend_adx(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray
    ) -> tuple[int, float]:
        """标准 ADX 趋势过滤。

        Returns:
            (trend_allowed, adx_value)
            trend_allowed: 1(多), -1(空), 0(不明)
        """
        adx_window = self.config["adx_window"]
        trend_ma = self.config.get("trend_filter_ma", 0)
        trend_allowed = 0

        # MA 方向过滤
        if trend_ma > 0 and len(close) > trend_ma + 2:
            long_ma = float(np.mean(close[-trend_ma:]))
            prev_long_ma = float(np.mean(close[-(trend_ma + 1):-1]))
            trend_up = long_ma >= prev_long_ma
            trend_allowed = 1 if trend_up else -1

        # 标准 ADX
        if len(close) > adx_window * 2 + 5:
            adx_val, plus_di, minus_di = self._compute_standard_adx(
                close, high, low, adx_window
            )

            if adx_val < self.config["adx_threshold"]:
                # ADX 低 → 无趋势，不允许突破
                return 0, adx_val

            # ADX 高 → 用 +DI/-DI 方向限制
            if plus_di > minus_di:
                trend_allowed = 1
            else:
                trend_allowed = -1

            return trend_allowed, adx_val

        return trend_allowed, 0.0

    def _compute_standard_adx(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray, window: int
    ) -> tuple[float, float, float]:
        """标准 ADX 计算（+DM/-DM/TR）。

        Returns:
            (adx, +DI, -DI)
        """
        n = min(window + 2, len(close) - 1)
        if n < window:
            return 0.0, 0.0, 0.0

        # TR
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )

        # +DM / -DM
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where(
            (up_move > down_move) & (up_move > 0), up_move, 0.0
        )
        minus_dm = np.where(
            (down_move > up_move) & (down_move > 0), down_move, 0.0
        )

        # 平滑（Wilder 方法：累计平均）
        def wilder_smooth(arr: np.ndarray, w: int) -> np.ndarray:
            out = np.zeros(w)
            out[0] = float(np.mean(arr[-w:]))
            for i in range(1, w):
                out[i] = (out[i - 1] * (w - 1) + arr[-(w - i)]) / w
            return out

        smooth_w = min(window, len(tr) - 1)
        if smooth_w < window:
            smooth_w = window

        tr_s = wilder_smooth(tr, smooth_w)
        pdm_s = wilder_smooth(plus_dm, smooth_w)
        mdm_s = wilder_smooth(minus_dm, smooth_w)

        # +DI / -DI
        plus_di = 100.0 * pdm_s[-1] / (tr_s[-1] + 1e-10)
        minus_di = 100.0 * mdm_s[-1] / (tr_s[-1] + 1e-10)

        # DX = |+DI - -DI| / (+DI + -DI)
        dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

        # ADX = SMA of DX
        adx = float(np.mean(dx))

        return adx, plus_di, minus_di

    @staticmethod
    def _compute_atr(
        close: np.ndarray, high: np.ndarray, low: np.ndarray, window: int
    ) -> float:
        if len(close) < 2:
            return 0.0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        if len(tr) < window:
            return float(np.mean(tr)) if len(tr) > 0 else 0.0
        return float(np.mean(tr[-window:]))


register_cta_strategy("donchian_breakout", DonchianBreakoutStrategy)

__all__ = ["DonchianBreakoutStrategy"]
