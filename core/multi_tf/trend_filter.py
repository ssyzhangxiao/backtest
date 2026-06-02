"""
多时间框架趋势判断模块。

构建周频/月频趋势判断，实现趋势强度指标：
  - ADX > 25：趋势存在
  - 均线排列：MA5 > MA20 > MA60 = 多头排列
  - MACD状态：DIF > DEA 且柱状线递增 = 多头

规则11要求：过滤后交易次数减少>30%且胜率提升>5%方为有效。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from utils.indicators import compute_adx as _compute_adx

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """趋势方向枚举。"""

    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0


@dataclass
class TrendResult:
    """趋势判断结果。"""

    direction: TrendDirection = TrendDirection.NEUTRAL
    adx_value: float = 0.0
    ma_alignment: TrendDirection = TrendDirection.NEUTRAL
    macd_signal: TrendDirection = TrendDirection.NEUTRAL
    strength: float = 0.0

    def summary(self) -> str:
        """返回趋势摘要。"""
        return (
            f"方向={self.direction.value} 强度={self.strength:.2f} "
            f"ADX={self.adx_value:.1f} "
            f"MA排列={self.ma_alignment.value} MACD={self.macd_signal.value}"
        )


class MultiTFFilter:
    """
    多时间框架趋势过滤器。

    综合ADX、均线排列和MACD三个指标判断周频/月频趋势方向。
    三个指标投票决定最终趋势方向。

    用法:
        mtf = MultiTFFilter()
        result = mtf.evaluate_weekly(close_weekly, high_weekly, low_weekly)
        if result.direction == TrendDirection.BULLISH:
            # 周频多头趋势，允许日频做多
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        ma_short: int = 5,
        ma_medium: int = 20,
        ma_long: int = 60,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal_period: int = 9,
    ):
        """
        初始化多时间框架过滤器。

        Args:
            adx_period: ADX计算周期
            adx_threshold: ADX趋势阈值
            ma_short: 短期均线周期
            ma_medium: 中期均线周期
            ma_long: 长期均线周期
            macd_fast: MACD快线周期
            macd_slow: MACD慢线周期
            macd_signal_period: MACD信号线周期
        """
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.ma_short = ma_short
        self.ma_medium = ma_medium
        self.ma_long = ma_long
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal_period

    def compute_adx(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> float:
        """
        计算ADX（平均趋向指数）。

        委托给公共工具函数 utils.indicators.compute_adx。

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        Returns:
            ADX值
        """
        adx_val, _, _ = _compute_adx(high, low, close, period=self.adx_period)
        return adx_val

    def compute_ma_alignment(self, close: np.ndarray) -> TrendDirection:
        """
        判断均线排列方向。

        MA5 > MA20 > MA60 = 多头排列
        MA5 < MA20 < MA60 = 空头排列
        其他 = 中性

        Args:
            close: 收盘价序列

        Returns:
            趋势方向
        """
        c = np.asarray(close, dtype=float)
        if len(c) < self.ma_long:
            return TrendDirection.NEUTRAL

        ma_s = np.mean(c[-self.ma_short :])
        ma_m = np.mean(c[-self.ma_medium :])
        ma_l = np.mean(c[-self.ma_long :])

        if ma_s > ma_m > ma_l:
            return TrendDirection.BULLISH
        elif ma_s < ma_m < ma_l:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL

    def compute_macd_signal(self, close: np.ndarray) -> TrendDirection:
        """
        判断MACD信号方向。

        DIF > DEA 且柱状线递增 = 多头
        DIF < DEA 且柱状线递减 = 空头
        其他 = 中性

        Args:
            close: 收盘价序列

        Returns:
            趋势方向
        """
        c = np.asarray(close, dtype=float)
        if len(c) < self.macd_slow + self.macd_signal_period:
            return TrendDirection.NEUTRAL

        # EMA计算
        series = pd.Series(c)
        ema_fast = series.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = series.ewm(span=self.macd_slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.macd_signal_period, adjust=False).mean()
        hist = dif - dea

        if len(hist) < 2:
            return TrendDirection.NEUTRAL

        dif_val = float(dif.iloc[-1])
        dea_val = float(dea.iloc[-1])
        hist_now = float(hist.iloc[-1])
        hist_prev = float(hist.iloc[-2])

        if dif_val > dea_val and hist_now > hist_prev:
            return TrendDirection.BULLISH
        elif dif_val < dea_val and hist_now < hist_prev:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL

    def evaluate(
        self,
        close: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
    ) -> TrendResult:
        """
        综合评估趋势方向。

        三个指标投票：ADX、均线排列、MACD。
        至少2个同方向才确定趋势。

        Args:
            close: 收盘价序列
            high: 最高价序列（ADX需要）
            low: 最低价序列（ADX需要）

        Returns:
            TrendResult 趋势判断结果
        """
        # ADX
        adx_value = 0.0
        adx_signal = TrendDirection.NEUTRAL
        if high is not None and low is not None:
            adx_value = self.compute_adx(high, low, close)
            if adx_value > self.adx_threshold:
                # ADX只判断趋势强度，方向由价格决定
                if len(close) > 1 and close[-1] > close[-2]:
                    adx_signal = TrendDirection.BULLISH
                elif len(close) > 1 and close[-1] < close[-2]:
                    adx_signal = TrendDirection.BEARISH

        # 均线排列
        ma_signal = self.compute_ma_alignment(close)

        # MACD
        macd_signal = self.compute_macd_signal(close)

        # 投票
        votes = [adx_signal, ma_signal, macd_signal]
        bull_count = sum(1 for v in votes if v == TrendDirection.BULLISH)
        bear_count = sum(1 for v in votes if v == TrendDirection.BEARISH)

        if bull_count >= 2:
            direction = TrendDirection.BULLISH
        elif bear_count >= 2:
            direction = TrendDirection.BEARISH
        else:
            direction = TrendDirection.NEUTRAL

        # 趋势强度：投票一致性 * ADX归一化
        consistency = max(bull_count, bear_count) / 3.0
        adx_strength = min(adx_value / 50.0, 1.0)
        strength = consistency * 0.6 + adx_strength * 0.4

        return TrendResult(
            direction=direction,
            adx_value=adx_value,
            ma_alignment=ma_signal,
            macd_signal=macd_signal,
            strength=strength,
        )

    def evaluate_weekly(
        self,
        close_daily: np.ndarray,
        high_daily: Optional[np.ndarray] = None,
        low_daily: Optional[np.ndarray] = None,
    ) -> TrendResult:
        """
        基于日频数据计算周频趋势。

        将日频数据聚合为周频后评估。

        Args:
            close_daily: 日频收盘价
            high_daily: 日频最高价
            low_daily: 日频最低价

        Returns:
            TrendResult 周频趋势判断结果
        """
        # 简单聚合：每5个交易日取最后一个
        close_w = close_daily[4::5] if len(close_daily) >= 5 else close_daily[-1:]

        high_w = None
        low_w = None
        if high_daily is not None and low_daily is not None:
            # 每周取最高/最低
            n_weeks = len(close_daily) // 5
            high_w = (
                np.array(
                    [np.max(high_daily[i * 5 : (i + 1) * 5]) for i in range(n_weeks)]
                )
                if n_weeks > 0
                else high_daily[-1:]
            )

            low_w = (
                np.array(
                    [np.min(low_daily[i * 5 : (i + 1) * 5]) for i in range(n_weeks)]
                )
                if n_weeks > 0
                else low_daily[-1:]
            )

        return self.evaluate(close_w, high_w, low_w)
