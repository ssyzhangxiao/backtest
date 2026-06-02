"""
时间框架信号过滤器。

日频信号与周频趋势方向一致时才执行交易，不一致时跳过或减仓。

规则11要求：
  - 周频权重60%，日频权重40%
  - 冲突场景占比<40%，冲突时盈亏比>1.0
"""

from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np

from .trend_filter import MultiTFFilter, TrendDirection, TrendResult

logger = logging.getLogger(__name__)

# 权重配置（规则11）
WEEKLY_WEIGHT = 0.6
DAILY_WEIGHT = 0.4


@dataclass
class FilterResult:
    """信号过滤结果。"""

    # 原始日频信号：1=做多, -1=做空, 0=不交易
    raw_signal: int = 0
    # 周频趋势方向
    weekly_trend: TrendDirection = TrendDirection.NEUTRAL
    # 过滤后信号
    filtered_signal: int = 0
    # 仓位调整比例（0~1）
    position_scale: float = 1.0
    # 是否冲突
    is_conflict: bool = False
    # 综合得分
    combined_score: float = 0.0

    def summary(self) -> str:
        """返回过滤摘要。"""
        conflict_str = "⚠️冲突" if self.is_conflict else "✅一致"
        return (
            f"日频={self.raw_signal} 周频={self.weekly_trend.value} "
            f"→ 过滤后={self.filtered_signal} 仓位={self.position_scale:.0%} "
            f"{conflict_str}"
        )


class SignalFilter:
    """
    时间框架信号过滤器。

    根据周频趋势过滤日频信号：
      - 日频多 + 周频多 → 全仓做多
      - 日频多 + 周频空 → 1/3仓做多（试探性）
      - 日频空 + 周频多 → 不交易
      - 日频空 + 周频空 → 全仓做空
      - 周频中性 → 日频信号正常执行

    用法:
        mtf = MultiTFFilter()
        weekly_trend = mtf.evaluate_weekly(close, high, low)
        sf = SignalFilter()
        result = sf.filter(daily_signal=1, weekly_trend=weekly_trend.direction)
    """

    def __init__(
        self,
        weekly_weight: float = WEEKLY_WEIGHT,
        daily_weight: float = DAILY_WEIGHT,
        conflict_position_scale: float = 0.33,
    ):
        """
        初始化信号过滤器。

        Args:
            weekly_weight: 周频权重（默认0.6）
            daily_weight: 日频权重（默认0.4）
            conflict_position_scale: 冲突时仓位比例（默认1/3）
        """
        self.weekly_weight = weekly_weight
        self.daily_weight = daily_weight
        self.conflict_position_scale = conflict_position_scale

    def filter(
        self,
        daily_signal: int,
        weekly_trend: TrendDirection,
    ) -> FilterResult:
        """
        过滤日频信号。

        Args:
            daily_signal: 日频信号（1=多, -1=空, 0=不交易）
            weekly_trend: 周频趋势方向

        Returns:
            FilterResult 过滤结果
        """
        if daily_signal == 0:
            return FilterResult(
                raw_signal=0,
                weekly_trend=weekly_trend,
                filtered_signal=0,
                position_scale=0.0,
                is_conflict=False,
                combined_score=0.0,
            )

        # 周频中性时，日频信号正常执行
        if weekly_trend == TrendDirection.NEUTRAL:
            return FilterResult(
                raw_signal=daily_signal,
                weekly_trend=weekly_trend,
                filtered_signal=daily_signal,
                position_scale=1.0,
                is_conflict=False,
                combined_score=daily_signal * 0.5,
            )

        # 日频信号方向与周频趋势的映射
        daily_dir = daily_signal  # 1 or -1
        weekly_dir = weekly_trend.value  # 1, -1, or 0

        is_aligned = (daily_dir * weekly_dir) > 0
        is_conflict = (daily_dir * weekly_dir) < 0

        if is_aligned:
            # 方向一致：全仓执行
            filtered_signal = daily_signal
            position_scale = 1.0
            combined_score = (
                daily_dir * self.daily_weight + weekly_dir * self.weekly_weight
            )
        elif is_conflict:
            # 方向冲突：根据规则11处理
            if daily_signal == 1 and weekly_trend == TrendDirection.BEARISH:
                # 日频多 + 周频空 → 1/3仓做多（试探性）
                filtered_signal = daily_signal
                position_scale = self.conflict_position_scale
            elif daily_signal == -1 and weekly_trend == TrendDirection.BULLISH:
                # 日频空 + 周频多 → 不交易
                filtered_signal = 0
                position_scale = 0.0
            else:
                filtered_signal = 0
                position_scale = 0.0

            combined_score = (
                daily_dir * self.daily_weight + weekly_dir * self.weekly_weight
            )
        else:
            filtered_signal = daily_signal
            position_scale = 1.0
            combined_score = daily_dir * 0.5

        return FilterResult(
            raw_signal=daily_signal,
            weekly_trend=weekly_trend,
            filtered_signal=filtered_signal,
            position_scale=position_scale,
            is_conflict=is_conflict,
            combined_score=combined_score,
        )
