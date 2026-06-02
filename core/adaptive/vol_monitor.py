"""
波动率监测模块。

实时计算市场波动率指标并判定波动率regime：
  - HV（历史波动率）：20日/60日/120日滚动标准差
  - ATR（平均真实波幅）：14日ATR及其分位数
  - 波动率regime判定：低/中/高三档

规则10要求：regime切换频率不得超过每月1次。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VolRegime(Enum):
    """波动率regime枚举。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class VolMonitorResult:
    """波动率监测结果。"""

    hv_20d: float = 0.0
    hv_60d: float = 0.0
    hv_120d: float = 0.0
    atr_14d: float = 0.0
    atr_percentile: float = 0.5
    regime: VolRegime = VolRegime.MEDIUM
    regime_changed: bool = False

    def summary(self) -> str:
        """返回监测摘要。"""
        return (
            f"HV(20d)={self.hv_20d:.4f} HV(60d)={self.hv_60d:.4f} "
            f"ATR(14d)={self.atr_14d:.2f} Pct={self.atr_percentile:.2f} "
            f"Regime={self.regime.value}"
        )


class VolatilityMonitor:
    """
    波动率监测器。

    计算历史波动率、ATR和波动率regime。
    regime切换频率受限（规则10：每月不超过1次）。

    用法:
        monitor = VolatilityMonitor()
        result = monitor.update(close_prices, high_prices, low_prices)
        if result.regime == VolRegime.HIGH:
            # 高波动率环境处理
    """

    def __init__(
        self,
        hv_windows: Optional[List[int]] = None,
        atr_window: int = 14,
        low_percentile: float = 0.25,
        high_percentile: float = 0.75,
        min_switch_interval_days: int = 21,
        lookback_for_percentile: int = 252,
    ):
        """
        初始化波动率监测器。

        Args:
            hv_windows: 历史波动率窗口列表，默认[20, 60, 120]
            atr_window: ATR计算窗口，默认14
            low_percentile: 低波动率分位数阈值，默认0.25
            high_percentile: 高波动率分位数阈值，默认0.75
            min_switch_interval_days: regime最短切换间隔（交易日），默认21（约1个月）
            lookback_for_percentile: ATR分位数计算的回看期，默认252（1年）
        """
        self.hv_windows = hv_windows or [20, 60, 120]
        self.atr_window = atr_window
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.min_switch_interval_days = min_switch_interval_days
        self.lookback_for_percentile = lookback_for_percentile

        # 状态追踪
        self._current_regime: VolRegime = VolRegime.MEDIUM
        self._last_switch_day: int = 0
        self._day_counter: int = 0
        self._atr_history: List[float] = []

    @property
    def current_regime(self) -> VolRegime:
        """当前波动率regime。"""
        return self._current_regime

    def compute_hv(self, close: np.ndarray, window: int) -> float:
        """
        计算历史波动率（年化）。

        HV = std(daily_returns) * sqrt(252)

        Args:
            close: 收盘价序列
            window: 滚动窗口（交易日）

        Returns:
            年化历史波动率
        """
        c = np.asarray(close, dtype=float)
        if len(c) < window + 1:
            return 0.0

        returns = np.diff(c[-window - 1:]) / c[-window - 1:-1]
        if len(returns) < 2:
            return 0.0

        return float(np.std(returns) * np.sqrt(252))

    def compute_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> float:
        """
        计算ATR（平均真实波幅）。

        TR = max(H-L, |H-C_prev|, |L-C_prev|)
        ATR = MA(TR, window)

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        Returns:
            ATR值
        """
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        c = np.asarray(close, dtype=float)

        if len(c) < self.atr_window + 1:
            return 0.0

        # 计算True Range
        tr = np.maximum(
            h[1:] - l[1:],
            np.maximum(
                np.abs(h[1:] - c[:-1]),
                np.abs(l[1:] - c[:-1]),
            ),
        )

        if len(tr) < self.atr_window:
            return float(np.mean(tr))

        # 简单移动平均
        return float(np.mean(tr[-self.atr_window:]))

    def compute_atr_percentile(self, atr_value: float) -> float:
        """
        计算ATR在历史中的分位数。

        Args:
            atr_value: 当前ATR值

        Returns:
            分位数（0~1）
        """
        self._atr_history.append(atr_value)

        # 保留回看期内的数据
        if len(self._atr_history) > self.lookback_for_percentile:
            self._atr_history = self._atr_history[-self.lookback_for_percentile:]

        if len(self._atr_history) < 20:
            return 0.5

        arr = np.array(self._atr_history)
        return float(np.mean(arr <= atr_value))

    def determine_regime(self, atr_percentile: float) -> VolRegime:
        """
        根据ATR分位数判定波动率regime。

        规则10：regime切换频率不得超过每月1次。

        Args:
            atr_percentile: ATR分位数

        Returns:
            波动率regime
        """
        if atr_percentile < self.low_percentile:
            target = VolRegime.LOW
        elif atr_percentile > self.high_percentile:
            target = VolRegime.HIGH
        else:
            target = VolRegime.MEDIUM

        # 切换频率限制
        days_since_switch = self._day_counter - self._last_switch_day
        if target != self._current_regime and days_since_switch < self.min_switch_interval_days:
            logger.debug(
                f"Regime切换被抑制：{self._current_regime.value}→{target.value}，"
                f"距上次切换仅{days_since_switch}天（最小间隔{self.min_switch_interval_days}天）"
            )
            return self._current_regime

        return target

    def update(
        self,
        close: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
    ) -> VolMonitorResult:
        """
        更新波动率监测。

        Args:
            close: 收盘价序列
            high: 最高价序列（可选，用于ATR）
            low: 最低价序列（可选，用于ATR）

        Returns:
            VolMonitorResult 监测结果
        """
        self._day_counter += 1

        # 计算HV
        hv_values: Dict[int, float] = {}
        for w in self.hv_windows:
            hv_values[w] = self.compute_hv(close, w)

        # 计算ATR
        atr_value = 0.0
        atr_pct = 0.5
        if high is not None and low is not None:
            atr_value = self.compute_atr(high, low, close)
            atr_pct = self.compute_atr_percentile(atr_value)

        # 判定regime
        new_regime = self.determine_regime(atr_pct)
        regime_changed = new_regime != self._current_regime

        if regime_changed:
            logger.info(
                f"Regime切换：{self._current_regime.value}→{new_regime.value} "
                f"(ATR分位数={atr_pct:.2f})"
            )
            self._current_regime = new_regime
            self._last_switch_day = self._day_counter

        return VolMonitorResult(
            hv_20d=hv_values.get(20, 0.0),
            hv_60d=hv_values.get(60, 0.0),
            hv_120d=hv_values.get(120, 0.0),
            atr_14d=atr_value,
            atr_percentile=atr_pct,
            regime=self._current_regime,
            regime_changed=regime_changed,
        )

    def reset(self):
        """重置监测器状态。"""
        self._current_regime = VolRegime.MEDIUM
        self._last_switch_day = 0
        self._day_counter = 0
        self._atr_history.clear()
