"""
追踪止损（Trailing Stop）。

两种模式：
  - 固定点数追踪：trail_price = highest - trail_pct * entry_price
  - ATR倍数追踪：trail_price = highest - N * ATR

规则13要求：追踪止损支持固定点数和ATR倍数两种模式。
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrailingStopResult:
    """追踪止损检查结果。"""

    triggered: bool = False
    stop_price: float = 0.0
    current_price: float = 0.0
    trailing_high: float = 0.0
    trailing_low: float = 0.0
    direction: str = "long"


class TrailingStop:
    """
    追踪止损管理器。

    支持固定百分比和ATR倍数两种追踪止损模式。
    多头：止损价 = max(前止损价, 最高价 - 追踪距离)
    空头：止损价 = min(前止损价, 最低价 + 追踪距离)

    用法:
        ts = TrailingStop(mode="atr", atr_multiplier=2.0)
        result = ts.check_long(entry_price=100, current_price=105, highest=108, atr=3.0)
    """

    def __init__(
        self,
        mode: str = "pct",
        trail_pct: float = 0.03,
        atr_multiplier: float = 2.0,
    ):
        """
        初始化追踪止损。

        Args:
            mode: 止损模式 "pct"（固定百分比）或 "atr"（ATR倍数）
            trail_pct: 追踪百分比（0~1），如0.03表示3%
            atr_multiplier: ATR倍数
        """
        self.mode = mode
        self.trail_pct = trail_pct
        self.atr_multiplier = atr_multiplier

        # 持仓追踪状态：{symbol: (trailing_high, trailing_low, stop_price)}
        self._state: Dict[str, Tuple[float, float, float]] = {}

    def _compute_trail_distance(self, atr_value: Optional[float] = None) -> float:
        """计算追踪距离。"""
        if self.mode == "atr" and atr_value is not None and atr_value > 0:
            return self.atr_multiplier * atr_value
        return self.trail_pct

    def check_long(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        highest_since_entry: float,
        atr_value: Optional[float] = None,
    ) -> TrailingStopResult:
        """
        检查多头追踪止损。

        Args:
            symbol: 品种代码
            entry_price: 入场价
            current_price: 当前价
            highest_since_entry: 入场以来最高价
            atr_value: ATR值（ATR模式需要）

        Returns:
            TrailingStopResult 止损检查结果
        """
        trail_dist = self._compute_trail_distance(atr_value)

        if self.mode == "pct":
            # 百分比模式：追踪距离 = 最高价 * trail_pct
            stop_price = highest_since_entry * (1 - trail_dist)
        else:
            # ATR模式：追踪距离 = N * ATR
            stop_price = highest_since_entry - trail_dist

        # 止损价只能上移不能下移
        prev_state = self._state.get(symbol)
        if prev_state is not None:
            prev_stop = prev_state[2]
            stop_price = max(stop_price, prev_stop)

        # 更新状态
        trailing_high = max(highest_since_entry, prev_state[0] if prev_state else entry_price)
        self._state[symbol] = (trailing_high, prev_state[1] if prev_state else entry_price, stop_price)

        triggered = current_price <= stop_price

        return TrailingStopResult(
            triggered=triggered,
            stop_price=stop_price,
            current_price=current_price,
            trailing_high=trailing_high,
            trailing_low=0.0,
            direction="long",
        )

    def check_short(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        lowest_since_entry: float,
        atr_value: Optional[float] = None,
    ) -> TrailingStopResult:
        """
        检查空头追踪止损。

        Args:
            symbol: 品种代码
            entry_price: 入场价
            current_price: 当前价
            lowest_since_entry: 入场以来最低价
            atr_value: ATR值

        Returns:
            TrailingStopResult 止损检查结果
        """
        trail_dist = self._compute_trail_distance(atr_value)

        if self.mode == "pct":
            stop_price = lowest_since_entry * (1 + trail_dist)
        else:
            stop_price = lowest_since_entry + trail_dist

        # 止损价只能下移不能上移
        prev_state = self._state.get(symbol)
        if prev_state is not None:
            prev_stop = prev_state[2]
            stop_price = min(stop_price, prev_stop)

        trailing_low = min(lowest_since_entry, prev_state[1] if prev_state else entry_price)
        self._state[symbol] = (prev_state[0] if prev_state else entry_price, trailing_low, stop_price)

        triggered = current_price >= stop_price

        return TrailingStopResult(
            triggered=triggered,
            stop_price=stop_price,
            current_price=current_price,
            trailing_high=0.0,
            trailing_low=trailing_low,
            direction="short",
        )

    def clear(self, symbol: str) -> None:
        """清除品种的追踪状态。"""
        self._state.pop(symbol, None)

    def reset(self) -> None:
        """重置所有状态。"""
        self._state.clear()
