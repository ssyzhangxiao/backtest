"""
追踪止损（Trailing Stop）。

两种模式：
  - 固定点数追踪：trail_price = highest - trail_pct * entry_price
  - ATR倍数追踪：trail_price = highest - N * ATR

规则13要求：追踪止损支持固定点数和ATR倍数两种模式。

P1整改（2026-06-07）：多空状态分离，避免同一品种多空交替时状态被覆盖。
  - 状态键使用 (symbol, direction) 组合，例如 ("rb2401", "long")
  - 内部 _state: Dict[Tuple[str, str], TrailingState]
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StopDirection(Enum):
    """止损方向枚举。"""

    LONG = "long"
    SHORT = "short"


@dataclass
class TrailingStopResult:
    """追踪止损检查结果。"""

    triggered: bool = False
    stop_price: float = 0.0
    current_price: float = 0.0
    trailing_high: float = 0.0
    trailing_low: float = 0.0
    direction: str = "long"


@dataclass
class _TrailingState:
    """单方向追踪状态。"""

    trailing_high: float = 0.0
    trailing_low: float = 0.0
    stop_price: float = 0.0
    entry_price: float = 0.0


class TrailingStop:
    """
    追踪止损管理器。

    支持固定百分比和ATR倍数两种追踪止损模式。
    多头：止损价 = max(前止损价, 最高价 - 追踪距离)
    空头：止损价 = min(前止损价, 最低价 + 追踪距离)

    P1整改：多空状态独立存储，避免同一品种多空交替时状态覆盖。
    内部使用 (symbol, direction) 作为状态键。

    用法:
        ts = TrailingStop(mode="atr", atr_multiplier=2.0)
        result = ts.check_long(
            symbol="rb2401",
            entry_price=100,
            current_price=105,
            highest=108,
            atr=3.0,
        )
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

        # P1整改：多空状态独立存储
        # 键: (symbol, direction), 值: _TrailingState
        self._state: Dict[Tuple[str, str], _TrailingState] = {}

    def _state_key(self, symbol: str, direction: StopDirection) -> Tuple[str, str]:
        """生成状态键。"""
        return (symbol, direction.value)

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

        # P1整改：使用 (symbol, direction) 键获取前状态
        key = self._state_key(symbol, StopDirection.LONG)
        prev = self._state.get(key)
        if prev is not None:
            # 止损价只能上移不能下移
            stop_price = max(stop_price, prev.stop_price)
            trailing_high = max(highest_since_entry, prev.trailing_high)
        else:
            trailing_high = max(highest_since_entry, entry_price)

        # 更新状态
        self._state[key] = _TrailingState(
            trailing_high=trailing_high,
            trailing_low=0.0,
            stop_price=stop_price,
            entry_price=entry_price,
        )

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

        # P1整改：使用 (symbol, direction) 键获取前状态
        key = self._state_key(symbol, StopDirection.SHORT)
        prev = self._state.get(key)
        if prev is not None:
            # 止损价只能下移不能上移
            stop_price = min(stop_price, prev.stop_price)
            trailing_low = min(lowest_since_entry, prev.trailing_low)
        else:
            trailing_low = min(lowest_since_entry, entry_price)

        # 更新状态
        self._state[key] = _TrailingState(
            trailing_high=0.0,
            trailing_low=trailing_low,
            stop_price=stop_price,
            entry_price=entry_price,
        )

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
        """清除品种的所有方向追踪状态。"""
        keys_to_remove = [k for k in self._state if k[0] == symbol]
        for k in keys_to_remove:
            self._state.pop(k, None)

    def clear_direction(self, symbol: str, direction: StopDirection) -> None:
        """清除品种指定方向的追踪状态。"""
        self._state.pop(self._state_key(symbol, direction), None)

    def reset(self) -> None:
        """重置所有状态。"""
        self._state.clear()

    def get_state(self, symbol: str, direction: StopDirection) -> Optional[_TrailingState]:
        """查询品种指定方向的追踪状态（调试用）。"""
        return self._state.get(self._state_key(symbol, direction))
