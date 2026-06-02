"""
复合止损管理器。

整合追踪止损、时间止损和固定止损，按优先级执行。

规则13要求：
  - 优先级：固定止损 > 追踪止损 > 时间止损
  - 任一触发即执行平仓
  - 止损触发后记录日志
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np

from .trailing_stop import TrailingStop, TrailingStopResult
from .time_stop import TimeStop, TimeStopResult

logger = logging.getLogger(__name__)


@dataclass
class CompositeStopResult:
    """复合止损检查结果。"""

    triggered: bool = False
    trigger_reason: str = ""
    trailing_result: Optional[TrailingStopResult] = None
    time_result: Optional[TimeStopResult] = None
    fixed_stop_triggered: bool = False
    stop_price: float = 0.0


class CompositeStopManager:
    """
    复合止损管理器。

    整合固定止损、追踪止损和时间止损。
    优先级：固定止损 > 追踪止损 > 时间止损。

    用法:
        manager = CompositeStopManager(fixed_stop_pct=0.05)
        result = manager.check_long(
            symbol="rb2401",
            entry_price=3800,
            current_price=3700,
            highest=3900,
            entry_day=0, current_day=12,
            atr_value=50.0,
        )
        if result.triggered:
            # 执行平仓
    """

    def __init__(
        self,
        fixed_stop_pct: float = 0.05,
        trailing_mode: str = "pct",
        trailing_pct: float = 0.03,
        trailing_atr_mult: float = 2.0,
        max_holding_days: int = 10,
        time_target_return: float = 0.01,
    ):
        """
        初始化复合止损管理器。

        Args:
            fixed_stop_pct: 固定止损百分比（如0.05=5%）
            trailing_mode: 追踪止损模式 "pct" 或 "atr"
            trailing_pct: 追踪百分比
            trailing_atr_mult: 追踪ATR倍数
            max_holding_days: 时间止损最大持仓天数
            time_target_return: 时间止损目标收益率
        """
        self.fixed_stop_pct = fixed_stop_pct
        self.trailing_stop = TrailingStop(
            mode=trailing_mode,
            trail_pct=trailing_pct,
            atr_multiplier=trailing_atr_mult,
        )
        self.time_stop = TimeStop(
            max_holding_days=max_holding_days,
            target_return=time_target_return,
        )

        # 固定止损价记录：{symbol: (long_stop, short_stop)}
        self._fixed_stops: Dict[str, Tuple[float, float]] = {}

    def set_entry(self, symbol: str, entry_price: float) -> None:
        """
        设置入场价，计算固定止损价。

        Args:
            symbol: 品种代码
            entry_price: 入场价
        """
        long_stop = entry_price * (1 - self.fixed_stop_pct)
        short_stop = entry_price * (1 + self.fixed_stop_pct)
        self._fixed_stops[symbol] = (long_stop, short_stop)
        self.trailing_stop.clear(symbol)

    def check_long(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        highest_since_entry: float,
        entry_day: int,
        current_day: int,
        atr_value: Optional[float] = None,
    ) -> CompositeStopResult:
        """
        检查多头复合止损。

        优先级：固定止损 > 追踪止损 > 时间止损

        Args:
            symbol: 品种代码
            entry_price: 入场价
            current_price: 当前价
            highest_since_entry: 入场以来最高价
            entry_day: 入场日索引
            current_day: 当前日索引
            atr_value: ATR值

        Returns:
            CompositeStopResult 复合止损结果
        """
        # 1. 固定止损
        fixed_stop = self._fixed_stops.get(symbol, (entry_price * (1 - self.fixed_stop_pct), 0.0))[0]
        if current_price <= fixed_stop:
            logger.info(f"[{symbol}] 固定止损触发：{current_price:.2f}<={fixed_stop:.2f}")
            return CompositeStopResult(
                triggered=True,
                trigger_reason="fixed_stop",
                fixed_stop_triggered=True,
                stop_price=fixed_stop,
            )

        # 2. 追踪止损
        trailing_result = self.trailing_stop.check_long(
            symbol=symbol,
            entry_price=entry_price,
            current_price=current_price,
            highest_since_entry=highest_since_entry,
            atr_value=atr_value,
        )
        if trailing_result.triggered:
            logger.info(
                f"[{symbol}] 追踪止损触发：{current_price:.2f}<="
                f"{trailing_result.stop_price:.2f}"
            )
            return CompositeStopResult(
                triggered=True,
                trigger_reason="trailing_stop",
                trailing_result=trailing_result,
                stop_price=trailing_result.stop_price,
            )

        # 3. 时间止损
        time_result = self.time_stop.check(
            entry_day=entry_day,
            current_day=current_day,
            entry_price=entry_price,
            current_price=current_price,
            direction="long",
        )
        if time_result.triggered:
            logger.info(
                f"[{symbol}] 时间止损触发：持仓{time_result.holding_days}天，"
                f"收益{time_result.current_return:.2%}"
            )
            return CompositeStopResult(
                triggered=True,
                trigger_reason="time_stop",
                time_result=time_result,
                stop_price=current_price,
            )

        return CompositeStopResult(
            triggered=False,
            trailing_result=trailing_result,
            time_result=time_result,
            stop_price=trailing_result.stop_price,
        )

    def check_short(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        lowest_since_entry: float,
        entry_day: int,
        current_day: int,
        atr_value: Optional[float] = None,
    ) -> CompositeStopResult:
        """
        检查空头复合止损。

        Args:
            symbol: 品种代码
            entry_price: 入场价
            current_price: 当前价
            lowest_since_entry: 入场以来最低价
            entry_day: 入场日索引
            current_day: 当前日索引
            atr_value: ATR值

        Returns:
            CompositeStopResult 复合止损结果
        """
        # 1. 固定止损
        fixed_stop = self._fixed_stops.get(symbol, (0.0, entry_price * (1 + self.fixed_stop_pct)))[1]
        if current_price >= fixed_stop:
            logger.info(f"[{symbol}] 固定止损触发：{current_price:.2f}>={fixed_stop:.2f}")
            return CompositeStopResult(
                triggered=True,
                trigger_reason="fixed_stop",
                fixed_stop_triggered=True,
                stop_price=fixed_stop,
            )

        # 2. 追踪止损
        trailing_result = self.trailing_stop.check_short(
            symbol=symbol,
            entry_price=entry_price,
            current_price=current_price,
            lowest_since_entry=lowest_since_entry,
            atr_value=atr_value,
        )
        if trailing_result.triggered:
            logger.info(
                f"[{symbol}] 追踪止损触发：{current_price:.2f}>="
                f"{trailing_result.stop_price:.2f}"
            )
            return CompositeStopResult(
                triggered=True,
                trigger_reason="trailing_stop",
                trailing_result=trailing_result,
                stop_price=trailing_result.stop_price,
            )

        # 3. 时间止损
        time_result = self.time_stop.check(
            entry_day=entry_day,
            current_day=current_day,
            entry_price=entry_price,
            current_price=current_price,
            direction="short",
        )
        if time_result.triggered:
            logger.info(
                f"[{symbol}] 时间止损触发：持仓{time_result.holding_days}天，"
                f"收益{time_result.current_return:.2%}"
            )
            return CompositeStopResult(
                triggered=True,
                trigger_reason="time_stop",
                time_result=time_result,
                stop_price=current_price,
            )

        return CompositeStopResult(
            triggered=False,
            trailing_result=trailing_result,
            time_result=time_result,
            stop_price=trailing_result.stop_price,
        )

    def clear(self, symbol: str) -> None:
        """清除品种的所有止损状态。"""
        self._fixed_stops.pop(symbol, None)
        self.trailing_stop.clear(symbol)

    def reset(self) -> None:
        """重置所有状态。"""
        self._fixed_stops.clear()
        self.trailing_stop.reset()
