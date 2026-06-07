"""
复合止损管理器。

整合追踪止损、时间止损和固定止损，按优先级执行。

规则13要求：
  - 优先级：固定止损 > 追踪止损 > 时间止损
  - 任一触发即执行平仓
  - 止损触发后记录日志

P0/P1整改（2026-06-07）：
  - **多空状态分离**：_fixed_stops 和 trailing_stop 状态都按 (symbol, direction) 存储
  - **统一 stop_price 语义**：时间止损返回 np.nan，固定/追踪止损返回实际价格
  - **可配置 verbose 日志开关**：默认 False，避免回测刷屏
  - **多空状态键** 修复为 Dict[Tuple[str, str], ...] 形式
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np

from .trailing_stop import TrailingStop, TrailingStopResult, StopDirection
from .time_stop import TimeStop, TimeStopResult

logger = logging.getLogger(__name__)


@dataclass
class CompositeStopResult:
    """
    复合止损检查结果。

    stop_price 语义统一：
      - 固定止损触发：固定止损价（>0）
      - 追踪止损触发：追踪止损价（>0）
      - 时间止损触发：np.nan（语义：按时平仓，无固定止损价）
      - 未触发：np.nan
    """

    triggered: bool = False
    trigger_reason: str = ""
    trailing_result: Optional[TrailingStopResult] = None
    time_result: Optional[TimeStopResult] = None
    fixed_stop_triggered: bool = False
    stop_price: float = field(default_factory=lambda: np.nan)
    direction: str = "long"


class CompositeStopManager:
    """
    复合止损管理器。

    整合固定止损、追踪止损和时间止损。
    优先级：固定止损 > 追踪止损 > 时间止损。

    P0整改（2026-06-07）：
      - 多空状态完全分离：内部 _fixed_stops 和 trailing_stop 状态都按方向存储
      - 同一品种多空交替时状态不会相互覆盖
      - 状态键统一为 (symbol, direction)

    用法:
        manager = CompositeStopManager(fixed_stop_pct=0.05)
        manager.set_entry("rb2401", 3800, direction="long")
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
        verbose: bool = False,
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
            verbose: 是否输出详细日志（默认 False，避免回测刷屏）
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
            verbose=verbose,
        )
        self.verbose = verbose

        # P0整改：固定止损价按方向分离存储
        # 键: (symbol, direction), 值: 固定止损价
        self._fixed_stops: Dict[Tuple[str, str], float] = {}

    def set_entry(
        self,
        symbol: str,
        entry_price: float,
        direction: str = "long",
    ) -> None:
        """
        设置入场价，计算固定止损价。

        P0整改：按方向存储固定止损价，多空独立。

        Args:
            symbol: 品种代码
            entry_price: 入场价
            direction: 持仓方向 "long" 或 "short"
        """
        if direction == "long":
            stop_price = entry_price * (1 - self.fixed_stop_pct)
        else:
            stop_price = entry_price * (1 + self.fixed_stop_pct)

        key = (symbol, direction)
        self._fixed_stops[key] = stop_price
        # 清除对应方向的追踪状态（重新开始追踪）
        self.trailing_stop.clear_direction(symbol, StopDirection(direction))

    def _get_fixed_stop(self, symbol: str, direction: str) -> float:
        """获取固定止损价（无记录时按当前价即时计算）。"""
        key = (symbol, direction)
        if key in self._fixed_stops:
            return self._fixed_stops[key]
        return np.nan

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
        direction = "long"

        # 1. 固定止损
        fixed_stop = self._get_fixed_stop(symbol, direction)
        if not np.isnan(fixed_stop) and current_price <= fixed_stop:
            if self.verbose:
                logger.info(
                    "[%s] 固定止损触发: %.2f <= %.2f",
                    symbol, current_price, fixed_stop,
                )
            return CompositeStopResult(
                triggered=True,
                trigger_reason="fixed_stop",
                fixed_stop_triggered=True,
                stop_price=fixed_stop,
                direction=direction,
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
            if self.verbose:
                logger.info(
                    "[%s] 追踪止损触发: %.2f <= %.2f",
                    symbol, current_price, trailing_result.stop_price,
                )
            return CompositeStopResult(
                triggered=True,
                trigger_reason="trailing_stop",
                trailing_result=trailing_result,
                stop_price=trailing_result.stop_price,
                direction=direction,
            )

        # 3. 时间止损（语义统一：返回 np.nan 表示无固定止损价）
        time_result = self.time_stop.check(
            entry_day=entry_day,
            current_day=current_day,
            entry_price=entry_price,
            current_price=current_price,
            direction=direction,
            symbol=symbol,
        )
        if time_result.triggered:
            return CompositeStopResult(
                triggered=True,
                trigger_reason="time_stop",
                time_result=time_result,
                stop_price=np.nan,
                direction=direction,
            )

        return CompositeStopResult(
            triggered=False,
            trailing_result=trailing_result,
            time_result=time_result,
            stop_price=trailing_result.stop_price,
            direction=direction,
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

        优先级：固定止损 > 追踪止损 > 时间止损

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
        direction = "short"

        # 1. 固定止损
        fixed_stop = self._get_fixed_stop(symbol, direction)
        if not np.isnan(fixed_stop) and current_price >= fixed_stop:
            if self.verbose:
                logger.info(
                    "[%s] 固定止损触发: %.2f >= %.2f",
                    symbol, current_price, fixed_stop,
                )
            return CompositeStopResult(
                triggered=True,
                trigger_reason="fixed_stop",
                fixed_stop_triggered=True,
                stop_price=fixed_stop,
                direction=direction,
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
            if self.verbose:
                logger.info(
                    "[%s] 追踪止损触发: %.2f >= %.2f",
                    symbol, current_price, trailing_result.stop_price,
                )
            return CompositeStopResult(
                triggered=True,
                trigger_reason="trailing_stop",
                trailing_result=trailing_result,
                stop_price=trailing_result.stop_price,
                direction=direction,
            )

        # 3. 时间止损
        time_result = self.time_stop.check(
            entry_day=entry_day,
            current_day=current_day,
            entry_price=entry_price,
            current_price=current_price,
            direction=direction,
            symbol=symbol,
        )
        if time_result.triggered:
            return CompositeStopResult(
                triggered=True,
                trigger_reason="time_stop",
                time_result=time_result,
                stop_price=np.nan,
                direction=direction,
            )

        return CompositeStopResult(
            triggered=False,
            trailing_result=trailing_result,
            time_result=time_result,
            stop_price=trailing_result.stop_price,
            direction=direction,
        )

    def clear(self, symbol: str, direction: Optional[str] = None) -> None:
        """
        清除止损状态。

        P0整改：默认仅清除指定方向；如不指定 direction 则清除所有方向。

        Args:
            symbol: 品种代码
            direction: 持仓方向 "long"/"short"，None 表示清除全部
        """
        if direction is None:
            keys_to_remove = [k for k in self._fixed_stops if k[0] == symbol]
            for k in keys_to_remove:
                self._fixed_stops.pop(k, None)
            self.trailing_stop.clear(symbol)
        else:
            self._fixed_stops.pop((symbol, direction), None)
            self.trailing_stop.clear_direction(symbol, StopDirection(direction))

    def reset(self) -> None:
        """重置所有状态。"""
        self._fixed_stops.clear()
        self.trailing_stop.reset()

    def get_state_snapshot(self) -> Dict[str, Dict[str, float]]:
        """
        导出当前所有固定止损价（用于调试/监控）。

        Returns:
            {symbol: {"long": long_stop, "short": short_stop}}
        """
        snapshot: Dict[str, Dict[str, float]] = {}
        for (sym, direction), stop in self._fixed_stops.items():
            snapshot.setdefault(sym, {})[direction] = stop
        return snapshot
