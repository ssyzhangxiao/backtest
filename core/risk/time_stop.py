"""
时间止损模块。

持仓N个交易日后，若收益率未达目标，强制平仓。
参数N可配置为3~20个交易日。

规则13要求：时间止损持仓N个交易日（5~15可配置）未达目标则强制平仓。

P1整改（2026-06-07）：
  - 提供 verbose 参数控制日志级别（避免回测刷屏）
  - 默认 verbose=False，触发时间止损时仅在 debug 级别输出
  - 当 verbose=True 时，触发时输出 info 级别日志
"""

from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimeStopResult:
    """时间止损检查结果。"""

    triggered: bool = False
    holding_days: int = 0
    max_holding_days: int = 0
    current_return: float = 0.0
    target_return: float = 0.0


class TimeStop:
    """
    时间止损管理器。

    持仓超过指定天数后，若收益率未达目标，强制平仓。

    用法:
        ts = TimeStop(max_holding_days=10, target_return=0.01, verbose=False)
        result = ts.check(entry_day=100, current_day=110, entry_price=100, current_price=100.5)
    """

    def __init__(
        self,
        max_holding_days: int = 10,
        target_return: float = 0.01,
        verbose: bool = False,
    ):
        """
        初始化时间止损。

        Args:
            max_holding_days: 最大持仓天数（3~20）
            target_return: 目标收益率（正数），未达此收益则触发
            verbose: 是否输出详细日志。
                - False（默认）：仅 debug 级别，避免回测刷屏
                - True：触发时输出 info 级别
        """
        self.max_holding_days = max(3, min(20, max_holding_days))
        self.target_return = target_return
        self.verbose = verbose

    def check(
        self,
        entry_day: int,
        current_day: int,
        entry_price: float,
        current_price: float,
        direction: str = "long",
        symbol: Optional[str] = None,
    ) -> TimeStopResult:
        """
        检查时间止损。

        Args:
            entry_day: 入场日索引
            current_day: 当前日索引
            entry_price: 入场价
            current_price: 当前价
            direction: 持仓方向 "long" 或 "short"
            symbol: 品种代码（用于日志，可选）

        Returns:
            TimeStopResult 止损检查结果
        """
        holding_days = current_day - entry_day

        if holding_days < self.max_holding_days:
            return TimeStopResult(
                triggered=False,
                holding_days=holding_days,
                max_holding_days=self.max_holding_days,
                current_return=0.0,
                target_return=self.target_return,
            )

        # 计算当前收益率
        if entry_price <= 0:
            current_return = 0.0
        elif direction == "long":
            current_return = (current_price - entry_price) / entry_price
        else:
            current_return = (entry_price - current_price) / entry_price

        # 持仓超时且未达目标
        triggered = current_return < self.target_return

        if triggered:
            sym_prefix = f"[{symbol}] " if symbol else ""
            msg = (
                f"{sym_prefix}时间止损触发：持仓{holding_days}天>={self.max_holding_days}天，"
                f"收益{current_return:.2%}<{self.target_return:.2%}"
            )
            if self.verbose:
                logger.info(msg)
            else:
                logger.debug(msg)

        return TimeStopResult(
            triggered=triggered,
            holding_days=holding_days,
            max_holding_days=self.max_holding_days,
            current_return=current_return,
            target_return=self.target_return,
        )
