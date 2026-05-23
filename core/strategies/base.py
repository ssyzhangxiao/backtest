"""
策略抽象基类模块。

包含 BaseStrategy 抽象基类，提供公共展期逻辑和各策略必须实现的接口。
"""
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from pybroker import ExecContext


class BaseStrategy(ABC):
    """
    策略抽象基类。

    提供公共展期逻辑，各策略继承此类并在 execute 开头调用
    _check_rollover 方法，若返回 True 则直接返回（已执行展期平仓）。

    展期逻辑：
    当检测到当前持仓合约不再是主力合约时，
    平掉旧合约仓位。新主力合约的开仓由其自身的信号触发。

    子类必须实现 execute 方法。
    """

    @staticmethod
    def _check_rollover(ctx: ExecContext) -> bool:
        """
        检查并执行展期平仓。

        当当前 symbol 不是主力合约时，平掉所有持仓并返回 True。
        调用方应在 execute 开头调用此方法，若返回 True 则直接返回。

        Args:
            ctx: PyBroker 执行上下文

        Returns:
            True 表示已执行展期平仓，调用方应直接返回；
            False 表示当前为主力合约，可继续执行策略逻辑
        """
        is_dominant = True
        try:
            val = ctx.is_dominant[-1]
            if isinstance(val, np.bool_):
                is_dominant = bool(val)
            elif isinstance(val, bool):
                is_dominant = val
            elif isinstance(val, np.generic):
                is_dominant = bool(val)
        except (AttributeError, IndexError, TypeError):
            pass

        if not is_dominant:
            long_pos = ctx.long_pos()
            if long_pos:
                ctx.sell_shares = long_pos.shares
            short_pos = ctx.short_pos()
            if short_pos:
                ctx.buy_shares = short_pos.shares
            return True

        return False

    @abstractmethod
    def execute(self, ctx: ExecContext) -> None:
        """策略执行逻辑，子类必须实现。"""
        ...

    def register_indicators(self) -> List:
        """
        注册 PyBroker 指标，子类可覆盖。

        Returns:
            指标列表，默认为空列表
        """
        return []