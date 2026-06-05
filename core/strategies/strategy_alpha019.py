"""
Alpha#019 因子策略（Alpha101因子3/4）。

定义：(-1 * sign(((close - delay(close,7)) + delta(close,7)))) * (1 + rank((1 + sum(returns,250))))
简化：短期反转信号 * 长期动量横截面排名

短期反转：7日价格变化方向的反转信号
长期动量：过去250日累计收益的品种内排名

因子计算委托 core/factors/basic_factors.py:compute_alpha019()
"""
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import pybroker
    from pybroker import ExecContext
else:
    try:
        import pybroker
        from pybroker import ExecContext
    except ImportError:
        pybroker = None
        ExecContext = None

from .base import BaseStrategy
from core.factors import compute_alpha019


class Alpha019Strategy(BaseStrategy):
    """
    Alpha#019 因子策略。

    短期反转：7日价格变化的反转信号
    长期动量：250日累计收益的品种内排名

    Attributes:
        short_window: 短期窗口（默认7天）
        long_window: 长期窗口（默认250天）
        position_size: 目标仓位比例
    """

    def __init__(
        self,
        short_window: int = 7,
        long_window: int = 250,
        position_size: float = 0.2,
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        self._factor_name = f"alpha019_{short_window}_{long_window}"

    def register_indicators(self) -> List:
        """注册Alpha#019因子指标（委托 core.factors.compute_alpha019）。"""
        short_window = self.short_window
        long_window = self.long_window

        def calc_alpha019(data):
            close_series = pd.Series(data.close)
            return compute_alpha019(close_series, short_window, long_window).values

        alpha019_indicator = pybroker.indicator(self._factor_name, calc_alpha019)
        return [alpha019_indicator]

    def execute(self, ctx: ExecContext) -> None:
        """每bar执行：因子>0.5做多，<-0.5做空。"""
        if self._check_rollover(ctx):
            return
        if ctx.bars < self.long_window:
            return

        try:
            factor_arr = ctx.indicator(self._factor_name)
        except (ValueError, KeyError):
            return
        if factor_arr is None or len(factor_arr) == 0:
            return

        factor_val = float(factor_arr[-1])
        if np.isnan(factor_val):
            return

        current_price = float(ctx.close[-1])
        self._init_position_session(ctx)

        long_pos = ctx.long_pos()
        short_pos = ctx.short_pos()

        if factor_val > 0.5:
            if short_pos:
                ctx.buy_shares = short_pos.shares
            elif not long_pos:
                ctx.buy_shares = ctx.calc_target_shares(self.position_size, current_price)
        elif factor_val < -0.5:
            if long_pos:
                ctx.sell_shares = long_pos.shares
            elif not short_pos:
                ctx.sell_shares = ctx.calc_target_shares(self.position_size, current_price)
        else:
            if long_pos:
                ctx.sell_shares = long_pos.shares
            if short_pos:
                ctx.buy_shares = short_pos.shares
