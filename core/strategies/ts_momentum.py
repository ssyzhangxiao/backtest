"""
时间序列动量策略（CTA因子1/4）。

核心逻辑：计算过去N个交易日的累计收益率，正收益做多，负收益做空。
参数：N=20天（默认）

信号：signal = 1 if ret_20 > 0 else -1 if ret_20 < 0 else 0
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


class TSMomentumStrategy(BaseStrategy):
    """
    时间序列动量策略。

    计算过去N日累计收益率，正收益做多，负收益做空。

    Attributes:
        window: 动量窗口（默认20天）
        position_size: 目标仓位比例
    """

    def __init__(
        self,
        window: int = 20,
        position_size: float = 0.2,
    ):
        self.window = window
        self.position_size = position_size
        self._indicator_name = f"ts_momentum_{window}"

    def register_indicators(self) -> List:
        """注册动量指标：N日累计收益率。"""
        window = self.window

        def calc_momentum(data):
            close_series = pd.Series(data.close)
            return close_series.pct_change(periods=window).values

        momentum_indicator = pybroker.indicator(self._indicator_name, calc_momentum)
        return [momentum_indicator]

    def execute(self, ctx: ExecContext) -> None:
        """每bar执行：动量>0做多，<0做空。"""
        if self._check_rollover(ctx):
            return
        if ctx.bars < self.window:
            return

        try:
            mom_arr = ctx.indicator(self._indicator_name)
        except (ValueError, KeyError):
            return
        if mom_arr is None or len(mom_arr) == 0:
            return

        mom_val = float(mom_arr[-1])
        if np.isnan(mom_val):
            return

        current_price = float(ctx.close[-1])
        self._init_position_session(ctx)

        long_pos = ctx.long_pos()
        short_pos = ctx.short_pos()

        if mom_val > 0:
            if short_pos:
                ctx.buy_shares = short_pos.shares
            elif not long_pos:
                ctx.buy_shares = ctx.calc_target_shares(self.position_size, current_price)
        elif mom_val < 0:
            if long_pos:
                ctx.sell_shares = long_pos.shares
            elif not short_pos:
                ctx.sell_shares = ctx.calc_target_shares(self.position_size, current_price)