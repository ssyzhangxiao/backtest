"""
Alpha#032 因子策略（Alpha101因子4/4）。

定义：scale(((sum(close,7)/7)-close)) + (20*scale(correlation(vwap, delay(close,5),230)))
参数：相关性窗口 230天，均线窗口固定7天

第一部分：价格偏离7日均线的程度
第二部分：VWAP与5日前收盘价的230天滚动相关性

因子计算委托 core/factors/basic_factors.py:compute_alpha032()
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
from core.factors import compute_alpha032


class Alpha032Strategy(BaseStrategy):
    """
    Alpha#032 因子策略。

    价格偏离7日均线 + 20倍的VWAP-收盘价相关性。

    Attributes:
        ma_window: 均线窗口（固定7天）
        corr_window: 相关性窗口（默认230天）
        position_size: 目标仓位比例
    """

    def __init__(
        self,
        ma_window: int = 7,
        corr_window: int = 230,
        position_size: float = 0.2,
    ):
        self.ma_window = ma_window
        self.corr_window = corr_window
        self.position_size = position_size
        self._factor_name = f"alpha032_{ma_window}_{corr_window}"

    def register_indicators(self) -> List:
        """注册Alpha#032因子指标（委托 core.factors.compute_alpha032）。"""
        ma_window = self.ma_window
        corr_window = self.corr_window

        def calc_alpha032(data):
            close_series = pd.Series(data.close)
            high_series = pd.Series(data.high)
            low_series = pd.Series(data.low)
            volume_series = pd.Series(data.volume)
            return compute_alpha032(close_series, high_series, low_series, volume_series, ma_window, corr_window).values

        alpha032_indicator = pybroker.indicator(self._factor_name, calc_alpha032)
        return [alpha032_indicator]

    def execute(self, ctx: ExecContext) -> None:
        """每bar执行：因子>0.5做多，<-0.5做空。"""
        if self._check_rollover(ctx):
            return
        if ctx.bars < self.corr_window:
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
