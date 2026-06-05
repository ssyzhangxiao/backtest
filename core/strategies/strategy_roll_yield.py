"""
期限结构/展期收益率策略（CTA因子2/4）。

核心逻辑：计算价格偏离长期均值的程度，作为期限结构的代理变量。
价格显著高于长期均值（升水/Contango）做空，显著低于（贴水/Backwardation）做多。

公式：roll_yield_signal = (close - SMA(close, lookback)) / SMA(close, lookback) * 100

因子计算委托 core/factors/basic_factors.py:compute_roll_yield()
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
from core.factors import compute_roll_yield


class RollYieldStrategy(BaseStrategy):
    """
    期限结构/展期收益率策略。

    计算价格偏离长期均值的程度作为展期收益率的代理变量。
    升水（价格高于均值）做空，贴水（价格低于均值）做多。

    Attributes:
        lookback: 长期均线周期（默认20天）
        entry_threshold: 开仓阈值（偏离百分比，默认2%）
        exit_threshold: 平仓阈值（回归均值，默认0.5%）
        position_size: 目标仓位比例
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        position_size: float = 0.2,
    ):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size
        self._ma_name = f"roll_yield_ma_{lookback}"

    def register_indicators(self) -> List:
        """注册展期收益指标：价格偏离长期均值的百分比（委托 core.factors.compute_roll_yield）。"""
        lookback = self.lookback

        def calc_factor(data):
            close_series = pd.Series(data.close)
            return compute_roll_yield(close_series, lookback).values

        factor_indicator = pybroker.indicator(self._ma_name, calc_factor)
        return [factor_indicator]

    def execute(self, ctx: ExecContext) -> None:
        """每bar执行：升水做空，贴水做多，回归均值平仓。"""
        if self._check_rollover(ctx):
            return
        if ctx.bars < self.lookback:
            return

        try:
            factor_arr = ctx.indicator(self._ma_name)
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

        if long_pos:
            if factor_val > -self.exit_threshold:
                ctx.sell_shares = long_pos.shares
                return
        if short_pos:
            if factor_val < self.exit_threshold:
                ctx.buy_shares = short_pos.shares
                return

        if not long_pos and not short_pos:
            if factor_val > self.entry_threshold:
                ctx.sell_shares = ctx.calc_target_shares(self.position_size, current_price)
            elif factor_val < -self.entry_threshold:
                ctx.buy_shares = ctx.calc_target_shares(self.position_size, current_price)
