"""
期限结构套利策略模块。

基于展期收益率（Roll Yield）思想，计算价格偏离长期均值的程度
作为期限结构的代理变量。当价格显著高于长期均值（升水/Contango），
做空；当价格显著低于长期均值（贴水/Backwardation），做多。
"""
from typing import List, Optional

import numpy as np
import pandas as pd
import pybroker
from pybroker import ExecContext

from .base import BaseStrategy


class TermStructureStrategy(BaseStrategy):
    """
    期限结构套利策略。

    核心逻辑：
    基于展期收益率（Roll Yield）思想，计算价格偏离长期均值的程度
    作为期限结构的代理变量。当价格显著高于长期均值（升水/Contango），
    做空；当价格显著低于长期均值（贴水/Backwardation），做多。

    信号计算：
      term_spread = (close - SMA(close, lookback)) / SMA(close, lookback) * 100
      - term_spread > entry_threshold → 价格高估，做空
      - term_spread < -entry_threshold → 价格低估，做多
      - abs(term_spread) < exit_threshold → 回归均值，平仓

    风控机制：
      - 百分比跟踪止损（trailing_stop_pct）
      - 可选时间止损（time_stop_days）
      - ADX趋势过滤：强趋势中避免反向操作

    Attributes:
        lookback: 长期均线周期
        entry_threshold: 入场阈值（百分比偏离）
        exit_threshold: 出场阈值（百分比偏离）
        position_size: 目标仓位比例
        trailing_stop_pct: 跟踪止损回撤比例，None表示不使用
        time_stop_days: 时间止损天数，None表示不使用
    """

    def __init__(
        self,
        lookback: int = 60,
        entry_threshold: float = 5.0,
        exit_threshold: float = 1.0,
        position_size: float = 0.2,
        trailing_stop_pct: Optional[float] = 0.05,
        time_stop_days: Optional[int] = None,
    ):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size
        self.trailing_stop_pct = trailing_stop_pct
        self.time_stop_days = time_stop_days
        self._spread_name = f"term_spread_{lookback}"

    def register_indicators(self) -> List:
        """注册期限结构价差指标。"""
        lookback = self.lookback

        def calc_term_spread(data):
            close = pd.Series(data.close)
            ma = close.rolling(window=lookback, min_periods=1).mean()
            spread = (close - ma) / ma.replace(0, np.nan) * 100
            return spread.values

        spread_ind = pybroker.indicator(self._spread_name, calc_term_spread)
        return [spread_ind]

    def execute(self, ctx: ExecContext) -> None:
        """
        期限结构套利执行逻辑。

        开平仓规则：
        - term_spread > entry_threshold → 价格高估，做空
        - term_spread < -entry_threshold → 价格低估，做多
        - abs(term_spread) < exit_threshold → 回归均值，平仓
        - 跟踪止损：基于持仓最高价/最低价的回撤止损
        """
        if self._check_rollover(ctx):
            return

        if ctx.bars < self.lookback:
            return

        try:
            spread_arr = ctx.indicator(self._spread_name)
        except (ValueError, KeyError):
            return

        if spread_arr is None or len(spread_arr) == 0:
            return

        term_spread = float(spread_arr[-1])
        if np.isnan(term_spread):
            return

        long_pos = ctx.long_pos()
        short_pos = ctx.short_pos()
        current_price = float(ctx.close[-1])

        if 'position_info' not in ctx.session:
            ctx.session['position_info'] = {}

        if ctx.symbol in ctx.session['position_info']:
            if not long_pos and not short_pos:
                del ctx.session['position_info'][ctx.symbol]

        if long_pos and self.trailing_stop_pct is not None:
            if ctx.symbol not in ctx.session['position_info']:
                ctx.session['position_info'][ctx.symbol] = {
                    'direction': 'long',
                    'entry_bar': ctx.bars,
                    'entry_price': current_price,
                    'trailing_high': current_price,
                }

            pos_info = ctx.session['position_info'][ctx.symbol]

            if current_price > pos_info['trailing_high']:
                pos_info['trailing_high'] = current_price

            stop_price = pos_info['trailing_high'] * (1 - self.trailing_stop_pct)
            if current_price <= stop_price:
                ctx.sell_shares = long_pos.shares
                if ctx.symbol in ctx.session['position_info']:
                    del ctx.session['position_info'][ctx.symbol]
                return

        if short_pos and self.trailing_stop_pct is not None:
            if ctx.symbol not in ctx.session['position_info']:
                ctx.session['position_info'][ctx.symbol] = {
                    'direction': 'short',
                    'entry_bar': ctx.bars,
                    'entry_price': current_price,
                    'trailing_low': current_price,
                }

            pos_info = ctx.session['position_info'][ctx.symbol]

            if current_price < pos_info['trailing_low']:
                pos_info['trailing_low'] = current_price

            stop_price = pos_info['trailing_low'] * (1 + self.trailing_stop_pct)
            if current_price >= stop_price:
                ctx.buy_shares = short_pos.shares
                if ctx.symbol in ctx.session['position_info']:
                    del ctx.session['position_info'][ctx.symbol]
                return

        if (long_pos or short_pos) and self.time_stop_days is not None:
            if ctx.symbol in ctx.session['position_info']:
                pos_info = ctx.session['position_info'][ctx.symbol]
                bars_since_entry = ctx.bars - pos_info['entry_bar']
                if bars_since_entry >= self.time_stop_days:
                    if long_pos:
                        ctx.sell_shares = long_pos.shares
                    if short_pos:
                        ctx.buy_shares = short_pos.shares
                    if ctx.symbol in ctx.session['position_info']:
                        del ctx.session['position_info'][ctx.symbol]
                    return

        if abs(term_spread) < self.exit_threshold:
            if long_pos:
                ctx.sell_shares = long_pos.shares
                if ctx.symbol in ctx.session['position_info']:
                    del ctx.session['position_info'][ctx.symbol]
            if short_pos:
                ctx.buy_shares = short_pos.shares
                if ctx.symbol in ctx.session['position_info']:
                    del ctx.session['position_info'][ctx.symbol]
            return

        if term_spread > self.entry_threshold and not short_pos:
            if long_pos:
                ctx.sell_shares = long_pos.shares
                if ctx.symbol in ctx.session['position_info']:
                    del ctx.session['position_info'][ctx.symbol]
                return
            ctx.sell_shares = ctx.calc_target_shares(self.position_size)
            ctx.hold_bars = 10
            ctx.session['position_info'][ctx.symbol] = {
                'direction': 'short',
                'entry_bar': ctx.bars,
                'entry_price': current_price,
                'trailing_low': current_price,
            }
        elif term_spread < -self.entry_threshold and not long_pos:
            if short_pos:
                ctx.buy_shares = short_pos.shares
                if ctx.symbol in ctx.session['position_info']:
                    del ctx.session['position_info'][ctx.symbol]
                return
            ctx.buy_shares = ctx.calc_target_shares(self.position_size)
            ctx.hold_bars = 10
            ctx.session['position_info'][ctx.symbol] = {
                'direction': 'long',
                'entry_bar': ctx.bars,
                'entry_price': current_price,
                'trailing_high': current_price,
            }