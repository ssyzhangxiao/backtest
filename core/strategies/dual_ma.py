"""
双均线趋势跟随策略模块。

当短期均线上穿长期均线且 ADX 表明趋势存在时做多；
当短期均线下穿长期均线且 ADX 表明趋势存在时做空。
"""
from typing import List, Optional

import numpy as np
import pandas as pd
import pybroker
from pybroker import ExecContext

from .base import BaseStrategy


class DualMAStrategy(BaseStrategy):
    """
    双均线趋势跟随策略。

    当短期均线上穿长期均线且 ADX 表明趋势存在时做多；
    当短期均线下穿长期均线且 ADX 表明趋势存在时做空。

    开平仓逻辑：
    - 金叉+趋势：若持空仓先平空（本bar返回，下bar再开多），
      若无反向持仓则直接开多
    - 死叉+趋势：若持多仓先平多（本bar返回，下bar再开空），
      若无反向持仓则直接开空

    新增止损功能：
    - 跟踪止损：基于持仓最高价/最低价的回撤止损
    - 时间止损：开仓后N天未达盈利目标则平仓

    环境指标依赖：
    使用 env_adx 列判断趋势强度。使用前必须通过
    pybroker.register_columns('env_adx') 注册，否则趋势判断默认返回 True。

    Attributes:
        short_ma: 短期均线周期
        long_ma: 长期均线周期
        adx_threshold: ADX 趋势阈值
        position_size: 目标仓位比例（占总权益）
        trailing_stop_pct: 跟踪止损回撤比例（0-1），None表示不使用
        time_stop_days: 时间止损天数，None表示不使用
    """

    def __init__(
        self,
        short_ma: int = 5,
        long_ma: int = 20,
        adx_threshold: float = 30.0,
        position_size: float = 0.3,
        trailing_stop_pct: Optional[float] = 0.03,
        time_stop_days: Optional[int] = 15,
    ):
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.adx_threshold = adx_threshold
        self.position_size = position_size
        self.trailing_stop_pct = trailing_stop_pct
        self.time_stop_days = time_stop_days
        self._sma_short_name = f"sma_{short_ma}"
        self._sma_long_name = f"sma_{long_ma}"

    def register_indicators(self) -> List:
        """
        注册 PyBroker 指标。

        PyBroker 指标函数接收 BarData 对象作为参数。
        BarData 包含 close, open, high, low, volume 等属性（numpy 数组）。
        需将 numpy 数组转为 pandas Series 才能使用 rolling 等方法。

        Returns:
            指标列表
        """
        short_ma = self.short_ma
        long_ma = self.long_ma

        def calc_sma_short(data):
            return pd.Series(data.close).rolling(window=short_ma, min_periods=1).mean().values

        def calc_sma_long(data):
            return pd.Series(data.close).rolling(window=long_ma, min_periods=1).mean().values

        sma_short = pybroker.indicator(self._sma_short_name, calc_sma_short)
        sma_long = pybroker.indicator(self._sma_long_name, calc_sma_long)
        return [sma_short, sma_long]

    def execute(self, ctx: ExecContext) -> None:
        """
        策略执行逻辑，每个 bar 对每个 symbol 调用一次。

        指标访问：ctx.indicator('sma_5') 返回 numpy 数组
        自定义列访问：ctx.is_dominant 返回 numpy 数组（需先 register_columns）

        展期逻辑：
        当检测到当前持仓合约不再是主力合约时，
        平掉旧合约仓位。新主力合约的开仓由其自身的信号触发。

        开平仓规则：
        - 金叉+趋势：先平空仓（若有），本bar不开多，下bar再开
        - 死叉+趋势：先平多仓（若有），本bar不开空，下bar再开
        - 跟踪止损：基于持仓最高价/最低价的回撤止损
        - 时间止损：开仓后N天未达盈利目标则平仓
        """
        if self._check_rollover(ctx):
            return

        if ctx.bars < self.long_ma:
            return

        try:
            sma_short_arr = ctx.indicator(self._sma_short_name)
            sma_long_arr = ctx.indicator(self._sma_long_name)
        except (ValueError, KeyError):
            return

        if sma_short_arr is None or sma_long_arr is None:
            return
        if ctx.bars < 2:
            return

        short_val = float(sma_short_arr[-1])
        long_val = float(sma_long_arr[-1])
        prev_short = float(sma_short_arr[-2])
        prev_long = float(sma_long_arr[-2])

        if np.isnan(short_val) or np.isnan(long_val):
            return
        if np.isnan(prev_short) or np.isnan(prev_long):
            return

        trend_ok = True
        try:
            adx_arr = ctx.env_adx
            if adx_arr is not None and len(adx_arr) > 0:
                adx_val = adx_arr[-1]
                adx_val = float(adx_val)
                if not np.isnan(adx_val):
                    trend_ok = adx_val > self.adx_threshold
        except (AttributeError, IndexError, TypeError, ValueError):
            pass

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
                    'trailing_low': current_price
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
                    'trailing_high': current_price,
                    'trailing_low': current_price
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

        golden_cross = prev_short <= prev_long and short_val > long_val
        death_cross = prev_short >= prev_long and short_val < long_val

        if golden_cross and trend_ok and not long_pos:
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
                'trailing_low': current_price
            }
        elif death_cross and trend_ok and not short_pos:
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
                'trailing_high': current_price,
                'trailing_low': current_price
            }