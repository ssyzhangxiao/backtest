"""
波动率突破策略模块。

基于 ATR 构建动态价格通道，当价格突破通道上轨时做多，
突破下轨时做空。通道随波动率自适应变化，大波动时通道变宽，
低波动时通道收窄。
"""
from typing import List, Optional

import numpy as np
import pandas as pd
import pybroker
from pybroker import ExecContext

from .base import BaseStrategy


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    波动率突破策略。

    核心逻辑：
    基于 ATR 构建动态价格通道，当价格突破通道上轨时做多，
    突破下轨时做空。通道随波动率自适应变化，大波动时通道变宽，
    低波动时通道收窄。

    通道计算：
      center = SMA(close, band_period)
      upper_band = center + atr_multiplier * ATR
      lower_band = center - atr_multiplier * ATR

    交易规则：
      - close > upper_band → 做多
      - close < lower_band → 做空
      - close 回归 center → 平仓

    风控机制：
      - ATR倍数跟踪止损（trailing_stop_atr_mult）
      - 可选时间止损（time_stop_days）
      - 持仓状态跟踪防止重复开仓

    Attributes:
        atr_period: ATR计算周期
        band_period: 均线通道周期
        atr_multiplier: ATR通道倍数
        position_size: 目标仓位比例
        trailing_stop_atr_mult: ATR跟踪止损倍数，None表示不使用
        time_stop_days: 时间止损天数，None表示不使用
    """

    def __init__(
        self,
        atr_period: int = 26,
        band_period: int = 30,
        atr_multiplier: float = 2.0,
        position_size: float = 0.2,
        trailing_stop_atr_mult: Optional[float] = 3.0,
        time_stop_days: Optional[int] = None,
    ):
        self.atr_period = atr_period
        self.band_period = band_period
        self.atr_multiplier = atr_multiplier
        self.position_size = position_size
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.time_stop_days = time_stop_days
        self._center_name = f"vb_center_{band_period}"
        self._upper_name = f"vb_upper_{band_period}_{atr_multiplier}"
        self._lower_name = f"vb_lower_{band_period}_{atr_multiplier}"
        self._stop_long_name = "vb_stop_long"
        self._stop_short_name = "vb_stop_short"

    def register_indicators(self) -> List:
        """注册波动率突破通道指标和跟踪止损指标。"""
        band_period = self.band_period
        atr_period = self.atr_period
        atr_multiplier = self.atr_multiplier
        trailing_stop_atr_mult = self.trailing_stop_atr_mult

        def calc_atr(data):
            high = pd.Series(data.high)
            low = pd.Series(data.low)
            close = pd.Series(data.close)
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=atr_period, min_periods=1).mean()
            return atr.values

        def calc_center(data):
            return pd.Series(data.close).rolling(window=band_period, min_periods=1).mean().values

        def calc_upper(data):
            close = pd.Series(data.close)
            center = close.rolling(window=band_period, min_periods=1).mean()
            atr_vals = calc_atr(data)
            return center.values + atr_multiplier * atr_vals

        def calc_lower(data):
            close = pd.Series(data.close)
            center = close.rolling(window=band_period, min_periods=1).mean()
            atr_vals = calc_atr(data)
            return center.values - atr_multiplier * atr_vals

        center_ind = pybroker.indicator(self._center_name, calc_center)
        upper_ind = pybroker.indicator(self._upper_name, calc_upper)
        lower_ind = pybroker.indicator(self._lower_name, calc_lower)

        indicators = [center_ind, upper_ind, lower_ind]

        if trailing_stop_atr_mult is not None:
            def calc_stop_long(data):
                close = pd.Series(data.close)
                atr_vals = calc_atr(data)
                return close.values - trailing_stop_atr_mult * atr_vals

            def calc_stop_short(data):
                close = pd.Series(data.close)
                atr_vals = calc_atr(data)
                return close.values + trailing_stop_atr_mult * atr_vals

            stop_long_ind = pybroker.indicator(self._stop_long_name, calc_stop_long)
            stop_short_ind = pybroker.indicator(self._stop_short_name, calc_stop_short)
            indicators.extend([stop_long_ind, stop_short_ind])

        return indicators

    def execute(self, ctx: ExecContext) -> None:
        """
        波动率突破执行逻辑。

        交易规则：
          - close > upper_band → 做多
          - close < lower_band → 做空
          - close 回归 center → 平仓
          - ATR跟踪止损
        """
        if self._check_rollover(ctx):
            return

        if ctx.bars < max(self.band_period, self.atr_period):
            return

        try:
            upper_arr = ctx.indicator(self._upper_name)
            lower_arr = ctx.indicator(self._lower_name)
            center_arr = ctx.indicator(self._center_name)
        except (ValueError, KeyError):
            return

        if upper_arr is None or lower_arr is None or center_arr is None:
            return

        upper_band = float(upper_arr[-1])
        lower_band = float(lower_arr[-1])
        center_val = float(center_arr[-1])

        if np.isnan(upper_band) or np.isnan(lower_band) or np.isnan(center_val):
            return

        current_price = float(ctx.close[-1])
        long_pos = ctx.long_pos()
        short_pos = ctx.short_pos()

        if 'position_info' not in ctx.session:
            ctx.session['position_info'] = {}

        if ctx.symbol in ctx.session['position_info']:
            if not long_pos and not short_pos:
                del ctx.session['position_info'][ctx.symbol]

        if long_pos and self.trailing_stop_atr_mult is not None:
            if ctx.symbol not in ctx.session['position_info']:
                ctx.session['position_info'][ctx.symbol] = {
                    'direction': 'long',
                    'entry_bar': ctx.bars,
                    'entry_price': current_price,
                }

            try:
                stop_long_arr = ctx.indicator(self._stop_long_name)
                if stop_long_arr is not None and len(stop_long_arr) > 0:
                    stop_price = float(stop_long_arr[-1])
                else:
                    stop_price = None
            except (ValueError, KeyError):
                stop_price = None

            if stop_price is not None and not np.isnan(stop_price):
                if current_price <= stop_price:
                    ctx.sell_shares = long_pos.shares
                    if ctx.symbol in ctx.session['position_info']:
                        del ctx.session['position_info'][ctx.symbol]
                    return

        if short_pos and self.trailing_stop_atr_mult is not None:
            if ctx.symbol not in ctx.session['position_info']:
                ctx.session['position_info'][ctx.symbol] = {
                    'direction': 'short',
                    'entry_bar': ctx.bars,
                    'entry_price': current_price,
                }

            try:
                stop_short_arr = ctx.indicator(self._stop_short_name)
                if stop_short_arr is not None and len(stop_short_arr) > 0:
                    stop_price = float(stop_short_arr[-1])
                else:
                    stop_price = None
            except (ValueError, KeyError):
                stop_price = None

            if stop_price is not None and not np.isnan(stop_price):
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

        if current_price > upper_band and not long_pos:
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
            }
        elif current_price < lower_band and not short_pos:
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
            }
        elif center_val is not None and not np.isnan(center_val):
            if long_pos and current_price < center_val:
                ctx.sell_shares = long_pos.shares
                if ctx.symbol in ctx.session['position_info']:
                    del ctx.session['position_info'][ctx.symbol]
            if short_pos and current_price > center_val:
                ctx.buy_shares = short_pos.shares
                if ctx.symbol in ctx.session['position_info']:
                    del ctx.session['position_info'][ctx.symbol]