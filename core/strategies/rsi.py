"""
RSI 反转策略模块。

当 RSI 低于超卖阈值时做多（反弹预期）；
当 RSI 高于超买阈值时做空（回落预期）。
仅在震荡市（ADX < 阈值）中开仓。
"""
from typing import List

import numpy as np
import pandas as pd
import pybroker
from pybroker import ExecContext

from .base import BaseStrategy


class RSIStrategy(BaseStrategy):
    """
    RSI 反转策略。

    当 RSI 低于超卖阈值时做多（反弹预期）；
    当 RSI 高于超买阈值时做空（回落预期）。
    仅在震荡市（ADX < 阈值）中开仓。

    开平仓逻辑：
    - 超卖+震荡市：若持空仓先平空（本bar返回，下bar再开多），
      若无反向持仓则直接开多
    - 超买+震荡市：若持多仓先平多（本bar返回，下bar再开空），
      若无反向持仓则直接开空
    - 非震荡市（ADX >= 阈值）：若持有仓位则平仓

    环境指标依赖：
    使用 env_adx 列判断震荡市。使用前必须通过
    pybroker.register_columns('env_adx') 注册，否则震荡市判断默认返回 True。

    Attributes:
        rsi_period: RSI 计算周期
        oversold: 超卖阈值
        overbought: 超买阈值
        adx_threshold: ADX 震荡市阈值（ADX < 此值视为震荡市）
        position_size: 目标仓位比例
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        adx_threshold: float = 25.0,
        position_size: float = 0.2,
    ):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.adx_threshold = adx_threshold
        self.position_size = position_size
        self._rsi_name = f"rsi_{rsi_period}"

    def register_indicators(self) -> List:
        """
        注册 RSI 指标。

        PyBroker 指标函数接收 BarData 对象，通过 data.close 访问收盘价。

        Returns:
            指标列表
        """
        period = self.rsi_period

        def calc_rsi(data):
            close = pd.Series(data.close)
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi.values

        rsi_ind = pybroker.indicator(self._rsi_name, calc_rsi)
        return [rsi_ind]

    def execute(self, ctx: ExecContext) -> None:
        """
        RSI 反转策略执行逻辑。

        仅在主力合约上根据 RSI 信号交易。
        仅在震荡市（ADX < 阈值）中开仓，非震荡市中若持有仓位则平仓。

        开平仓规则：
        - 超卖+震荡市：先平空仓（若有），本bar不开多，下bar再开
        - 超买+震荡市：先平多仓（若有），本bar不开空，下bar再开
        - 非震荡市：平掉所有持仓
        """
        if self._check_rollover(ctx):
            return

        if ctx.bars < self.rsi_period:
            return

        try:
            rsi_arr = ctx.indicator(self._rsi_name)
        except (ValueError, KeyError):
            return

        if rsi_arr is None or len(rsi_arr) == 0:
            return

        rsi = rsi_arr[-1]
        rsi = float(rsi)
        if np.isnan(rsi):
            return

        in_range_market = True
        try:
            adx_arr = ctx.env_adx
            if adx_arr is not None and len(adx_arr) > 0:
                adx_val = adx_arr[-1]
                adx_val = float(adx_val)
                if not np.isnan(adx_val):
                    in_range_market = adx_val < self.adx_threshold
        except (AttributeError, IndexError, TypeError, ValueError):
            pass

        long_pos = ctx.long_pos()
        short_pos = ctx.short_pos()

        if not in_range_market:
            if long_pos:
                ctx.sell_shares = long_pos.shares
            if short_pos:
                ctx.buy_shares = short_pos.shares
            return

        if rsi < self.oversold and not long_pos:
            if short_pos:
                ctx.buy_shares = short_pos.shares
                return
            ctx.buy_shares = ctx.calc_target_shares(self.position_size)
            ctx.hold_bars = 5
        elif rsi > self.overbought and not short_pos:
            if long_pos:
                ctx.sell_shares = long_pos.shares
                return
            ctx.sell_shares = ctx.calc_target_shares(self.position_size)
            ctx.hold_bars = 5