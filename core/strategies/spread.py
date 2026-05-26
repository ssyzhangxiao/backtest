"""
跨期套利策略模块。

同时持有近月和远月合约，利用价差变化获利。
当近月-远月价差扩大时做多近月+做空远月；
当价差缩小时反向操作。
"""
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pybroker import ExecContext
else:
    try:
        from pybroker import ExecContext
    except ImportError:
        ExecContext = None  # type: ignore

from .base import BaseStrategy


class SpreadStrategy(BaseStrategy):
    """
    跨期套利策略。

    同时持有近月和远月合约，利用价差变化获利。
    当近月-远月价差扩大时做多近月+做空远月；
    当价差缩小时反向操作。

    在 PyBroker 中，跨期策略通过 ctx.session 在不同 symbol 间
    共享状态来实现。每个 bar 对每个 symbol 分别调用 execute，
    通过 session 协调两个合约的操作。

    使用前必须通过构造函数指定 near_symbol 和 far_symbol，
    策略会在首次执行时将它们写入 ctx.session。

    Attributes:
        spread_ma_period: 价差均线周期
        spread_entry_threshold: 价差偏离均值的入场阈值（标准差倍数）
        position_size: 单腿仓位比例
        near_symbol: 近月合约代码，默认 None（为 None 时策略不执行）
        far_symbol: 远月合约代码，默认 None（为 None 时策略不执行）
    """

    def __init__(
        self,
        spread_ma_period: int = 20,
        spread_entry_threshold: float = 2.0,
        position_size: float = 0.15,
        near_symbol: Optional[str] = None,
        far_symbol: Optional[str] = None,
    ):
        self.spread_ma_period = spread_ma_period
        self.spread_entry_threshold = spread_entry_threshold
        self.position_size = position_size
        self.near_symbol = near_symbol
        self.far_symbol = far_symbol

    def execute(self, ctx: ExecContext) -> None:
        """
        跨期套利执行逻辑。

        使用 ctx.session 存储跨合约共享的价差数据。
        session 中存储：
        - 'near_symbol': 近月合约代码
        - 'far_symbol': 远月合约代码
        - 'prices': 各合约最新收盘价
        - 'spread_history': 价差历史序列

        注意：PyBroker 对每个 symbol 分别调用 execute，
        因此需要通过 session 协调两个合约的操作。
        """
        if self.near_symbol is None or self.far_symbol is None:
            return

        if "near_symbol" not in ctx.session:
            ctx.session["near_symbol"] = self.near_symbol
            ctx.session["far_symbol"] = self.far_symbol

        near_symbol = ctx.session["near_symbol"]
        far_symbol = ctx.session["far_symbol"]

        if ctx.symbol not in (near_symbol, far_symbol):
            return

        if "prices" not in ctx.session:
            ctx.session["prices"] = {}

        ctx.session["prices"][ctx.symbol] = float(ctx.close[-1])

        if (
            near_symbol not in ctx.session["prices"]
            or far_symbol not in ctx.session["prices"]
        ):
            return

        if "spread_history" not in ctx.session:
            ctx.session["spread_history"] = []

        near_price = ctx.session["prices"][near_symbol]
        far_price = ctx.session["prices"][far_symbol]
        spread = near_price - far_price

        if np.isnan(spread):
            return

        ctx.session["spread_history"].append(spread)
        if len(ctx.session["spread_history"]) > 252:
            ctx.session["spread_history"] = ctx.session["spread_history"][-252:]

        if len(ctx.session["spread_history"]) < self.spread_ma_period:
            return

        spread_series = pd.Series(ctx.session["spread_history"])
        spread_ma = spread_series.rolling(window=self.spread_ma_period, min_periods=1).mean().iloc[-1]
        spread_std = spread_series.rolling(window=self.spread_ma_period, min_periods=1).std().iloc[-1]

        if spread_std is None or np.isnan(spread_std) or spread_std == 0:
            return

        z_score = (spread - spread_ma) / spread_std

        if ctx.symbol == near_symbol:
            long_pos = ctx.long_pos()
            short_pos = ctx.short_pos()

            if z_score < -self.spread_entry_threshold and not long_pos:
                if short_pos:
                    ctx.buy_shares = short_pos.shares
                    return
                ctx.buy_shares = ctx.calc_target_shares(self.position_size)
                ctx.hold_bars = 10
            elif z_score > self.spread_entry_threshold and not short_pos:
                if long_pos:
                    ctx.sell_shares = long_pos.shares
                    return
                ctx.sell_shares = ctx.calc_target_shares(self.position_size)
                ctx.hold_bars = 10
            elif abs(z_score) < 0.5:
                if long_pos:
                    ctx.sell_shares = long_pos.shares
                if short_pos:
                    ctx.buy_shares = short_pos.shares

        elif ctx.symbol == far_symbol:
            long_pos = ctx.long_pos()
            short_pos = ctx.short_pos()

            if z_score < -self.spread_entry_threshold and not short_pos:
                if long_pos:
                    ctx.sell_shares = long_pos.shares
                    return
                ctx.sell_shares = ctx.calc_target_shares(self.position_size)
                ctx.hold_bars = 10
            elif z_score > self.spread_entry_threshold and not long_pos:
                if short_pos:
                    ctx.buy_shares = short_pos.shares
                    return
                ctx.buy_shares = ctx.calc_target_shares(self.position_size)
                ctx.hold_bars = 10
            elif abs(z_score) < 0.5:
                if long_pos:
                    ctx.sell_shares = long_pos.shares
                if short_pos:
                    ctx.buy_shares = short_pos.shares