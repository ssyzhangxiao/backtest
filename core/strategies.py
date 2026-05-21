"""
PyBroker 策略实现模块。

包含三个策略类，每个策略封装参数和执行逻辑，通过 PyBroker 的
strategy.add_execution(fn, symbols) 注册到回测引擎。

策略说明：
- DualMAStrategy: 双均线趋势跟随策略
- RSIStrategy: RSI反转策略
- SpreadStrategy: 跨期套利策略

PyBroker 的执行模型：
  每个 bar 对每个 symbol 调用一次执行函数，传入 ExecContext。
  ctx 包含当前 bar 的价格数据、持仓信息、自定义列等。
  通过 ctx.buy_shares / ctx.sell_shares 下单。
  通过 ctx.session 在同一策略的不同 symbol 间共享状态。

指标访问方式：
  PyBroker 中指标通过 ctx.indicator('name') 访问，返回 numpy 数组。
  自定义数据列通过 ctx.column_name 访问（需先 register_columns）。
"""

import pybroker
from pybroker import ExecContext
import pandas as pd
import numpy as np
from typing import Dict


class DualMAStrategy:
    """
    双均线趋势跟随策略。

    当短期均线上穿长期均线且 ADX 表明趋势存在时做多；
    当短期均线下穿长期均线且 ADX 表明趋势存在时做空。

    展期处理：在 execute 中检查当前 symbol 是否为当日主力合约，
    若非主力合约则平仓，若为主力合约且信号允许则开仓。

    Attributes:
        short_ma: 短期均线周期
        long_ma: 长期均线周期
        adx_threshold: ADX 趋势阈值
        position_size: 目标仓位比例（占总权益）
    """

    def __init__(
        self,
        short_ma: int = 5,
        long_ma: int = 20,
        adx_threshold: float = 25.0,
        position_size: float = 0.3,
    ):
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.adx_threshold = adx_threshold
        self.position_size = position_size
        self._sma_short_name = f"sma_{short_ma}"
        self._sma_long_name = f"sma_{long_ma}"

    def register_indicators(self):
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
            return pd.Series(data.close).rolling(window=short_ma, min_periods=1).mean()

        def calc_sma_long(data):
            return pd.Series(data.close).rolling(window=long_ma, min_periods=1).mean()

        sma_short = pybroker.indicator(self._sma_short_name, calc_sma_short)
        sma_long = pybroker.indicator(self._sma_long_name, calc_sma_long)
        return [sma_short, sma_long]

    def execute(self, ctx: ExecContext):
        """
        策略执行逻辑，每个 bar 对每个 symbol 调用一次。

        指标访问：ctx.indicator('sma_5') 返回 numpy 数组
        自定义列访问：ctx.is_dominant 返回 numpy 数组（需先 register_columns）

        展期逻辑：
        当检测到当前持仓合约不再是主力合约时，
        平掉旧合约仓位。新主力合约的开仓由其自身的信号触发。
        """
        if ctx.bars < self.long_ma:
            return

        is_dominant = True
        try:
            is_dominant = bool(ctx.is_dominant[-1])
        except (AttributeError, IndexError, TypeError):
            pass

        if not is_dominant:
            long_pos = ctx.long_pos()
            if long_pos:
                ctx.sell_shares = long_pos.shares
            short_pos = ctx.short_pos()
            if short_pos:
                ctx.buy_shares = short_pos.shares
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

        short_val = sma_short_arr[-1]
        long_val = sma_long_arr[-1]
        prev_short = sma_short_arr[-2]
        prev_long = sma_long_arr[-2]

        if np.isnan(short_val) or np.isnan(long_val):
            return
        if np.isnan(prev_short) or np.isnan(prev_long):
            return

        trend_ok = True
        try:
            adx_arr = ctx.env_adx
            if adx_arr is not None and len(adx_arr) > 0:
                adx_val = adx_arr[-1]
                if not np.isnan(adx_val):
                    trend_ok = adx_val > self.adx_threshold
        except (AttributeError, IndexError, TypeError):
            pass

        long_pos = ctx.long_pos()
        short_pos = ctx.short_pos()

        golden_cross = prev_short <= prev_long and short_val > long_val
        death_cross = prev_short >= prev_long and short_val < long_val

        if golden_cross and trend_ok and not long_pos:
            if short_pos:
                ctx.buy_shares = short_pos.shares
            ctx.buy_shares = ctx.calc_target_shares(self.position_size)
            ctx.hold_bars = 10
        elif death_cross and trend_ok and not short_pos:
            if long_pos:
                ctx.sell_shares = long_pos.shares


class RSIStrategy:
    """
    RSI 反转策略。

    当 RSI 低于超卖阈值时做多（反弹预期）；
    当 RSI 高于超买阈值时做空（回落预期）。
    仅在震荡市（ADX < 阈值）中激活。

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

    def register_indicators(self):
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
            return rsi

        rsi_ind = pybroker.indicator(self._rsi_name, calc_rsi)
        return [rsi_ind]

    def execute(self, ctx: ExecContext):
        """
        RSI 反转策略执行逻辑。

        仅在非主力合约时执行展期平仓，主力合约上根据 RSI 信号交易。
        仅在震荡市（ADX < 阈值）中开仓。
        """
        if ctx.bars < self.rsi_period:
            return

        is_dominant = True
        try:
            is_dominant = bool(ctx.is_dominant[-1])
        except (AttributeError, IndexError, TypeError):
            pass

        if not is_dominant:
            long_pos = ctx.long_pos()
            if long_pos:
                ctx.sell_shares = long_pos.shares
            short_pos = ctx.short_pos()
            if short_pos:
                ctx.buy_shares = short_pos.shares
            return

        try:
            rsi_arr = ctx.indicator(self._rsi_name)
        except (ValueError, KeyError):
            return

        if rsi_arr is None or len(rsi_arr) == 0:
            return

        rsi = rsi_arr[-1]
        if np.isnan(rsi):
            return

        in_range_market = True
        try:
            adx_arr = ctx.env_adx
            if adx_arr is not None and len(adx_arr) > 0:
                adx_val = adx_arr[-1]
                if not np.isnan(adx_val):
                    in_range_market = adx_val < self.adx_threshold
        except (AttributeError, IndexError, TypeError):
            pass

        long_pos = ctx.long_pos()
        short_pos = ctx.short_pos()

        if rsi < self.oversold and in_range_market and not long_pos:
            if short_pos:
                ctx.buy_shares = short_pos.shares
            ctx.buy_shares = ctx.calc_target_shares(self.position_size)
            ctx.hold_bars = 5
        elif rsi > self.overbought and in_range_market and not short_pos:
            if long_pos:
                ctx.sell_shares = long_pos.shares


class SpreadStrategy:
    """
    跨期套利策略。

    同时持有近月和远月合约，利用价差变化获利。
    当近月-远月价差扩大时做多近月+做空远月；
    当价差缩小时反向操作。

    在 PyBroker 中，跨期策略通过 ctx.session 在不同 symbol 间
    共享状态来实现。每个 bar 对每个 symbol 分别调用 execute，
    通过 session 协调两个合约的操作。

    Attributes:
        spread_ma_period: 价差均线周期
        spread_entry_threshold: 价差偏离均值的入场阈值（标准差倍数）
        position_size: 单腿仓位比例
    """

    def __init__(
        self,
        spread_ma_period: int = 20,
        spread_entry_threshold: float = 2.0,
        position_size: float = 0.15,
    ):
        self.spread_ma_period = spread_ma_period
        self.spread_entry_threshold = spread_entry_threshold
        self.position_size = position_size

    def execute(self, ctx: ExecContext):
        """
        跨期套利执行逻辑。

        使用 ctx.session 存储跨合约共享的价差数据。
        session 中存储：
        - 'near_close': 近月合约收盘价序列
        - 'far_close': 远月合约收盘价序列
        - 'spread_signal': 当前信号状态

        注意：PyBroker 对每个 symbol 分别调用 execute，
        因此需要通过 session 协调两个合约的操作。
        """
        if "near_symbol" not in ctx.session:
            return

        near_symbol = ctx.session["near_symbol"]
        far_symbol = ctx.session["far_symbol"]

        if ctx.symbol not in (near_symbol, far_symbol):
            return

        if "prices" not in ctx.session:
            ctx.session["prices"] = {}

        ctx.session["prices"][ctx.symbol] = ctx.close[-1]

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
        ctx.session["spread_history"].append(spread)

        if len(ctx.session["spread_history"]) < self.spread_ma_period:
            return

        spread_series = pd.Series(ctx.session["spread_history"])
        spread_ma = spread_series.rolling(window=self.spread_ma_period).mean().iloc[-1]
        spread_std = spread_series.rolling(window=self.spread_ma_period).std().iloc[-1]

        if pd.isna(spread_std) or spread_std == 0:
            return

        z_score = (spread - spread_ma) / spread_std

        if ctx.symbol == near_symbol:
            long_pos = ctx.long_pos()
            short_pos = ctx.short_pos()

            if z_score > self.spread_entry_threshold and not short_pos:
                if long_pos:
                    ctx.sell_shares = long_pos.shares
                ctx.sell_shares = ctx.calc_target_shares(self.position_size)
            elif z_score < -self.spread_entry_threshold and not long_pos:
                if short_pos:
                    ctx.buy_shares = short_pos.shares
                ctx.buy_shares = ctx.calc_target_shares(self.position_size)
            elif abs(z_score) < 0.5:
                if long_pos:
                    ctx.sell_shares = long_pos.shares
                if short_pos:
                    ctx.buy_shares = short_pos.shares

        elif ctx.symbol == far_symbol:
            long_pos = ctx.long_pos()
            short_pos = ctx.short_pos()

            if z_score > self.spread_entry_threshold and not long_pos:
                if short_pos:
                    ctx.buy_shares = short_pos.shares
                ctx.buy_shares = ctx.calc_target_shares(self.position_size)
            elif z_score < -self.spread_entry_threshold and not short_pos:
                if long_pos:
                    ctx.sell_shares = long_pos.shares
                ctx.sell_shares = ctx.calc_target_shares(self.position_size)
            elif abs(z_score) < 0.5:
                if long_pos:
                    ctx.sell_shares = long_pos.shares
                if short_pos:
                    ctx.buy_shares = short_pos.shares


STRATEGY_REGISTRY: Dict[str, type] = {
    "dual_ma": DualMAStrategy,
    "rsi": RSIStrategy,
    "spread": SpreadStrategy,
}


def get_strategy_class(name: str) -> type:
    """
    根据名称获取策略类。

    Args:
        name: 策略名称，'dual_ma', 'rsi', 'spread'

    Returns:
        策略类

    Raises:
        ValueError: 策略名称不存在
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略: {name}，可选: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]


def create_strategy(name: str, **kwargs) -> object:
    """
    根据名称和参数创建策略实例。

    Args:
        name: 策略名称
        **kwargs: 策略参数

    Returns:
        策略实例
    """
    cls = get_strategy_class(name)
    return cls(**kwargs)
