"""
策略抽象基类模块。

包含 BaseStrategy 抽象基类，提供公共展期逻辑、持仓跟踪、止损管理等通用方法，
各策略继承后可直接调用，避免重复代码。
"""
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pybroker import ExecContext
else:
    try:
        from pybroker import ExecContext
    except ImportError:
        ExecContext = None  # type: ignore


class BaseStrategy(ABC):
    """
    策略抽象基类。

    提供公共展期逻辑、持仓跟踪与止损管理。各策略继承此类后：
    - 在 execute 开头调用 _check_rollover 处理展期平仓
    - 调用 _init_position_session / _register_*_entry 管理持仓状态
    - 调用 _check_trailing_stop_long/short 实现百分比跟踪止损
    - 调用 _check_time_stop 实现时间止损

    子类必须实现 execute 方法。
    """

    # ------------------------------------------------------------------
    # 展期检查
    # ------------------------------------------------------------------

    @staticmethod
    def _check_rollover(ctx: ExecContext) -> bool:
        """
        检查并执行展期平仓。

        当当前 symbol 不是主力合约时，平掉所有持仓并返回 True。
        调用方应在 execute 开头调用此方法，若返回 True 则直接返回。

        Args:
            ctx: PyBroker 执行上下文

        Returns:
            True 表示已执行展期平仓，调用方应直接返回；
            False 表示当前为主力合约，可继续执行策略逻辑
        """
        is_dominant = True
        try:
            val = ctx.is_dominant[-1]
            if isinstance(val, np.bool_):
                is_dominant = bool(val)
            elif isinstance(val, bool):
                is_dominant = val
            elif isinstance(val, np.generic):
                is_dominant = bool(val)
        except (AttributeError, IndexError, TypeError):
            pass

        if not is_dominant:
            long_pos = ctx.long_pos()
            if long_pos:
                ctx.sell_shares = long_pos.shares
            short_pos = ctx.short_pos()
            if short_pos:
                ctx.buy_shares = short_pos.shares
            return True

        return False

    # ------------------------------------------------------------------
    # 持仓跟踪 Session 管理
    # ------------------------------------------------------------------

    @staticmethod
    def _init_position_session(ctx: ExecContext) -> None:
        """
        初始化持仓跟踪 session，清理已平仓标的的残留信息。

        应在 execute 中、止损检查之前调用。
        """
        if 'position_info' not in ctx.session:
            ctx.session['position_info'] = {}
        symbol = ctx.symbol
        if symbol in ctx.session['position_info']:
            if not ctx.long_pos() and not ctx.short_pos():
                del ctx.session['position_info'][symbol]

    @staticmethod
    def _register_long_entry(ctx: ExecContext, current_price: float) -> None:
        """记录多头开仓信息（entry_bar, entry_price, trailing_high）。"""
        ctx.session['position_info'][ctx.symbol] = {
            'direction': 'long',
            'entry_bar': ctx.bars,
            'entry_price': current_price,
            'trailing_high': current_price,
        }

    @staticmethod
    def _register_short_entry(ctx: ExecContext, current_price: float) -> None:
        """记录空头开仓信息（entry_bar, entry_price, trailing_low）。"""
        ctx.session['position_info'][ctx.symbol] = {
            'direction': 'short',
            'entry_bar': ctx.bars,
            'entry_price': current_price,
            'trailing_low': current_price,
        }

    @staticmethod
    def _clear_position(ctx: ExecContext) -> None:
        """清除当前标的的持仓跟踪信息。"""
        ctx.session['position_info'].pop(ctx.symbol, None)

    # ------------------------------------------------------------------
    # 百分比跟踪止损
    # ------------------------------------------------------------------

    @staticmethod
    def _check_trailing_stop_long(
        ctx: ExecContext, current_price: float, stop_pct: float
    ) -> bool:
        """
        多头百分比跟踪止损：更新最高价，回撤超阈值则平多。

        Args:
            ctx: 执行上下文
            current_price: 当前收盘价
            stop_pct: 回撤比例（0~1），如 0.03 表示回撤 3%

        Returns:
            True 表示已触发止损平仓，调用方应 return
        """
        if ctx.symbol not in ctx.session.get('position_info', {}):
            return False
        info = ctx.session['position_info'][ctx.symbol]
        if current_price > info.get('trailing_high', current_price):
            info['trailing_high'] = current_price
        stop_price = info['trailing_high'] * (1 - stop_pct)
        if current_price <= stop_price:
            ctx.sell_shares = ctx.long_pos().shares
            del ctx.session['position_info'][ctx.symbol]
            return True
        return False

    @staticmethod
    def _check_trailing_stop_short(
        ctx: ExecContext, current_price: float, stop_pct: float
    ) -> bool:
        """
        空头百分比跟踪止损：更新最低价，反弹超阈值则平空。

        Args:
            ctx: 执行上下文
            current_price: 当前收盘价
            stop_pct: 反弹比例（0~1），如 0.03 表示反弹 3%

        Returns:
            True 表示已触发止损平仓，调用方应 return
        """
        if ctx.symbol not in ctx.session.get('position_info', {}):
            return False
        info = ctx.session['position_info'][ctx.symbol]
        if current_price < info.get('trailing_low', current_price):
            info['trailing_low'] = current_price
        stop_price = info['trailing_low'] * (1 + stop_pct)
        if current_price >= stop_price:
            ctx.buy_shares = ctx.short_pos().shares
            del ctx.session['position_info'][ctx.symbol]
            return True
        return False

    # ------------------------------------------------------------------
    # 时间止损
    # ------------------------------------------------------------------

    @staticmethod
    def _check_time_stop(
        ctx: ExecContext, time_stop_days: int
    ) -> bool:
        """
        时间止损：持仓超过指定天数则平仓。

        Args:
            ctx: 执行上下文
            time_stop_days: 最大持仓天数

        Returns:
            True 表示已触发时间止损平仓，调用方应 return
        """
        if ctx.symbol not in ctx.session.get('position_info', {}):
            return False
        info = ctx.session['position_info'][ctx.symbol]
        if ctx.bars - info['entry_bar'] >= time_stop_days:
            long_pos = ctx.long_pos()
            short_pos = ctx.short_pos()
            if long_pos:
                ctx.sell_shares = long_pos.shares
            if short_pos:
                ctx.buy_shares = short_pos.shares
            del ctx.session['position_info'][ctx.symbol]
            return True
        return False

    # ------------------------------------------------------------------
    # 持仓量指标辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_oi_change(
        open_interest, period: int = 5
    ) -> float:
        """
        计算持仓量变化率（相对于 N 日均值）。

        期货持仓量变化领先于价格，增仓+价格上涨 = 多头主动,
        增仓+价格下跌 = 空头主动, 减仓 = 资金撤离。

        Args:
            open_interest: 持仓量序列（numpy array 或可索引对象）
            period: 均值周期

        Returns:
            变化率（-1~+∞），正值=增仓，负值=减仓，NaN时返回0
        """
        if open_interest is None or len(open_interest) < period:
            return 0.0
        try:
            oi_arr = np.asarray(open_interest[-period:], dtype=float)
            oi_ma = np.nanmean(oi_arr) if len(oi_arr) > 0 else np.nan
            oi_now = float(oi_arr[-1]) if not np.isnan(oi_arr[-1]) else np.nan
            if np.isnan(oi_now) or np.isnan(oi_ma) or oi_ma <= 0:
                return 0.0
            return (oi_now - oi_ma) / oi_ma
        except (ValueError, TypeError, IndexError):
            return 0.0

    @staticmethod
    def _compute_oi_divergence(
        price_change: float, oi_change: float
    ) -> int:
        """
        检测价格与持仓量背离方向。

        期货经典信号：
          - 量增价涨（同向多）：增仓+价格上涨 → 买盘强劲，返回 1
          - 量增价跌（同向空）：增仓+价格下跌 → 卖盘强劲，返回 -1
          - 量减价涨（背离空）：减仓+价格上涨 → 虚涨无力，返回 -1
          - 量减价跌（背离多）：减仓+价格下跌 → 虚跌见底，返回 1
          - 无显著变化：返回 0

        Args:
            price_change: 价格短期变化率（-1~+∞）
            oi_change: 持仓量短期变化率（-1~+∞）

        Returns:
            1=做多信号, -1=做空信号, 0=无信号
        """
        oi_threshold = 0.03   # 持仓量变化 > 3% 视为显著
        px_threshold = 0.005  # 价格变化 > 0.5% 视为显著

        oi_up = oi_change > oi_threshold
        oi_down = oi_change < -oi_threshold
        px_up = price_change > px_threshold
        px_down = price_change < -px_threshold

        if oi_up and px_up:
            return 1   # 量增价涨 → 多头强势
        if oi_up and px_down:
            return -1  # 量增价跌 → 空头强势
        if oi_down and px_up:
            return -1  # 量减价涨 → 虚涨，偏空
        if oi_down and px_down:
            return 1   # 量减价跌 → 虚跌，偏多

        return 0

    # ------------------------------------------------------------------
    # 抽象接口
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self, ctx: ExecContext) -> None:
        """策略执行逻辑，子类必须实现。"""
        ...

    def register_indicators(self) -> List:
        """
        注册 PyBroker 指标，子类可覆盖。

        Returns:
            指标列表，默认为空列表
        """
        return []