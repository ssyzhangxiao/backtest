"""
风控模块。

在 PyBroker 策略的 execute 函数中实现风险控制逻辑，包括：
- 单笔止损
- 仓位控制（多头与空头）
- 总仓位上限
- 展期成本容忍度
- 日亏损限制

风控逻辑通过包装策略执行函数实现，在策略信号产生后、
订单提交前进行风控检查。
"""

import logging
from typing import Dict, Optional

from pybroker import ExecContext

logger = logging.getLogger(__name__)


class RiskManager:
    """
    风控管理器。

    通过包装策略执行函数，在交易前进行风控检查。
    也可以直接在策略的 execute 方法中调用风控方法。

    Attributes:
        stop_loss_pct: 单笔止损比例（如0.02表示2%止损）
        max_position_pct: 单合约最大仓位占比
        max_total_position_pct: 总仓位上限占比
        rollover_cost_tolerance: 展期成本容忍度（元）
        daily_loss_limit: 日亏损上限（占总权益比例）
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.02,
        max_position_pct: float = 0.2,
        max_total_position_pct: float = 0.4,
        rollover_cost_tolerance: float = 50.0,
        daily_loss_limit: float = 0.03,
    ):
        if stop_loss_pct <= 0:
            raise ValueError(f"stop_loss_pct 必须大于0，当前值: {stop_loss_pct}")
        if max_position_pct <= 0 or max_position_pct > 1:
            raise ValueError(
                f"max_position_pct 必须在(0, 1]范围内，当前值: {max_position_pct}"
            )
        if max_total_position_pct <= 0 or max_total_position_pct > 1:
            raise ValueError(
                f"max_total_position_pct 必须在(0, 1]范围内，当前值: {max_total_position_pct}"
            )
        if daily_loss_limit <= 0:
            raise ValueError(f"daily_loss_limit 必须大于0，当前值: {daily_loss_limit}")

        self.stop_loss_pct = stop_loss_pct
        self.max_position_pct = max_position_pct
        self.max_total_position_pct = max_total_position_pct
        self.rollover_cost_tolerance = rollover_cost_tolerance
        self.daily_loss_limit = daily_loss_limit
        self._prev_equity: Optional[float] = None
        self._current_date: Optional[str] = None

    @staticmethod
    def _compute_pnl_pct_long(pos) -> float:
        """
        计算多头持仓的盈亏百分比。

        pnl_pct = pnl / cost_basis * 100
        cost_basis = equity - pnl（因为 equity = shares * close, pnl = (close - entry) * shares）

        Args:
            pos: PyBroker Position 对象

        Returns:
            盈亏百分比（如 -2.0 表示亏损2%）
        """
        pnl = float(pos.pnl)
        cost_basis = float(pos.equity) - pnl
        if cost_basis <= 0:
            return 0.0
        return pnl / cost_basis * 100

    @staticmethod
    def _compute_pnl_pct_short(pos) -> float:
        """
        计算空头持仓的盈亏百分比。

        pnl_pct = pnl / cost_basis * 100
        cost_basis = entry_price * shares

        优先使用 pos.cost_basis（PyBroker 直接提供），
        若不可用则用 margin + pnl 反推：
          margin = close * shares（空头保证金）
          pnl = (entry - close) * shares
          => margin + pnl = entry * shares = cost_basis

        Args:
            pos: PyBroker Position 对象

        Returns:
            盈亏百分比（如 -2.0 表示亏损2%）
        """
        pnl = float(pos.pnl)
        # 优先使用 PyBroker 直接提供的 cost_basis
        if hasattr(pos, 'cost_basis') and pos.cost_basis is not None:
            try:
                cost_basis = float(pos.cost_basis)
            except (ValueError, TypeError):
                cost_basis = 0.0
        else:
            # 回退：margin + pnl = entry_price * shares
            try:
                cost_basis = float(pos.margin) + pnl
            except (ValueError, TypeError, AttributeError):
                cost_basis = 0.0
        if cost_basis <= 0:
            return 0.0
        return pnl / cost_basis * 100

    def check_stop_loss(self, ctx: ExecContext) -> bool:
        """
        检查是否触发止损。

        当持仓浮亏超过止损比例时，平掉该仓位。

        Args:
            ctx: PyBroker 执行上下文

        Returns:
            是否触发了止损
        """
        triggered = False
        stop_threshold = -self.stop_loss_pct * 100

        long_pos = ctx.long_pos()
        if long_pos:
            pnl_pct = self._compute_pnl_pct_long(long_pos)
            if pnl_pct < stop_threshold:
                ctx.sell_shares = int(long_pos.shares)
                logger.debug(
                    "多头止损触发: %s pnl_pct=%.2f%% < %.2f%%",
                    ctx.symbol,
                    pnl_pct,
                    stop_threshold,
                )
                triggered = True

        short_pos = ctx.short_pos()
        if short_pos:
            pnl_pct = self._compute_pnl_pct_short(short_pos)
            if pnl_pct < stop_threshold:
                ctx.buy_shares = int(short_pos.shares)
                logger.debug(
                    "空头止损触发: %s pnl_pct=%.2f%% < %.2f%%",
                    ctx.symbol,
                    pnl_pct,
                    stop_threshold,
                )
                triggered = True

        return triggered

    def check_position_limit(self, ctx: ExecContext, intended_shares: int) -> int:
        """
        检查单合约仓位限制，返回实际可下单数量。

        Args:
            ctx: PyBroker 执行上下文
            intended_shares: 意向下单数量

        Returns:
            实际可下单数量
        """
        if intended_shares <= 0:
            return 0

        max_shares = ctx.calc_target_shares(self.max_position_pct)
        actual = min(intended_shares, int(max_shares))
        return max(actual, 0)

    def check_total_position_limit(self, ctx: ExecContext) -> bool:
        """
        检查总仓位是否超过上限。

        Args:
            ctx: PyBroker 执行上下文

        Returns:
            是否超过总仓位上限（True表示超限，不应开新仓）
        """
        total_equity = float(ctx.total_equity)
        if total_equity <= 0:
            return True

        positions_value = 0.0
        long_pos = ctx.long_pos()
        if long_pos:
            positions_value += float(long_pos.market_value)

        short_pos = ctx.short_pos()
        if short_pos:
            positions_value += abs(float(short_pos.market_value))

        position_pct = positions_value / total_equity
        return position_pct >= self.max_total_position_pct

    def check_daily_loss_limit(self, ctx: ExecContext) -> bool:
        """
        检查当日亏损是否超过限制。

        通过比较当日开盘权益与当前权益计算日亏损。
        首个交易日不检查。

        Args:
            ctx: PyBroker 执行上下文

        Returns:
            是否超过日亏损限制（True表示超限，不应开新仓）
        """
        current_date = str(ctx.dt)
        if self._current_date != current_date:
            self._current_date = current_date
            self._prev_equity = float(ctx.total_equity)
            return False

        if self._prev_equity is None or self._prev_equity <= 0:
            return False

        current_equity = float(ctx.total_equity)
        daily_loss_pct = (self._prev_equity - current_equity) / self._prev_equity
        if daily_loss_pct > self.daily_loss_limit:
            logger.warning(
                "日亏损超限: %s 亏损=%.2f%% > 限制=%.2f%%",
                ctx.symbol,
                daily_loss_pct * 100,
                self.daily_loss_limit * 100,
            )
            return True

        return False

    def check_rollover_cost(self, spread_cost: float) -> bool:
        """
        检查展期成本是否在容忍范围内。

        Args:
            spread_cost: 展期价差成本

        Returns:
            是否可接受（True表示成本可接受）
        """
        return spread_cost <= self.rollover_cost_tolerance

    def wrap_with_risk_control(self, strategy_fn):
        """
        创建带有风控检查的策略执行函数。

        包装流程：
        1. 先执行止损检查（始终允许止损平仓）
        2. 检查日亏损限制（超限则跳过策略逻辑）
        3. 检查总仓位上限（超限则跳过策略逻辑，但允许减仓）
        4. 执行策略逻辑
        5. 策略产生信号后，检查仓位限制

        Args:
            strategy_fn: 原始策略执行函数

        Returns:
            包装后的执行函数
        """
        rm = self

        def wrapped_execute(ctx: ExecContext):
            if rm.check_stop_loss(ctx):
                return

            long_before = ctx.long_pos()
            short_before = ctx.short_pos()
            had_long = long_before is not None and long_before.shares > 0
            had_short = short_before is not None and short_before.shares > 0

            strategy_fn(ctx)

            long_after = ctx.long_pos()
            short_after = ctx.short_pos()

            if ctx.buy_shares and ctx.buy_shares > 0:
                if had_short and short_after:
                    ctx.buy_shares = min(ctx.buy_shares, int(short_after.shares))
                elif had_short:
                    ctx.buy_shares = min(ctx.buy_shares, int(short_before.shares))
                else:
                    over_daily = rm.check_daily_loss_limit(ctx)
                    over_total = rm.check_total_position_limit(ctx)
                    if over_daily or over_total:
                        ctx.buy_shares = 0
                    else:
                        ctx.buy_shares = rm.check_position_limit(ctx, ctx.buy_shares)

            if ctx.sell_shares and ctx.sell_shares > 0:
                if had_long and long_after:
                    ctx.sell_shares = min(ctx.sell_shares, int(long_after.shares))
                elif had_long:
                    ctx.sell_shares = min(ctx.sell_shares, int(long_before.shares))
                else:
                    over_daily = rm.check_daily_loss_limit(ctx)
                    over_total = rm.check_total_position_limit(ctx)
                    if over_daily or over_total:
                        ctx.sell_shares = 0
                    else:
                        ctx.sell_shares = rm.check_position_limit(ctx, ctx.sell_shares)

        return wrapped_execute

    def get_risk_config(self) -> Dict:
        """
        获取风控配置摘要。

        Returns:
            风控配置字典
        """
        return {
            "stop_loss_pct": self.stop_loss_pct,
            "max_position_pct": self.max_position_pct,
            "max_total_position_pct": self.max_total_position_pct,
            "rollover_cost_tolerance": self.rollover_cost_tolerance,
            "daily_loss_limit": self.daily_loss_limit,
        }
