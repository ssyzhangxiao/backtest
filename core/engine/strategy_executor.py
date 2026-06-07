"""
策略执行器 — 仅保留 RiskManagerAdapter（PyBroker 风控适配层兼容层）。

⚠️ P0/P1/P2 整改（2026-06-07）：
  - StrategyExecutorFactory 已完全删除（由 PyBrokerExecutorBuilder 蓝图模式替代）
  - RiskManagerAdapter 简化为薄壳，所有风控决策委托给 RiskController
  - **P0-3 整改**：止损检查统一通过 RiskController.check_composite_stop
    （RiskManagerAdapter 内部不再保留固定止损 / ATR 动态止损的具体实现）
  - 新代码请直接使用 core.risk_controller.RiskController

蓝图执行器路径: core/engine/pybroker_executor.py
蓝图替代旧执行器，逻辑更清晰且无重复造轮子。

位置: core/engine/strategy_executor.py
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from core.risk_controller import RiskController, RiskConfig
from core.risk.composite_stop import CompositeStopResult

logger = logging.getLogger(__name__)

try:
    from pybroker import ExecContext
    PYBROKER_AVAILABLE = True
except ImportError:
    PYBROKER_AVAILABLE = False
    ExecContext = Any  # type: ignore


def _get_indicator(ctx: Any, name: str) -> Optional[float]:
    """从 PyBroker ctx 获取指标值（辅助函数）。"""
    try:
        val = ctx.indicator(name)
        if val is not None and hasattr(val, '__len__') and len(val) > 0:
            return val[-1]
        if val is not None:
            return val
        return None
    except (ValueError, KeyError):
        return None
    except Exception as e:
        logger.debug("获取指标 %s 异常: %s", name, e)
        return None


def _get_close(ctx: Any) -> Optional[float]:
    """安全获取当前收盘价。"""
    try:
        close = ctx.close
        if hasattr(close, "__getitem__") and len(close) > 0:
            return float(close[-1])
        return float(close) if close is not None else None
    except Exception:
        return None


class RiskManagerAdapter:
    """
    风控适配器（DEPRECATED 兼容层，P0-3 整改后）：

    - 仅作为 PyBroker ExecContext 与 RiskController 之间的桥接
    - 所有真实风控决策（包括固定/ATR/时间/追踪止损）已迁移到 RiskController
    - **P0-3 整改**：本类内部不再保留固定止损 / ATR 动态止损的具体实现，
      统一通过 RiskController.check_composite_stop 委托给 CompositeStopManager

    ⚠️ DEPRECATED：新代码请直接使用 RiskController，
       PyBroker 侧推荐用 core/engine/pybroker_executor.PyBrokerExecutorBuilder。
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.05,
        max_position_pct: float = 0.2,
        max_total_position_pct: float = 0.4,
        daily_loss_limit: float = 0.03,
        use_atr_stop: bool = False,
        atr_multiplier: float = 2.0,
        trailing_mode: str = "pct",
        trailing_pct: float = 0.03,
        trailing_atr_mult: float = 2.0,
        max_holding_days: int = 10,
        time_target_return: float = 0.01,
        stop_loss_verbose: bool = False,
    ):
        """
        初始化风控适配器。

        P0-3 整改：所有止损相关参数（固定/ATR/追踪/时间）均委托给 RiskController
        内部的 CompositeStopManager，本类仅做参数透传。
        """
        self.stop_loss_pct = stop_loss_pct
        self.max_position_pct = max_position_pct
        self.max_total_position_pct = max_total_position_pct
        self.daily_loss_limit = daily_loss_limit
        # P0-3 整改：use_atr_stop / atr_multiplier 仅保留为兼容字段（实际不再使用）
        self.use_atr_stop = use_atr_stop
        self.atr_multiplier = atr_multiplier

        # P0-3 整改：构造 RiskController 时把所有止损参数透传
        # RiskController 内部会创建 CompositeStopManager
        self._controller = RiskController(RiskConfig(
            stop_loss_pct=stop_loss_pct,
            max_position_pct=max_position_pct,
            max_total_position_pct=max_total_position_pct,
            use_composite_stop=True,
            fixed_stop_pct=stop_loss_pct,
            trailing_mode=trailing_mode,
            trailing_pct=trailing_pct,
            trailing_atr_mult=trailing_atr_mult,
            max_holding_days=max_holding_days,
            time_target_return=time_target_return,
            stop_loss_verbose=stop_loss_verbose,
        ))

    @property
    def controller(self) -> RiskController:
        return self._controller

    # ── 委托方法：所有真实风控逻辑由 RiskController / CompositeStopManager 提供 ──
    def check_stop_loss(self, ctx: ExecContext) -> bool:
        """
        检查止损（PyBroker ExecContext 兼容接口）。

        P0-3 整改（2026-06-07）：
          不再在本方法中实现固定止损/ATR 动态止损的具体逻辑，
          统一委托给 RiskController.check_composite_stop，
          由 CompositeStopManager 统一处理固定/追踪/时间止损。

        委托逻辑：遍历多/空头持仓，调用 check_composite_stop，
        任一品种触发止损时调整 ctx.buy/sell_shares 强制平仓。
        """
        current_close = _get_close(ctx)
        atr_val = _get_indicator(ctx, "atr_14")
        triggered = False

        long_pos = ctx.long_pos()
        if long_pos:
            entry_price = float(getattr(long_pos, "avg_price", 0.0)) or current_close or 0.0
            highest = float(getattr(long_pos, "highest", 0.0)) or current_close or 0.0
            entry_day = int(getattr(long_pos, "entry_day", 0))
            current_day = int(getattr(ctx, "bars", 0)) or 0
            sym = getattr(long_pos, "symbol", "long")
            result = self._controller.check_composite_stop(
                symbol=sym,
                direction="long",
                entry_price=entry_price,
                current_price=current_close or 0.0,
                highest_since_entry=highest,
                lowest_since_entry=current_close or 0.0,
                entry_day=entry_day,
                current_day=current_day,
                atr_value=atr_val,
                auto_register_entry=False,
            )
            if result.triggered:
                ctx.sell_shares = int(long_pos.shares)
                triggered = True
                _logger_composite(result, sym, "long")

        short_pos = ctx.short_pos()
        if short_pos:
            entry_price = float(getattr(short_pos, "avg_price", 0.0)) or current_close or 0.0
            lowest = float(getattr(short_pos, "lowest", 0.0)) or current_close or 0.0
            entry_day = int(getattr(short_pos, "entry_day", 0))
            current_day = int(getattr(ctx, "bars", 0)) or 0
            sym = getattr(short_pos, "symbol", "short")
            result = self._controller.check_composite_stop(
                symbol=sym,
                direction="short",
                entry_price=entry_price,
                current_price=current_close or 0.0,
                highest_since_entry=current_close or 0.0,
                lowest_since_entry=lowest,
                entry_day=entry_day,
                current_day=current_day,
                atr_value=atr_val,
                auto_register_entry=False,
            )
            if result.triggered:
                ctx.buy_shares = int(short_pos.shares)
                triggered = True
                _logger_composite(result, sym, "short")

        return triggered

    def apply_position_limit(self, ctx: ExecContext, intended_shares: int) -> int:
        """
        兼容层：单品种仓位限制（保持原有简单语义）。
        """
        if intended_shares <= 0:
            return 0
        try:
            max_shares = ctx.calc_target_shares(self.max_position_pct)
            return max(min(intended_shares, int(max_shares)), 0)
        except Exception as e:
            logger.debug("apply_position_limit 异常: %s", e)
            return intended_shares

    def is_total_position_exceeded(self, ctx: ExecContext) -> bool:
        """兼容层：判断总仓位是否超过上限。"""
        try:
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
            return positions_value / total_equity >= self.max_total_position_pct
        except Exception as e:
            logger.debug("is_total_position_exceeded 异常: %s", e)
            return False

    def check_daily_loss_limit(self, ctx: ExecContext) -> bool:
        """
        兼容层：当日亏损是否超过限制（简单日内跟踪）。

        注：完整日级风控已由 PortfolioManager.adjust() 集中处理（持仓/回撤），
        本方法仅保留单品种执行前的最后一道安全网。
        """
        try:
            current_date = str(ctx.dt)
            if not hasattr(self, "_current_date") or self._current_date != current_date:
                self._current_date = current_date  # type: ignore[attr-defined]
                self._prev_equity = float(ctx.total_equity)  # type: ignore[attr-defined]
                return False
            prev = getattr(self, "_prev_equity", None)
            if not prev or prev <= 0:
                return False
            return (prev - float(ctx.total_equity)) / prev > self.daily_loss_limit
        except Exception as e:
            logger.debug("check_daily_loss_limit 异常: %s", e)
            return False

    def wrap_with_risk_control(self, strategy_fn: Callable) -> Callable:
        """兼容层：包裹策略执行函数，应用风控检查。"""
        rm = self

        def wrapped_execute(ctx: ExecContext) -> None:
            if rm.check_stop_loss(ctx):
                return

            long_before = ctx.long_pos()
            short_before = ctx.short_pos()
            had_long = long_before is not None and long_before.shares > 0
            had_short = short_before is not None and short_before.shares > 0

            strategy_fn(ctx)

            if ctx.buy_shares and ctx.buy_shares > 0:
                if had_short and short_before:
                    ctx.buy_shares = min(ctx.buy_shares, int(short_before.shares))
                else:
                    if rm.check_daily_loss_limit(ctx) or rm.is_total_position_exceeded(ctx):
                        ctx.buy_shares = 0
                    else:
                        ctx.buy_shares = rm.apply_position_limit(ctx, ctx.buy_shares)

            if ctx.sell_shares and ctx.sell_shares > 0:
                if had_long and long_before:
                    ctx.sell_shares = min(ctx.sell_shares, int(long_before.shares))
                else:
                    if rm.check_daily_loss_limit(ctx) or rm.is_total_position_exceeded(ctx):
                        ctx.sell_shares = 0
                    else:
                        ctx.sell_shares = rm.apply_position_limit(ctx, ctx.sell_shares)

        return wrapped_execute


def _logger_composite(result: CompositeStopResult, symbol: str, direction: str) -> None:
    """统一记录复合止损触发日志（info 级别）。"""
    if result.triggered:
        logger.info(
            "[%s] %s 复合止损触发: reason=%s, stop_price=%s",
            symbol, direction, result.trigger_reason, result.stop_price,
        )


def __getattr__(name: str) -> Any:
    """P0-1整改：StrategyExecutorFactory 已被 PyBrokerExecutorBuilder 替代。"""
    if name == "StrategyExecutorFactory":
        raise ImportError(
            "StrategyExecutorFactory 已被废弃（P0-1 整改），"
            "请改用 core.engine.pybroker_executor.PyBrokerExecutorBuilder。"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
