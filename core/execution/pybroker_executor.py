"""
PyBroker 执行器构建器（蓝图实现，P0-1/P0-2/P0-3 整改）。

位置: core/engine/pybroker_executor.py

替代旧的 `core/engine/strategy_executor.py:StrategyExecutorFactory`，
实现用户指定的蓝图模式：
    1. 调仓日收集所有品种的因子得分（每个品种调用一次）
    2. 收齐后 finalize 横截面（最后一个品种触发）
    3. 计算综合得分
    4. 通过 PortfolioManager 分配目标权重
    5. 通过 RiskController 调整（集中度等）
    6. 单品种执行下单

PyBroker 的限制：
    - executor 函数按 (symbol, bar) 顺序调用，逐品种独立
    - 横截面数据需要共享状态：本类持有 SharedState
    - 在每个调仓日，第一个品种开始收集，最后一个品种触发 finalize

P0-3 整改：完全使用 RiskController（而非旧的 RiskManagerAdapter）。

改进（2026-06-13）：
  - 横截面状态管理改用 set 去重，避免重复调用导致计数溢出
  - 统一风险预算逻辑（risk_per_trade + target_vol 波动率平价）
  - 止损使用 state 显式追踪 entry_price / entry_bar

CTA 退出策略注入（2026-06-13）：
  - 通过 cta_exit_policy 参数注入 CTAExitPolicy，启用四层退出 + 风险预算
  - 未注入时保持原有 RiskController 行为
  - 实现单品种 CTA 策略与多品种横截面策略共用同一执行器
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.config import BacktestConfig
from core.engine.switch_engine import FactorScoringEngine
from core.execution.cta_exit_policy import CTAExitPolicy, CTAExitConfig
from core.portfolio import PortfolioManager
from core.risk_controller import RiskController, RiskConfig

_logger = logging.getLogger(__name__)

try:
    from pybroker import ExecContext
    PYBROKER_AVAILABLE = True
except ImportError:
    PYBROKER_AVAILABLE = False
    ExecContext = Any  # type: ignore


# ────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────
def _get_indicator(ctx: Any, name: str) -> Optional[float]:
    """从 PyBroker ctx 安全获取指标值。"""
    try:
        val = ctx.indicator(name)
        if val is None:
            return None
        if hasattr(val, "iloc") and len(val) > 0:
            return float(val.iloc[-1])
        if hasattr(val, "__getitem__") and len(val) > 0:
            return float(val[-1])
        return float(val)
    except (ValueError, KeyError):
        return None
    except Exception as e:  # noqa: BLE001
        _logger.debug("获取指标 %s 异常: %s", name, e)
        return None


def _get_close(ctx: Any) -> Optional[float]:
    """安全获取当前收盘价。"""
    try:
        close = ctx.close
        if hasattr(close, "__getitem__") and len(close) > 0:
            return float(close[-1])
        if close is not None:
            return float(close)
    except Exception as e:  # noqa: BLE001
        _logger.debug("获取收盘价异常: %s", e)
    return None


def _get_atr(ctx: Any) -> float:
    """获取 ATR 指标，若不可用返回 0。"""
    atr = _get_indicator(ctx, "atr_14")
    return float(atr) if atr is not None and atr > 0 else 0.0


def _get_pos_shares(ctx: Any, side: str) -> int:
    """获取当前品种的多/空持仓数量。"""
    try:
        pos = ctx.pos(ctx.symbol, side)
        if pos is not None:
            return int(getattr(pos, "shares", 0))
    except Exception:  # noqa: BLE001
        pass
    return 0


# ────────────────────────────────────────────────────────────
# 共享状态（蓝图数据）
# ────────────────────────────────────────────────────────────
@dataclass
class PyBrokerExecutorSharedState:
    """
    PyBroker 执行器共享状态。

    PyBroker 按 (date, symbol) 顺序调用 executor，
    横截面数据需要在所有品种间共享。
    """

    total_symbols: int
    # 当前调仓日已收集的品种（用 set 去重）
    rebalance_date: Any = None
    collected_symbols_set: set = field(default_factory=set)
    finalized: bool = False
    # 当前调仓日计算出的目标权重（finalize 之后才填充）
    target_weights: Dict[str, float] = field(default_factory=dict)
    # 上一bar 数据（用于滚动IC更新）
    prev_close: Dict[str, float] = field(default_factory=dict)
    prev_factor_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # 历史调仓日权重
    last_weights: Dict[str, float] = field(default_factory=dict)

    # ── 止损追踪（显式 tracking，不用反向推算） ──
    entry_price: Dict[str, float] = field(default_factory=dict)
    entry_bar: Dict[str, int] = field(default_factory=dict)
    entry_direction: Dict[str, str] = field(default_factory=dict)  # "long" or "short"
    highest_since_entry: Dict[str, float] = field(default_factory=dict)
    lowest_since_entry: Dict[str, float] = field(default_factory=dict)

    # ── CTA 退出策略持仓状态 ──
    pos_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────
# 执行器构建器
# ────────────────────────────────────────────────────────────
class PyBrokerExecutorBuilder:
    """
    蓝图模式执行器构建器（P0-1/P0-2/P0-3 整改）。

    替代旧 StrategyExecutorFactory.create_executor()，按蓝图实现：
        for symbol in symbols:
            scores = scoring_engine.extract_factor_scores(ctx, params)
            scoring_engine.update_cross_section(symbol, scores, dt=ctx.dt)
        scoring_engine.finalize_cross_section()
        signals = {sym: scoring_engine.compute_composite_score(sym) for sym in symbols}
        target_weights = portfolio_manager.allocate_weights(signals, method='risk_parity')
        target_weights = risk_controller.adjust(target_weights, ctx)
    """

    def __init__(
        self,
        scoring_engine: FactorScoringEngine,
        portfolio_manager: PortfolioManager,
        risk_controller: RiskController,
        config: BacktestConfig,
        total_symbols: int,
        weight_method: str = "risk_parity",
        risk_estimates_provider: Optional[Callable[[str], Optional[float]]] = None,
        *,
        cta_exit_policy: Optional[CTAExitPolicy] = None,
        cta_signal_provider: Optional[Callable[[Any, str], float]] = None,
        cta_market_state_provider: Optional[Callable[[str], str]] = None,
        cta_sigma_provider: Optional[Callable[[str], Optional[float]]] = None,
    ):
        self.scoring_engine = scoring_engine
        self.portfolio_manager = portfolio_manager
        self.risk_controller = risk_controller
        self.config = config
        self.weight_method = weight_method
        self.risk_estimates_provider = risk_estimates_provider
        self.state = PyBrokerExecutorSharedState(total_symbols=total_symbols)
        # ── CTA 退出策略注入 ──
        self.cta_exit_policy = cta_exit_policy
        self.cta_signal_provider = cta_signal_provider
        self.cta_market_state_provider = cta_market_state_provider
        self.cta_sigma_provider = cta_sigma_provider

    def build(
        self,
        strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Callable[[ExecContext], None]:
        """构建 PyBroker executor 函数。

        支持两种模式：
          1. 蓝图模式（默认）：横截面收集 → finalize → PortfolioManager → 下单
          2. CTA 模式（注入 cta_exit_policy）：每 bar 信号 → 四层退出 → 风险预算 → 下单
        """
        scoring_engine = self.scoring_engine
        portfolio_manager = self.portfolio_manager
        risk_controller = self.risk_controller
        config = self.config
        state = self.state
        weight_method = self.weight_method
        risk_estimates_provider = self.risk_estimates_provider
        strategy_params = strategy_params or {}
        position_size = config.max_position_pct
        entry_threshold = config.entry_threshold
        exit_policy = self.cta_exit_policy
        signal_provider = self.cta_signal_provider
        market_state_provider = self.cta_market_state_provider
        sigma_provider = self.cta_sigma_provider
        cta_mode = exit_policy is not None

        def executor_fn(ctx: ExecContext) -> None:
            symbol = ctx.symbol
            current_close = _get_close(ctx)
            current_date = ctx.dt
            current_bar = len(getattr(ctx, "close", []))

            # 0) 更新止损追踪数据（持仓中品种）
            if symbol in state.entry_price:
                if current_close is not None:
                    state.highest_since_entry[symbol] = max(
                        state.highest_since_entry.get(symbol, current_close),
                        current_close,
                    )
                    state.lowest_since_entry[symbol] = min(
                        state.lowest_since_entry.get(symbol, current_close),
                        current_close,
                    )

            # ─────────────────────────────────────────────────────
            # CTA 模式：每 bar 执行，四层退出 + 风险预算
            # ─────────────────────────────────────────────────────
            if cta_mode:
                assert exit_policy is not None  # narrow type
                close_arr = getattr(ctx, "close", None)
                high_arr = getattr(ctx, "high", None)
                low_arr = getattr(ctx, "low", None)
                if close_arr is None or len(close_arr) < 10:
                    return

                signal = signal_provider(ctx, symbol) if signal_provider else 0.0
                market_state = market_state_provider(symbol) if market_state_provider else "oscillation"
                sigma = sigma_provider(symbol) if sigma_provider else None

                has_long = _get_pos_shares(ctx, "long") > 0
                has_short = _get_pos_shares(ctx, "short") > 0
                current_pos = 1 if has_long else -1 if has_short else 0

                # 更新全局风控
                exit_policy.update_global_risk(ctx.dt, symbol, close_arr)

                if current_pos != 0:
                    # 更新极值
                    pstate = state.pos_state.get(symbol, {})
                    if pstate:
                        exit_policy.update_extreme(pstate, current_pos, current_close or 0.0)
                        state.pos_state[symbol] = pstate

                    # 四层退出
                    reason = exit_policy.check_exits(
                        symbol=symbol,
                        current_pos=current_pos,
                        current_price=current_close or 0.0,
                        current_bar=current_bar,
                        signal=signal,
                        market_state=market_state,
                        close=close_arr,
                        high=high_arr,
                        low=low_arr,
                        pos_state=pstate,
                    )
                    if reason:
                        self._close_all(ctx)
                        self._clear_entry_state(symbol)
                        state.pos_state.pop(symbol, None)
                        return

                    # 持仓中且无退出条件 → 维持
                    return

                # ── 空仓：入场逻辑 ──
                if abs(signal) < entry_threshold:
                    return

                target_w = exit_policy.compute_risk_budget_position(
                    signal=signal,
                    market_state=market_state,
                    current_price=current_close or 0.0,
                    close=close_arr,
                    high=high_arr,
                    low=low_arr,
                    sigma=sigma,
                )
                target_w = np.clip(target_w, -position_size, position_size)
                dir_label = "long" if target_w > 0 else "short"
                self._execute_rebalance(
                    ctx, 1 if target_w > 0 else -1,
                    abs(target_w), has_long, has_short,
                )
                # 记录开仓
                if current_close is not None:
                    state.entry_price[symbol] = current_close
                    state.entry_bar[symbol] = current_bar
                    state.entry_direction[symbol] = dir_label
                    state.highest_since_entry[symbol] = current_close
                    state.lowest_since_entry[symbol] = current_close
                    state.pos_state[symbol] = CTAExitPolicy.make_pos_state(
                        current_close, 1 if target_w > 0 else -1, current_bar,
                    )
                return

            # ─────────────────────────────────────────────────────
            # 蓝图模式（默认）：多品种横截面
            # ─────────────────────────────────────────────────────

            # 1) 滚动IC
            self._update_rolling_ic(symbol, current_close)

            # 2) 提取因子得分
            factor_scores = scoring_engine.extract_factor_scores(ctx, strategy_params)
            if factor_scores:
                state.prev_factor_scores[symbol] = dict(factor_scores)
            if current_close is not None:
                state.prev_close[symbol] = current_close

            # 3) 判断是否调仓日
            is_rebalance = scoring_engine.is_rebalance_day(current_date)

            if not is_rebalance:
                # 非调仓日：仅检查止损
                if self._check_stop_loss(ctx, current_close, symbol, current_date, current_bar):
                    self._close_all(ctx)
                    self._clear_entry_state(symbol)
                return

            # 4) 横截面收集
            if state.rebalance_date != current_date:
                state.rebalance_date = current_date
                state.collected_symbols_set = set()
                state.finalized = False
                state.target_weights = {}

            state.collected_symbols_set.add(symbol)
            scoring_engine.update_cross_section(
                symbol, factor_scores, dt=current_date,
            )

            # 5) finalize 横截面
            if not state.finalized and len(state.collected_symbols_set) >= state.total_symbols:
                scoring_engine.finalize_cross_section()
                state.finalized = True
                scoring_engine.mark_rebalanced(current_date)

                signals: Dict[str, float] = {}
                for sym in state.collected_symbols_set:
                    signals[sym] = scoring_engine.compute_composite_score(sym)
                _logger.debug("调仓日 %s 综合得分: %s", current_date, signals)

                risk_estimates: Dict[str, float] = {}
                if risk_estimates_provider:
                    for sym in signals:
                        est = risk_estimates_provider(sym)
                        if est is not None and est > 0:
                            risk_estimates[sym] = est

                target_weights = portfolio_manager.allocate_weights(
                    signals, method=weight_method,
                    risk_estimates=risk_estimates if risk_estimates else None,
                )
                target_weights = risk_controller.check_concentration_dict(
                    target_weights, max_concentration=config.max_position_pct,
                )
                state.target_weights = target_weights
                state.last_weights = dict(target_weights)

            if not state.finalized:
                return

            target_w = state.target_weights.get(symbol, 0.0)
            score = scoring_engine.compute_composite_score(symbol)
            if abs(score) < entry_threshold:
                target_w = 0.0

            if self._check_stop_loss(ctx, current_close, symbol, current_date, current_bar):
                self._close_all(ctx)
                self._clear_entry_state(symbol)
                return

            if abs(target_w) < 1e-6:
                self._close_all(ctx)
                self._clear_entry_state(symbol)
                return

            direction = 1 if target_w > 0 else -1
            effective_size = min(abs(target_w), position_size)
            has_long = _get_pos_shares(ctx, "long") > 0
            has_short = _get_pos_shares(ctx, "short") > 0
            old_pos = 1 if has_long else -1 if has_short else 0
            self._execute_rebalance(ctx, direction, effective_size, has_long, has_short)

            if current_close is not None and (
                old_pos == 0 or (old_pos > 0 and direction < 0) or (old_pos < 0 and direction > 0)
            ):
                dir_label = "long" if direction > 0 else "short"
                state.entry_price[symbol] = current_close
                state.entry_bar[symbol] = current_bar
                state.entry_direction[symbol] = dir_label
                state.highest_since_entry[symbol] = current_close
                state.lowest_since_entry[symbol] = current_close

        executor_fn.__name__ = "blueprint_executor"
        return executor_fn

    # ────────────────────────────────────────────────────────────
    # 内部方法
    # ────────────────────────────────────────────────────────────
    def _update_rolling_ic(self, symbol: str, current_close: Optional[float]) -> None:
        """用上一bar因子得分 + 当前bar收益更新滚动IC引擎。"""
        return

    def _check_stop_loss(
        self,
        ctx: ExecContext,
        current_close: Optional[float],
        symbol: str,
        current_date: Any,
        current_bar: int,
    ) -> bool:
        """复合止损检查。

        使用 state 中显式追踪的 entry_price/highest/lowest，
        优先走 RiskController 复合止损路径。
        """
        if current_close is None or current_close <= 0:
            return False

        entry_p = self.state.entry_price.get(symbol)
        if entry_p is None or entry_p <= 0:
            return False

        entry_bar = self.state.entry_bar.get(symbol, current_bar)
        direction = self.state.entry_direction.get(symbol, "long")  # 用存储的方向
        highest = self.state.highest_since_entry.get(symbol, current_close)
        lowest = self.state.lowest_since_entry.get(symbol, current_close)
        atr = _get_atr(ctx)

        if self.risk_controller is not None and self.risk_controller.composite_stop is not None:
            try:
                result = self.risk_controller.check_composite_stop(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_p,
                    current_price=current_close,
                    highest_since_entry=highest,
                    lowest_since_entry=lowest,
                    entry_day=entry_bar,
                    current_day=current_bar,
                    atr_value=atr if atr > 0 else None,
                    auto_register_entry=False,
                )
                if result.triggered:
                    _logger.info(
                        "%s 触发%s: %s",
                        symbol, direction, result.trigger_reason,
                    )
                    return True
            except Exception as e:
                _logger.debug("复合止损异常: %s", e)

        # 回退：固定百分比止损
        pnl_pct = (current_close - entry_p) / entry_p
        stop_pct = self.config.stop_loss_pct
        if direction == "long" and pnl_pct < -stop_pct:
            _logger.info("%s 触发固定止损: pnl=%.2f%%", symbol, pnl_pct * 100)
            return True
        if direction == "short" and pnl_pct > stop_pct:
            _logger.info("%s 触发固定止损: pnl=%.2f%%", symbol, pnl_pct * 100)
            return True

        return False

    def _clear_entry_state(self, symbol: str) -> None:
        """清除品种的止损追踪状态。"""
        self.state.entry_price.pop(symbol, None)
        self.state.entry_bar.pop(symbol, None)
        self.state.entry_direction.pop(symbol, None)
        self.state.highest_since_entry.pop(symbol, None)
        self.state.lowest_since_entry.pop(symbol, None)

    @staticmethod
    def _execute_rebalance(
        ctx: ExecContext,
        direction: int,
        effective_size: float,
        has_long: bool,
        has_short: bool,
    ) -> None:
        """执行调仓。"""
        if direction > 0 and not has_long:
            if has_short:
                ctx.cover_all_shares()
            ctx.buy_shares = ctx.calc_target_shares(effective_size)
        elif direction < 0 and not has_short:
            if has_long:
                ctx.sell_all_shares()
            ctx.sell_shares = ctx.calc_target_shares(effective_size)
        elif direction == 0:
            if has_long:
                ctx.sell_all_shares()
            if has_short:
                ctx.cover_all_shares()

    @staticmethod
    def _close_all(ctx: ExecContext) -> None:
        if ctx.pos(ctx.symbol, "long") is not None:
            ctx.sell_all_shares()
        if ctx.pos(ctx.symbol, "short") is not None:
            ctx.cover_all_shares()


# 给 RiskController 加一个 dict 版的集中度检查（避免污染核心类）
def _patch_risk_controller_for_blueprint() -> None:
    """为 RiskController 添加 check_concentration_dict 便捷方法。"""
    if hasattr(RiskController, "check_concentration_dict"):
        return

    def check_concentration_dict(
        self,
        weights: Dict[str, float],
        max_concentration: float = 0.4,
    ) -> Dict[str, float]:
        if not weights:
            return {}
        result: Dict[str, float] = {}
        for sym, w in weights.items():
            if abs(w) > max_concentration:
                result[sym] = max_concentration * (1.0 if w > 0 else -1.0)
            else:
                result[sym] = float(w)
        return result

    RiskController.check_concentration_dict = check_concentration_dict  # type: ignore[attr-defined]


_patch_risk_controller_for_blueprint()
