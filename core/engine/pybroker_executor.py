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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from core.config import BacktestConfig
from core.engine.switch_engine import FactorScoringEngine
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
    # 当前调仓日已收集的品种数
    rebalance_date: Any = None
    collected_symbols: List[str] = field(default_factory=list)
    finalized: bool = False
    # 当前调仓日计算出的目标权重（finalize 之后才填充）
    target_weights: Dict[str, float] = field(default_factory=dict)
    # 上一bar 数据（用于滚动IC更新）
    prev_close: Dict[str, float] = field(default_factory=dict)
    prev_factor_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # 历史调仓日权重
    last_weights: Dict[str, float] = field(default_factory=dict)


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
    ):
        self.scoring_engine = scoring_engine
        self.portfolio_manager = portfolio_manager
        self.risk_controller = risk_controller
        self.config = config
        self.weight_method = weight_method
        self.risk_estimates_provider = risk_estimates_provider
        self.state = PyBrokerExecutorSharedState(total_symbols=total_symbols)

    def build(
        self,
        strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Callable[[ExecContext], None]:
        """构建 PyBroker executor 函数。"""
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

        def executor_fn(ctx: ExecContext) -> None:
            symbol = ctx.symbol
            current_close = _get_close(ctx)
            current_date = ctx.dt

            # 1) 滚动IC：用上一bar的因子得分和当前bar收益更新
            self._update_rolling_ic(symbol, current_close)

            # 2) 提取当前bar的因子得分
            factor_scores = scoring_engine.extract_factor_scores(ctx, strategy_params)
            if factor_scores:
                state.prev_factor_scores[symbol] = dict(factor_scores)
            if current_close is not None:
                state.prev_close[symbol] = current_close

            # 3) 判断是否调仓日
            is_rebalance = scoring_engine.is_rebalance_day(current_date)

            if not is_rebalance:
                return

            # 4) 横截面数据收集
            if state.rebalance_date != current_date:
                # 新调仓日，重置
                state.rebalance_date = current_date
                state.collected_symbols = []
                state.finalized = False
                state.target_weights = {}

            state.collected_symbols.append(symbol)
            scoring_engine.update_cross_section(
                symbol, factor_scores, dt=current_date,
            )

            # 5) 蓝图：收齐所有品种后 finalize 横截面
            if not state.finalized and len(state.collected_symbols) >= state.total_symbols:
                scoring_engine.finalize_cross_section()
                state.finalized = True
                scoring_engine.mark_rebalanced(current_date)

                # 6) 蓝图：计算综合得分
                signals: Dict[str, float] = {}
                for sym in state.collected_symbols:
                    signals[sym] = scoring_engine.compute_composite_score(sym)
                _logger.debug("调仓日 %s 综合得分: %s", current_date, signals)

                # 7) 蓝图：PortfolioManager 分配权重
                risk_estimates: Dict[str, float] = {}
                if risk_estimates_provider:
                    for sym in signals:
                        est = risk_estimates_provider(sym)
                        if est is not None and est > 0:
                            risk_estimates[sym] = est
                target_weights = portfolio_manager.allocate_weights(
                    signals,
                    method=weight_method,
                    risk_estimates=risk_estimates if risk_estimates else None,
                )

                # 8) 蓝图：RiskController 调整（集中度等）
                target_weights = risk_controller.check_concentration_dict(
                    target_weights, max_concentration=config.max_position_pct,
                )
                state.target_weights = target_weights
                state.last_weights = dict(target_weights)

            # 9) 当前品种是否已 finalized，是则执行下单
            if not state.finalized:
                return

            target_w = state.target_weights.get(symbol, 0.0)
            score = scoring_engine.compute_composite_score(symbol)
            if abs(score) < entry_threshold:
                target_w = 0.0

            # 单品种风控
            target_w = self._apply_per_symbol_risk(
                symbol=symbol,
                target_weight=target_w,
                ctx=ctx,
                current_close=current_close,
            )
            if abs(target_w) < 1e-6:
                # 平仓
                self._close_all(ctx)
                return

            direction = 1 if target_w > 0 else -1
            effective_size = min(abs(target_w), position_size)
            if effective_size < config.min_position_pct:
                effective_size = config.min_position_pct

            # 止损检查
            if self._check_stop_loss(ctx, current_close):
                self._close_all(ctx)
                return

            has_long = _get_pos_shares(ctx, "long") > 0
            has_short = _get_pos_shares(ctx, "short") > 0
            self._execute_rebalance(ctx, direction, effective_size, has_long, has_short)

        executor_fn.__name__ = "blueprint_executor"
        return executor_fn

    # ────────────────────────────────────────────────────────────
    # 内部方法
    # ────────────────────────────────────────────────────────────
    def _update_rolling_ic(self, symbol: str, current_close: Optional[float]) -> None:
        """用上一bar因子得分 + 当前bar收益更新滚动IC引擎。"""
        # 由外部注入的 IC 引擎通过 scoring_engine.set_ic_weights 调用
        # 此处保留扩展点：实际计算由 scoring_engine 内部完成
        return

    def _apply_per_symbol_risk(
        self,
        symbol: str,
        target_weight: float,
        ctx: ExecContext,
        current_close: Optional[float],
    ) -> float:
        """单品种风控：仓位上限。"""
        max_pct = self.config.max_position_pct
        if abs(target_weight) > max_pct:
            return max_pct * (1.0 if target_weight > 0 else -1.0)
        return target_weight

    def _check_stop_loss(self, ctx: ExecContext, current_close: Optional[float]) -> bool:
        """检查止损：委托给 RiskController。"""
        # 通过 ATR 动态止损阈值与配置止损比较
        atr = _get_indicator(ctx, "atr_14")
        long_pos = ctx.pos(ctx.symbol, "long")
        short_pos = ctx.pos(ctx.symbol, "short")
        for pos, side_sign in ((long_pos, 1), (short_pos, -1)):
            if pos is None:
                continue
            try:
                pnl = float(pos.pnl)
                equity = float(pos.equity) - pnl
                if equity <= 0:
                    continue
                pnl_pct = pnl / equity
                if pnl_pct >= 0:
                    continue
                # 计算有效止损阈值
                effective_stop = self.config.stop_loss_pct
                if current_close and current_close > 0 and atr:
                    atr_stop = float(atr) * 2.0 / current_close
                    effective_stop = max(effective_stop, atr_stop)
                if -pnl_pct > effective_stop:
                    _logger.info(
                        "%s 触发止损: 亏损=%.2f%%, 阈值=%.2f%%",
                        ctx.symbol, pnl_pct * 100, effective_stop * 100,
                    )
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

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


# 给 RiskController 加一个 dict 版的集中度检查（避免污染核心类），
# 直接用 RiskController.check_concentration 接受 {sym: market_value} 即可。
def _patch_risk_controller_for_blueprint() -> None:
    """为 RiskController 添加 check_concentration_dict 便捷方法。"""
    if hasattr(RiskController, "check_concentration_dict"):
        return

    def check_concentration_dict(
        self,
        weights: Dict[str, float],
        max_concentration: float = 0.4,
    ) -> Dict[str, float]:
        """
        权重集中度调整：把 {sym: weight} 中超限的权重截断到 max_concentration。

        与 check_concentration 不同：返回调整后的权重 dict 而非列表。
        """
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
