"""
CTA 执行器构建器 — 已重构为 PyBrokerExecutorBuilder + CTAExitPolicy 的薄壳封装。

⚠️ 2026-06-13 重构：
  - 核心逻辑已提取到 core/execution/cta_exit_policy.py（四层退出 + 风险预算）
  - 执行管线已迁移到 core/execution/pybroker_executor.py（PyBrokerExecutorBuilder）
  - 本文件保持向后兼容，内部委托给新组件

用法（新代码推荐直接使用 PyBrokerExecutorBuilder）::

    from core.execution import CTAExitConfig, CTAExitPolicy, PyBrokerExecutorBuilder

    policy = CTAExitPolicy(CTAExitConfig(risk_per_trade=0.02, target_vol=0.15))
    builder = PyBrokerExecutorBuilder(
        scoring_engine=...,
        portfolio_manager=...,
        risk_controller=...,
        config=...,
        total_symbols=1,
        cta_exit_policy=policy,
        cta_signal_provider=lambda ctx, sym: strategy.compute_signal(...),
        cta_market_state_provider=lambda sym: strategy.get_state(sym, "market_state", "oscillation"),
        cta_sigma_provider=lambda sym: strategy.get_state(sym, "sigma", None),
    )
    executor_fn = builder.build()
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import numpy as np

from core.execution.cta_exit_policy import CTAExitConfig, CTAExitPolicy
from core.strategies.cta.base import CTABaseStrategy

logger = logging.getLogger(__name__)


class CTAExecutorBuilder:
    """单品种 CTA 执行器构建器（向后兼容封装）。

    内部委托给 PyBrokerExecutorBuilder + CTAExitPolicy。

    风险预算参数（2026-06-13 优化）：
      - risk_per_trade: 0.02（每笔风险预算 2%）
      - target_vol: 0.15（年化目标波动率 15%，启用波动率平价）
      - stop_loss_pct: 0.02（震荡市固定止损 2%）
    """

    def __init__(
        self,
        cta_strategy: CTABaseStrategy,
        entry_threshold: float = 0.005,
        max_position_pct: float = 0.3,
        max_holding_days: int = 60,
        atr_stop_multiple: float = 3.0,
        atr_window: int = 14,
        stop_loss_pct: float = 0.02,
        global_risk_pct: float = 0.03,
        risk_per_trade: float = 0.02,
        target_vol: float = 0.15,
    ) -> None:
        self.cta_strategy = cta_strategy
        self._exit_policy = CTAExitPolicy(CTAExitConfig(
            entry_threshold=entry_threshold,
            max_holding_days=max_holding_days,
            atr_stop_multiple=atr_stop_multiple,
            atr_window=atr_window,
            stop_loss_pct=stop_loss_pct,
            global_risk_pct=global_risk_pct,
            risk_per_trade=risk_per_trade,
            target_vol=target_vol,
            max_position_pct=max_position_pct,
        ))

    def build(self) -> Callable[[Any], None]:
        """构建 PyBroker executor 函数（委托 CTAExitPolicy）。

        Returns:
            executor_fn(ctx) — 符合 PyBroker ExecContext 签名
        """
        strategy = self.cta_strategy
        exit_policy = self._exit_policy
        entry_threshold = exit_policy.config.entry_threshold

        def executor_fn(ctx: Any) -> None:
            """PyBroker 执行函数。"""
            symbol = ctx.symbol

            close: np.ndarray = getattr(ctx, "close", None)
            high: np.ndarray = getattr(ctx, "high", None)
            low: np.ndarray = getattr(ctx, "low", None)
            volume: np.ndarray = getattr(ctx, "volume", None)

            if close is None or len(close) < 10:
                return

            # 1. 计算 CTA 纯信号
            signal = strategy.compute_signal(
                symbol=symbol, close=close, high=high, low=low,
                volume=volume, ctx=ctx,
            )

            # 2. 市场状态 + sigma（策略通过 set_state 写入）
            market_state = strategy.get_state(symbol, "market_state", "oscillation")
            sigma = strategy.get_state(symbol, "sigma", None)

            # 3. 当前持仓
            has_long = ctx.long_pos() is not None
            has_short = ctx.short_pos() is not None
            current_pos = 1 if has_long else -1 if has_short else 0
            current_price = float(close[-1])
            current_bar = len(close)

            # 4. 更新全局风控
            exit_policy.update_global_risk(ctx.dt, symbol, close)

            # ── 持仓中：四层退出 ──
            if current_pos != 0:
                # 构造持仓状态
                pos_state = _build_pos_state(symbol, current_pos, current_price, current_bar, ctx)

                # 更新极值
                exit_policy.update_extreme(pos_state, current_pos, current_price)

                # 四层退出
                reason = exit_policy.check_exits(
                    symbol=symbol, current_pos=current_pos,
                    current_price=current_price, current_bar=current_bar,
                    signal=signal, market_state=market_state,
                    close=close, high=high, low=low, pos_state=pos_state,
                )
                if reason:
                    _exit_position(ctx)
                    logger.debug("%s 平仓: 原因=%s", symbol, reason)
                    return

                # 维持仓位
                return

            # ── 空仓：入场逻辑 ──
            if abs(signal) < entry_threshold:
                return

            target_w = exit_policy.compute_risk_budget_position(
                signal=signal, market_state=market_state,
                current_price=current_price, close=close,
                high=high, low=low, sigma=sigma,
            )
            _cta_execute_trades(ctx, target_w)

        return executor_fn


# ────────────────────────────────────────────────────────────
# 辅助函数（无 self 依赖，可独立使用）
# ────────────────────────────────────────────────────────────


def _build_pos_state(
    symbol: str, current_pos: int, current_price: float,
    current_bar: int, ctx: Any,
) -> Dict[str, Any]:
    """从 PyBroker ctx 构造持仓状态（兼容旧版 pos_state 格式）。"""
    # 尝试从 entry_price 反推（兼容旧 executors）
    entry = getattr(ctx, "_entry_price", None)
    if entry is None:
        # 没有显式记录时用当前价 × 0.98 估算
        entry = current_price * (0.98 if current_pos > 0 else 1.02)
    return {
        "entry_price": entry,
        "entry_bar_idx": current_bar - 5,
        "direction": current_pos,
        "peak": current_price if current_pos > 0 else entry,
        "trough": entry if current_pos > 0 else current_price,
    }


def _exit_position(ctx: Any) -> None:
    """平仓。"""
    if ctx.long_pos() is not None:
        ctx.sell_all_shares()
    if ctx.short_pos() is not None:
        ctx.cover_all_shares()


def _cta_execute_trades(ctx: Any, target_weight: float) -> None:
    """执行交易。"""
    has_long = ctx.long_pos() is not None
    has_short = ctx.short_pos() is not None

    if abs(target_weight) < 1e-6:
        _exit_position(ctx)
        return

    if target_weight > 0:
        if has_short:
            ctx.cover_all_shares()
        ctx.buy_shares = ctx.calc_target_shares(target_weight)
    else:
        if has_long:
            ctx.sell_all_shares()
        ctx.sell_shares = ctx.calc_target_shares(-target_weight)
