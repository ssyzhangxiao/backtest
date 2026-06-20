"""
CTA 退出策略 — 已迁移到 RiskController 的薄壳兼容层。

⚠️ 2026-06-13 重构：
  - 核心逻辑已迁移到 core.risk_controller.RiskController
  - 本文件保持向后兼容，内部委托给 RiskController 实例
  - 新代码请直接使用 RiskController

新用法::

    from core.risk_controller import RiskController

    rc = RiskController()
    weight = rc.compute_risk_budget_weight(signal=0.5, current_price=100.0, ...)
    if rc.check_logical_falsification(signal, current_pos):
        ...  # exit position
    rc.update_global_risk(date, symbol, daily_return)
    if rc.is_global_risk_triggered():
        ...  # circuit breaker triggered
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from core.risk_controller import RiskConfig as RiskControllerConfig
from core.risk_controller import RiskController

logger = logging.getLogger(__name__)


@dataclass
class CTAExitConfig:
    """CTA 退出策略配置（兼容封装，委托 RiskConfig）。

    2026-06-13：核心字段映射到 RiskConfig，新建项目请直接使用 RiskConfig。
    """

    entry_threshold: float = 0.005
    max_holding_days: int = 60
    atr_stop_multiple: float = 3.0
    atr_window: int = 14
    stop_loss_pct: float = 0.02
    global_risk_pct: float = 0.03
    risk_per_trade: float = 0.02
    target_vol: float = 0.15
    max_position_pct: float = 0.3
    min_signal: float = 0.005
    strategy_name: str = ""

    def to_risk_config(self) -> RiskControllerConfig:
        """转换为 RiskConfig（映射字段）。"""
        return RiskControllerConfig(
            stop_loss_pct=self.stop_loss_pct,
            max_position_pct=self.max_position_pct,
            risk_per_trade=self.risk_per_trade,
            target_vol=self.target_vol,
            global_risk_pct=self.global_risk_pct,
            atr_window=self.atr_window,
            atr_stop_multiple=self.atr_stop_multiple,
            max_holding_days=self.max_holding_days,
            logical_falsification_threshold=0.1,
        )


class CTAExitPolicy:
    """CTA 退出策略引擎 — 兼容封装，委托 RiskController。

    2026-06-13：核心逻辑已迁移到 RiskController，本类维持旧接口。
    ⚠️ 2026-06-20：已标记为 deprecated。请新代码直接使用 ``RiskController``。
    """

    def __init__(self, config: CTAExitConfig):
        import warnings
        warnings.warn(
            "CTAExitPolicy is deprecated since 2026-06-13; use RiskController instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config
        self._risk_controller = RiskController(config.to_risk_config())

    # ────────────────────────────────────────────────────────────
    # 四层退出检查（入口）
    # ────────────────────────────────────────────────────────────

    def check_exits(
        self,
        symbol: str,
        current_pos: int,
        current_price: float,
        current_bar: int,
        signal: float,
        market_state: str,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        pos_state: Dict[str, Any],
    ) -> Optional[str]:
        """四层退出检查（委托 RiskController + 保持逻辑）。"""
        rc = self._risk_controller

        # 第 0 层：策略自退出（信号归零）
        if current_pos != 0 and abs(signal) < self.config.entry_threshold:
            return "StrategyExit"

        if current_pos == 0 or not pos_state:
            return None

        # 第 3 层：时间性强制退出
        entry_bar = pos_state.get("entry_bar_idx", 0)
        if current_bar - entry_bar >= self.config.max_holding_days:
            logger.debug("TimeStop: %d 天", current_bar - entry_bar)
            return "TimeStop"

        # 第 1 层：技术性止损
        reason = self._check_technical_stop(
            symbol, market_state, pos_state, current_price, close, high, low,
        )
        if reason:
            return reason

        # 第 2 层：逻辑性证伪
        if rc.check_logical_falsification(signal, current_pos):
            return "SignalExit"

        # 第 4 层：全局风控
        if rc.is_global_risk_triggered():
            return "GlobalRisk"

        return None

    def _check_technical_stop(self, symbol, market_state, pos_state, current_price, close, high, low):
        """第 1 层：技术性止损。"""
        if market_state == "trend":
            return self._check_atr_trail_stop(symbol, pos_state, current_price, close, high, low)
        return self._check_fixed_stop(pos_state, current_price)

    def _check_atr_trail_stop(self, symbol, pos_state, current_price, close, high, low):
        """ATR 移动止损（趋势市场）。"""
        entry_price = pos_state.get("entry_price", current_price)
        cur_dir = 1 if pos_state.get("direction", 0) >= 0 else -1
        peak = pos_state.get("peak", entry_price)
        trough = pos_state.get("trough", entry_price)

        atr = RiskController.compute_truncated_atr(close, high, low, self.config.atr_window)
        if atr <= 1e-8:
            return None

        stop_distance = self.config.atr_stop_multiple * atr
        if cur_dir > 0:
            trail_stop = peak - stop_distance
            if current_price < trail_stop:
                logger.debug("%s ATRStop: price=%.2f<stop=%.2f", symbol, current_price, trail_stop)
                return "ATRStop"
        else:
            trail_stop = trough + stop_distance
            if current_price > trail_stop:
                logger.debug("%s ATRStop: price=%.2f>stop=%.2f", symbol, current_price, trail_stop)
                return "ATRStop"
        return None

    def _check_fixed_stop(self, pos_state, current_price):
        """固定百分比止损。"""
        entry_price = pos_state.get("entry_price", current_price)
        pnl_pct = (current_price - entry_price) / entry_price
        cur_dir = 1 if pos_state.get("direction", 0) >= 0 else -1
        stop = self.config.stop_loss_pct
        if cur_dir > 0 and pnl_pct < -stop:
            return "FixedStop"
        if cur_dir < 0 and pnl_pct > stop:
            return "FixedStop"
        return None

    # ────────────────────────────────────────────────────────────
    # 风险预算仓位计算（委托 RiskController）
    # ────────────────────────────────────────────────────────────

    def compute_risk_budget_position(
        self,
        signal: float,
        market_state: str,
        current_price: float,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        sigma: Optional[float] = None,
    ) -> float:
        """风险预算仓位计算（委托 RiskController.compute_risk_budget_weight）。"""
        return self._risk_controller.compute_risk_budget_weight(
            signal=signal, current_price=current_price,
            market_state=market_state, close=close, high=high, low=low,
            sigma=sigma,
        )

    # ────────────────────────────────────────────────────────────
    # 风控状态管理（委托 RiskController）
    # ────────────────────────────────────────────────────────────

    def update_global_risk(self, ctx_date: Any, symbol: str, close: np.ndarray) -> None:
        """更新全局风控（委托 RiskController.update_global_risk）。"""
        if close is not None and len(close) >= 2:
            daily_return = (float(close[-1]) - float(close[-2])) / float(close[-2])
            self._risk_controller.update_global_risk(str(ctx_date), symbol, daily_return)

    @staticmethod
    def update_extreme(pos_state: Dict[str, Any], direction: int, price: float) -> None:
        """更新持仓极值（委托 RiskController.update_extreme）。"""
        RiskController.update_extreme(pos_state, direction, price)

    @staticmethod
    def make_pos_state(entry_price: float, direction: int, entry_bar_idx: int) -> Dict[str, Any]:
        """创建持仓状态记录（委托 RiskController.make_pos_state）。"""
        return RiskController.make_pos_state(entry_price, direction, entry_bar_idx)

    # ────────────────────────────────────────────────────────────
    # 工具方法（委托 RiskController 静态方法）
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def compute_truncated_atr(close: np.ndarray, high: np.ndarray, low: np.ndarray, window: int) -> float:
        """计算截断 ATR（委托 RiskController.compute_truncated_atr）。"""
        return RiskController.compute_truncated_atr(close, high, low, window)

    def reset_global_risk(self) -> None:
        """重置全局风控（委托 RiskController.reset_global_risk）。"""
        self._risk_controller.reset_global_risk()


__all__ = ["CTAExitConfig", "CTAExitPolicy"]
