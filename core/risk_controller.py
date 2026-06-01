"""
独立风控控制器。

纯风控逻辑类，无任何框架依赖。
BrokerAdapter 不再包含风控决策逻辑，仅负责执行。

风控规则：
  - 止损：单品种亏损超过阈值时强制平仓
  - ATR 动态止损：止损阈值取 max(固定止损, 2*ATR/Close)
  - 持仓上限：总持仓不超过设定比例
  - 单品种仓位上限：单品种不超过设定比例
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import logging

_logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """风控配置。"""

    stop_loss_pct: float = 0.05
    max_position_pct: float = 0.2
    max_total_position_pct: float = 0.6
    stop_loss_cooldown: int = 1


class RiskController:
    """
    风控控制器。

    职责：
      1. 检查单品种是否触发止损
      2. 检查持仓是否超限
      3. 计算有效止损阈值（ATR 动态止损）
      4. 维护止损冷却期

    不负责：
      - 执行交易（由 BrokerAdapter 负责）
      - 因子计算（由 FactorScoringEngine 负责）
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self._cooldown_until: Dict[str, int] = {}

    def check_stop_loss(
        self,
        symbol: str,
        position_market_value: float,
        position_pnl: float,
        current_close: float,
        atr_val: Optional[float] = None,
        trading_day_index: int = 0,
    ) -> bool:
        """
        检查是否触发止损。

        Args:
            symbol: 品种代码
            position_market_value: 持仓市值
            position_pnl: 持仓盈亏
            current_close: 当前收盘价
            atr_val: ATR 值（可选，用于动态止损）
            trading_day_index: 当前交易日序号

        Returns:
            True 表示触发止损
        """
        if position_market_value <= 0:
            return False

        # 冷却期检查
        cooldown_until = self._cooldown_until.get(symbol, 0)
        if trading_day_index < cooldown_until:
            return False

        # 计算有效止损阈值
        effective_stop = self.config.stop_loss_pct
        if atr_val is not None and current_close > 0:
            atr_stop_pct = 2.0 * float(atr_val) / current_close
            effective_stop = max(self.config.stop_loss_pct, atr_stop_pct)

        loss_ratio = -float(position_pnl) / float(position_market_value)
        if loss_ratio > effective_stop:
            self._cooldown_until[symbol] = (
                trading_day_index + self.config.stop_loss_cooldown
            )
            _logger.info(
                "品种 %s 触发止损: 亏损率=%.2f%%, 阈值=%.2f%%",
                symbol, loss_ratio * 100, effective_stop * 100,
            )
            return True

        return False

    def check_position_limit(
        self,
        symbol: str,
        current_position_pct: float,
        total_position_pct: float,
    ) -> bool:
        """
        检查是否超过持仓限制。

        Args:
            symbol: 品种代码
            current_position_pct: 当前品种持仓占总权益比例
            total_position_pct: 总持仓占总权益比例

        Returns:
            True 表示超限
        """
        if current_position_pct > self.config.max_position_pct:
            _logger.warning(
                "品种 %s 仓位超限: %.2f%% > %.2f%%",
                symbol, current_position_pct * 100, self.config.max_position_pct * 100,
            )
            return True

        if total_position_pct > self.config.max_total_position_pct:
            _logger.warning(
                "总仓位超限: %.2f%% > %.2f%%",
                total_position_pct * 100, self.config.max_total_position_pct * 100,
            )
            return True

        return False

    def is_in_cooldown(self, symbol: str, trading_day_index: int) -> bool:
        """检查品种是否在止损冷却期内。"""
        cooldown_until = self._cooldown_until.get(symbol, 0)
        return trading_day_index < cooldown_until

    def clear_cooldown(self, symbol: str):
        """清除品种的冷却期。"""
        self._cooldown_until.pop(symbol, None)
