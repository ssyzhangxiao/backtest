"""
ATR倍数动态调整适配器。

根据波动率分位数自动调整ATR倍数：
  - 分位数 < 25%：ATR倍数 = 0.5（紧止损）
  - 分位数 25-75%：ATR倍数 = 1.5（标准止损）
  - 分位数 > 75%：ATR倍数 = 3.0（宽止损）

规则10要求：ATR倍数范围0.5~3.0，按波动率分位数分档。
"""

from dataclasses import dataclass
from typing import Optional
import logging

from .vol_monitor import VolRegime

logger = logging.getLogger(__name__)

# ATR倍数范围限制（规则10）
MIN_ATR_MULT = 0.5
MAX_ATR_MULT = 3.0


@dataclass
class ATRAdapterConfig:
    """ATR自适应配置。"""

    # 各regime对应的ATR倍数
    low_vol_mult: float = 0.5
    medium_vol_mult: float = 1.5
    high_vol_mult: float = 3.0

    # 基准倍数
    base_mult: float = 1.5


class AdaptiveATR:
    """
    ATR倍数动态调整适配器。

    根据波动率regime自动调整ATR止损倍数。
    高波动率环境使用更宽的止损，低波动率使用更紧的止损。

    用法:
        atr_adapter = AdaptiveATR()
        mult = atr_adapter.get_multiplier(regime)
        stop_price = entry_price - mult * atr_value
    """

    def __init__(self, config: Optional[ATRAdapterConfig] = None):
        self.config = config or ATRAdapterConfig()
        self._current_mult: float = self.config.base_mult

    @property
    def current_multiplier(self) -> float:
        """当前ATR倍数。"""
        return self._current_mult

    def get_target_multiplier(self, regime: VolRegime) -> float:
        """
        根据regime获取目标ATR倍数。

        Args:
            regime: 波动率regime

        Returns:
            目标ATR倍数
        """
        if regime == VolRegime.LOW:
            return self.config.low_vol_mult
        elif regime == VolRegime.HIGH:
            return self.config.high_vol_mult
        else:
            return self.config.medium_vol_mult

    def adjust_multiplier(self, regime: VolRegime) -> float:
        """
        根据regime调整ATR倍数。

        Args:
            regime: 波动率regime

        Returns:
            调整后的ATR倍数
        """
        target = self.get_target_multiplier(regime)

        # 裁剪到合法范围
        target = max(MIN_ATR_MULT, min(MAX_ATR_MULT, target))

        if target != self._current_mult:
            logger.debug(
                f"ATR倍数调整：{self._current_mult:.1f}→{target:.1f} "
                f"(regime={regime.value})"
            )
            self._current_mult = target

        return self._current_mult

    def compute_stop_price(
        self,
        entry_price: float,
        atr_value: float,
        regime: VolRegime,
        direction: str = "long",
    ) -> float:
        """
        计算ATR动态止损价。

        Args:
            entry_price: 入场价格
            atr_value: ATR值
            regime: 波动率regime
            direction: 持仓方向 "long" 或 "short"

        Returns:
            止损价格
        """
        mult = self.adjust_multiplier(regime)
        stop_distance = mult * atr_value

        if direction == "long":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def reset(self):
        """重置到基准倍数。"""
        self._current_mult = self.config.base_mult
