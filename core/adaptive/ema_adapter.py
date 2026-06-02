"""
EMA窗口自适应适配器。

根据波动率regime动态调整EMA窗口：
  - 高波动率：缩短窗口（快速响应）
  - 低波动率：延长窗口（过滤噪音）

规则10要求：窗口范围3~20日，单次调整步长≤2日。
"""

from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np

from .vol_monitor import VolRegime

logger = logging.getLogger(__name__)

# 窗口范围限制（规则10）
MIN_WINDOW = 3
MAX_WINDOW = 20
MAX_STEP = 2


@dataclass
class EMAAdapterConfig:
    """EMA自适应配置。"""

    # 基准窗口
    base_window: int = 5

    # 各regime对应的目标窗口
    low_vol_window: int = 10
    medium_vol_window: int = 5
    high_vol_window: int = 3

    # 单次最大调整步长
    max_step: int = MAX_STEP


class AdaptiveEMA:
    """
    EMA窗口自适应适配器。

    根据波动率regime动态调整EMA计算窗口。
    窗口变化受步长限制，避免突变。

    用法:
        ema = AdaptiveEMA()
        ema_value = ema.compute(close_prices, regime)
    """

    def __init__(self, config: Optional[EMAAdapterConfig] = None):
        self.config = config or EMAAdapterConfig()
        self._current_window: int = self.config.base_window

    @property
    def current_window(self) -> int:
        """当前EMA窗口。"""
        return self._current_window

    def get_target_window(self, regime: VolRegime) -> int:
        """
        根据regime获取目标窗口。

        Args:
            regime: 波动率regime

        Returns:
            目标窗口大小
        """
        if regime == VolRegime.LOW:
            target = self.config.low_vol_window
        elif regime == VolRegime.HIGH:
            target = self.config.high_vol_window
        else:
            target = self.config.medium_vol_window

        # 裁剪到合法范围
        return max(MIN_WINDOW, min(MAX_WINDOW, target))

    def adjust_window(self, regime: VolRegime) -> int:
        """
        根据regime调整窗口，受步长限制。

        规则10：单次调整步长≤2日。

        Args:
            regime: 波动率regime

        Returns:
            调整后的窗口大小
        """
        target = self.get_target_window(regime)
        diff = target - self._current_window

        # 步长限制
        if abs(diff) > self.config.max_step:
            step = self.config.max_step * np.sign(diff)
            new_window = int(self._current_window + step)
        else:
            new_window = target

        # 范围限制
        new_window = max(MIN_WINDOW, min(MAX_WINDOW, new_window))

        if new_window != self._current_window:
            logger.debug(
                f"EMA窗口调整：{self._current_window}→{new_window} "
                f"(regime={regime.value}, target={target})"
            )
            self._current_window = new_window

        return self._current_window

    def compute(self, data: np.ndarray, regime: VolRegime) -> float:
        """
        计算自适应EMA值。

        先根据regime调整窗口，再计算EMA。

        Args:
            data: 价格序列
            regime: 波动率regime

        Returns:
            EMA值
        """
        window = self.adjust_window(regime)
        arr = np.asarray(data, dtype=float)

        if len(arr) < window:
            return float(arr[-1]) if len(arr) > 0 else 0.0

        # EMA计算
        alpha = 2.0 / (window + 1)
        ema = float(arr[-window])
        for i in range(-window + 1, 0):
            ema = alpha * arr[i] + (1 - alpha) * ema

        return ema

    def reset(self):
        """重置到基准窗口。"""
        self._current_window = self.config.base_window
