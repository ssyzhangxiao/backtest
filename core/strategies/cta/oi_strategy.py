"""
持仓量衍生信号 CTA 策略 — 纯信号生成器。

将 compute_oi_signal() 封装为 CTA 策略接口，使 factor pool 可批量调度。

状态依赖：
  - 需要在 _CTABatchWrapper.compute_all() 中预注入 _oi_signal (np.ndarray)
  - compute_signal 直接读取预注入数组对应 bar 位置

配置参数:
    window: OI 波动率滚动窗口（默认 20）
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from core.factors.oi_signal import compute_oi_signal  # noqa: F401  # noqa: F401  # 显式依赖（factor_pool 调用）
from core.strategies.cta.base import CTABaseStrategy
from core.strategies.cta.registry import register_cta_strategy


class OISignalStrategy(CTABaseStrategy):
    """持仓量衍生信号 CTA 策略 — 纯信号生成器。

    配置参数:
        window: OI 波动率窗口（默认 20）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {
            "window": 20,
            **(config or {}),
        }
        super().__init__(merged)

    def compute_signal(
        self,
        symbol: str,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray | None = None,
        ctx: Any = None,
    ) -> float:
        """返回 OI 信号在当前 bar 的取值。

        优先使用预注入的 _oi_signal 数组（batch 计算，避免逐 bar 重复算）。
        如果未注入，则基于 close+open_interest（无法获取）退化为 0。
        """
        cached = self._state.get(symbol, {}).get("_oi_signal")
        if cached is not None and len(cached) >= len(close):
            val = float(cached[len(close) - 1])
            return float(np.clip(val, -1.0, 1.0))
        return 0.0


register_cta_strategy("oi_signal", OISignalStrategy)

__all__ = ["OISignalStrategy"]
