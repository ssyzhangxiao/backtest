"""CTA 策略抽象基类。

每个品种独立运行，策略内部按品种维护状态。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class CTABaseStrategy(ABC):
    """CTA 策略基类。

    子类必须实现 compute_signal() 方法。
    可选实现 on_entry() / on_exit() 回调。

    每个实例维护一个 _state 字典，按品种存储状态。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._state: Dict[str, Dict[str, Any]] = {}

    # ── 核心接口 ──

    @abstractmethod
    def compute_signal(
        self,
        symbol: str,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: Optional[np.ndarray] = None,
        ctx: Any = None,
    ) -> float:
        """计算当前品种的 CTA 信号。

        Args:
            symbol: 品种代码
            close: 收盘价序列（最新在末尾）
            high: 最高价序列
            low: 最低价序列
            volume: 成交量序列
            ctx: PyBroker ExecContext（可选，用于获取额外数据）

        Returns:
            信号值 [-1, 1]，正=做多，负=做空，0=空仓
        """

    # ── 可选回调 ──

    def on_entry(self, symbol: str, price: float) -> None:
        """开仓回调。"""

    def on_exit(self, symbol: str, price: float) -> None:
        """平仓回调。"""

    # ── 状态管理 ──

    def get_state(self, symbol: str, key: str, default: Any = None) -> Any:
        """读取品种状态。"""
        return self._state.get(symbol, {}).get(key, default)

    def set_state(self, symbol: str, key: str, value: Any) -> None:
        """写入品种状态。"""
        if symbol not in self._state:
            self._state[symbol] = {}
        self._state[symbol][key] = value

    def clear_state(self, symbol: str) -> None:
        """清除品种状态。"""
        self._state.pop(symbol, None)

    # ── 工具方法 ──

    @staticmethod
    def _validate(close: np.ndarray, min_len: int = 30) -> bool:
        """验证输入数据是否足够计算。"""
        if close is None or len(close) < min_len:
            return False
        if np.isnan(close).all():
            return False
        return True

    @staticmethod
    def _clip_signal(signal: float) -> float:
        """裁剪信号到 [-1, 1]。"""
        return float(np.clip(signal, -1.0, 1.0))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
