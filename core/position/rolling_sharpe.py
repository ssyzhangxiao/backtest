"""
滚动Sharpe比率计算模块。

计算策略的滚动Sharpe比率，用于动态仓位调整和策略表现预警。
窗口长度可配置为1M/3M/6M，默认3M。

规则12要求：仓位调整基于滚动Sharpe动态调整。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 默认窗口映射
WINDOW_MAP = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
}

# 年化因子（交易日）
ANNUALIZATION_FACTOR = np.sqrt(252)


@dataclass
class RollingSharpeResult:
    """滚动Sharpe计算结果。"""

    sharpe: float = 0.0
    rolling_mean: float = 0.0
    rolling_std: float = 0.0
    window_used: int = 0
    observations: int = 0


class RollingSharpeManager:
    """
    滚动Sharpe比率管理器。

    计算策略的滚动Sharpe比率，支持多策略并行计算。

    用法:
        manager = RollingSharpeManager(window="3M")
        result = manager.update(daily_returns)
        if result.sharpe < 0:
            # 策略近期表现不佳
    """

    def __init__(
        self,
        window: str = "3M",
        custom_window_days: Optional[int] = None,
        risk_free_rate: float = 0.0,
    ):
        """
        初始化滚动Sharpe管理器。

        Args:
            window: 窗口类型 "1M"/"3M"/"6M"
            custom_window_days: 自定义窗口天数（覆盖window参数）
            risk_free_rate: 无风险利率（年化）
        """
        if custom_window_days is not None:
            self.window_days = custom_window_days
        else:
            self.window_days = WINDOW_MAP.get(window, 63)

        self.risk_free_rate = risk_free_rate
        self._returns_history: List[float] = []

    @property
    def current_sharpe(self) -> float:
        """当前滚动Sharpe值。"""
        if len(self._returns_history) < self.window_days:
            return 0.0
        returns = np.array(self._returns_history[-self.window_days:])
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret < 1e-10:
            return 0.0
        daily_rf = self.risk_free_rate / 252
        return (mean_ret - daily_rf) / std_ret * ANNUALIZATION_FACTOR

    def update(self, daily_return: float) -> RollingSharpeResult:
        """
        更新日收益率并计算滚动Sharpe。

        Args:
            daily_return: 当日收益率

        Returns:
            RollingSharpeResult 计算结果
        """
        self._returns_history.append(daily_return)

        # 限制历史长度
        max_len = self.window_days * 3
        if len(self._returns_history) > max_len:
            self._returns_history = self._returns_history[-max_len:]

        n = len(self._returns_history)
        if n < self.window_days:
            return RollingSharpeResult(
                sharpe=0.0,
                rolling_mean=0.0,
                rolling_std=0.0,
                window_used=self.window_days,
                observations=n,
            )

        returns = np.array(self._returns_history[-self.window_days:])
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))

        if std_ret < 1e-10:
            sharpe = 0.0
        else:
            daily_rf = self.risk_free_rate / 252
            sharpe = (mean_ret - daily_rf) / std_ret * ANNUALIZATION_FACTOR

        return RollingSharpeResult(
            sharpe=sharpe,
            rolling_mean=mean_ret,
            rolling_std=std_ret,
            window_used=self.window_days,
            observations=n,
        )

    def compute_series(
        self, returns: np.ndarray
    ) -> np.ndarray:
        """
        计算完整滚动Sharpe序列。

        Args:
            returns: 日收益率序列

        Returns:
            滚动Sharpe序列（前window_days-1个为NaN）
        """
        ret = np.asarray(returns, dtype=float)
        n = len(ret)
        result = np.full(n, np.nan, dtype=float)
        w = self.window_days

        if n < w:
            return result

        for i in range(w, n + 1):
            window_ret = ret[i - w : i]
            mean_r = np.mean(window_ret)
            std_r = np.std(window_ret)
            if std_r < 1e-10:
                result[i - 1] = 0.0
            else:
                daily_rf = self.risk_free_rate / 252
                result[i - 1] = (mean_r - daily_rf) / std_r * ANNUALIZATION_FACTOR

        return result

    def reset(self):
        """重置历史数据。"""
        self._returns_history.clear()
