"""
因子衰减监控。

跟踪因子IC随时间的变化趋势，检测因子性能衰减：
  - 滚动IC趋势检测：IC是否显著下滑
  - 衰减告警：IC低于阈值或连续下降
  - 衰减历史记录

位置: core/engine/factor_decay.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DecayStatus(Enum):
    """因子衰减状态。"""

    HEALTHY = "healthy"         # 健康：IC稳定
    WARNING = "warning"         # 警告：IC下降趋势
    DECAYING = "decaying"       # 衰减：IC持续低于阈值
    DEAD = "dead"               # 失效：IC趋近于零或为负


@dataclass
class DecayAlert:
    """衰减告警。"""

    factor_name: str
    status: DecayStatus
    current_ic: float
    trend_slope: float           # IC趋势斜率
    consecutive_decline: int     # 连续下降次数
    timestamp: str = ""


@dataclass
class FactorDecayConfig:
    """衰减监控配置。"""

    # 趋势检测窗口（交易日数）
    trend_window: int = 40

    # IC健康阈值（绝对值低于此值视为衰减）
    ic_healthy_threshold: float = 0.03

    # IC死区阈值（绝对值低于此值视为失效）
    ic_dead_threshold: float = 0.01

    # 连续下降次数触发告警
    max_consecutive_decline: int = 5

    # 衰减斜率阈值（负斜率绝对值超过此值视为趋势性衰减）
    decay_slope_threshold: float = -0.001


class FactorDecayMonitor:
    """
    因子衰减监控器。

    跟踪每个因子的滚动IC，检测性能衰减趋势。

    使用方式:
        monitor = FactorDecayMonitor()
        for each bar:
            monitor.update(factor_name, ic_value, date)
        alerts = monitor.check_decay()
    """

    def __init__(self, config: Optional[FactorDecayConfig] = None):
        self.config = config or FactorDecayConfig()
        self._ic_history: Dict[str, List[float]] = {}
        self._date_history: List[str] = []
        self._alerts: List[DecayAlert] = []
        self._current_status: Dict[str, DecayStatus] = {}

    @property
    def current_status(self) -> Dict[str, DecayStatus]:
        """各因子当前衰减状态。"""
        return dict(self._current_status)

    @property
    def alerts(self) -> List[DecayAlert]:
        """告警列表。"""
        return list(self._alerts)

    def update(self, factor_name: str, ic_value: float, date: str = ""):
        """
        更新因子IC观测。

        Args:
            factor_name: 因子名称
            ic_value: 当期IC值
            date: 日期（可选）
        """
        if factor_name not in self._ic_history:
            self._ic_history[factor_name] = []
        self._ic_history[factor_name].append(float(ic_value))

        if date and (not self._date_history or self._date_history[-1] != date):
            self._date_history.append(date)

    def check_decay(self) -> List[DecayAlert]:
        """
        检测各因子是否衰减。

        Returns:
            衰减告警列表
        """
        new_alerts: List[DecayAlert] = []
        config = self.config

        for name, ic_series in self._ic_history.items():
            if len(ic_series) < config.trend_window:
                continue

            recent = ic_series[-config.trend_window:]
            current_ic = recent[-1]
            abs_ic = abs(current_ic)

            # 1. 判断衰减状态
            if abs_ic < config.ic_dead_threshold:
                status = DecayStatus.DEAD
            elif abs_ic < config.ic_healthy_threshold:
                # 检查是否有下降趋势
                if len(recent) >= 10:
                    x = np.arange(len(recent))
                    slope, _ = np.polyfit(x, recent, 1)
                else:
                    slope = 0.0

                if slope < config.decay_slope_threshold:
                    status = DecayStatus.DECAYING
                else:
                    status = DecayStatus.WARNING
            else:
                status = DecayStatus.HEALTHY

            # 2. 检测连续下降
            consecutive = self._count_consecutive_decline(recent)

            if consecutive >= config.max_consecutive_decline:
                if status == DecayStatus.HEALTHY:
                    status = DecayStatus.WARNING

            # 3. IC趋势斜率
            if len(recent) >= 10:
                x = np.arange(len(recent))
                trend_slope, _ = np.polyfit(x, recent, 1)
            else:
                trend_slope = 0.0

            prev_status = self._current_status.get(name)

            if prev_status != status:
                alert = DecayAlert(
                    factor_name=name,
                    status=status,
                    current_ic=round(current_ic, 6),
                    trend_slope=round(trend_slope, 6),
                    consecutive_decline=consecutive,
                    timestamp=self._date_history[-1] if self._date_history else "",
                )
                new_alerts.append(alert)
                self._alerts.append(alert)
                logger.warning(
                    "因子 %s 状态变更: %s -> %s (IC=%.4f, slope=%.6f)",
                    name, prev_status.value if prev_status else "new",
                    status.value, current_ic, trend_slope,
                )

            self._current_status[name] = status

        return new_alerts

    @staticmethod
    def _count_consecutive_decline(series: List[float]) -> int:
        """计算序列末尾连续下降次数。"""
        if len(series) < 2:
            return 0
        count = 0
        for i in range(len(series) - 1, 0, -1):
            if series[i] < series[i - 1]:
                count += 1
            else:
                break
        return count

    def get_decay_summary(self) -> pd.DataFrame:
        """获取衰减监控摘要。"""
        if not self._ic_history:
            return pd.DataFrame()

        rows = []
        for name, ic_series in self._ic_history.items():
            if len(ic_series) < 2:
                continue
            recent = ic_series[-self.config.trend_window:]
            status = self._current_status.get(name, DecayStatus.HEALTHY)

            rows.append({
                "因子": name,
                "当前IC": round(ic_series[-1], 6),
                "IC均值": round(np.mean(recent), 6),
                "IC标准差": round(np.std(recent), 6),
                "观测数": len(ic_series),
                "状态": status.value,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("当前IC", ascending=False).reset_index(drop=True)
        return df

    def reset(self):
        """重置所有状态。"""
        self._ic_history.clear()
        self._date_history.clear()
        self._alerts.clear()
        self._current_status.clear()