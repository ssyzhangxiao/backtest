"""
因子衰减监控（DEPRECATED 兼容层）。

⚠️ P1-3整改（2026-06-07）：
  本文件已合并到 core/factors/factor_evaluator.py 的 FactorEvaluator.detect_decay() 方法。
  保留本文件仅作为向后兼容层：
    - 旧导入路径仍可使用
    - 所有计算委托给 FactorEvaluator
    - 旧类的接口行为完全等价

新代码请直接使用:
    from core.factors.factor_evaluator import FactorEvaluator
    evaluator = FactorEvaluator(...)
    decay_status = evaluator.detect_decay(ic_history, ...)

位置: core/engine/factor_decay.py（仅作兼容层）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DecayStatus(Enum):
    """因子衰减状态。"""

    HEALTHY = "healthy"
    WARNING = "warning"
    DECAYING = "decaying"
    DEAD = "dead"


@dataclass
class DecayAlert:
    """衰减告警。"""

    factor_name: str
    status: DecayStatus
    current_ic: float
    trend_slope: float
    consecutive_decline: int
    timestamp: str = ""


@dataclass
class FactorDecayConfig:
    """衰减监控配置（兼容层）。"""

    trend_window: int = 40
    ic_healthy_threshold: float = 0.03
    ic_dead_threshold: float = 0.01
    max_consecutive_decline: int = 5
    decay_slope_threshold: float = -0.001


_STATUS_MAP = {
    "healthy": DecayStatus.HEALTHY,
    "warning": DecayStatus.WARNING,
    "decaying": DecayStatus.DECAYING,
    "dead": DecayStatus.DEAD,
}


class FactorDecayMonitor:
    """
    因子衰减监控器（DEPRECATED 兼容层）。

    ⚠️ P1-3整改：核心逻辑已迁移到 FactorEvaluator.detect_decay()。
    本类仅做兼容转发，行为完全等价。
    """

    def __init__(self, config: Optional[FactorDecayConfig] = None):
        self.config = config or FactorDecayConfig()
        self._ic_history: Dict[str, List[float]] = {}
        self._date_history: List[str] = []
        self._alerts: List[DecayAlert] = []
        self._current_status: Dict[str, DecayStatus] = {}

        # 委托给 FactorEvaluator
        from core.factors.factor_evaluator import FactorEvaluator
        self._evaluator = FactorEvaluator()
        logger.debug("FactorDecayMonitor 已委托给 FactorEvaluator.detect_decay")

    @property
    def current_status(self) -> Dict[str, DecayStatus]:
        return dict(self._current_status)

    @property
    def alerts(self) -> List[DecayAlert]:
        return list(self._alerts)

    def update(self, factor_name: str, ic_value: float, date: str = ""):
        """更新因子IC观测。"""
        if factor_name not in self._ic_history:
            self._ic_history[factor_name] = []
        self._ic_history[factor_name].append(float(ic_value))

        if date and (not self._date_history or self._date_history[-1] != date):
            self._date_history.append(date)

    def check_decay(self) -> List[DecayAlert]:
        """
        检测各因子是否衰减（委托给 FactorEvaluator）。

        P1-3整改：实际计算全部交给 FactorEvaluator.detect_decay。
        本方法仅负责历史告警记录与状态变更通知（兼容旧接口）。
        """
        config = self.config
        decay_results = self._evaluator.detect_decay(
            ic_history=self._ic_history,
            trend_window=config.trend_window,
            ic_healthy_threshold=config.ic_healthy_threshold,
            ic_dead_threshold=config.ic_dead_threshold,
            max_consecutive_decline=config.max_consecutive_decline,
            decay_slope_threshold=config.decay_slope_threshold,
        )

        new_alerts: List[DecayAlert] = []
        for name, info in decay_results.items():
            status = _STATUS_MAP[info["status"]]
            prev_status = self._current_status.get(name)
            if prev_status != status:
                alert = DecayAlert(
                    factor_name=name,
                    status=status,
                    current_ic=info["current_ic"],
                    trend_slope=info["trend_slope"],
                    consecutive_decline=info["consecutive_decline"],
                    timestamp=self._date_history[-1] if self._date_history else "",
                )
                new_alerts.append(alert)
                self._alerts.append(alert)
                logger.warning(
                    "因子 %s 状态变更: %s -> %s (IC=%.4f, slope=%.6f)",
                    name,
                    prev_status.value if prev_status else "new",
                    status.value,
                    info["current_ic"],
                    info["trend_slope"],
                )
            self._current_status[name] = status

        return new_alerts

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
                "IC均值": round(float(np.mean(recent)), 6),
                "IC标准差": round(float(np.std(recent)), 6),
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
