"""
智能策略切换引擎。

根据市场环境变化、策略绩效表现和交易成本，
智能决定何时切换策略，并确保平滑过渡。

核心机制:
  - 多条件触发：环境变化 + 绩效差异 + 成本考量
  - 平滑切换：避免过度交易和持仓冲突
  - 切换冷却：防止频繁无效切换
  - 决策日志：记录每次切换的完整决策过程
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

import pandas as pd
import numpy as np

from core.market_regime import MarketRegime
from core.strategy_library import StrategyLibrary, StrategyProfile


class SwitchReason(Enum):
    """策略切换原因。"""
    REGIME_CHANGE = "regime_change"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    PARAMETER_ADAPTATION = "parameter_adaptation"
    MANUAL = "manual"


@dataclass
class SwitchDecision:
    """策略切换决策。"""
    timestamp: str
    from_strategy: str
    to_strategy: str
    reason: SwitchReason
    regime: str
    confidence: float
    from_sharpe: float
    to_sharpe: float
    expected_improvement: float
    switching_cost: float
    approved: bool
    details: Dict = field(default_factory=dict)


@dataclass
class SwitchConfig:
    """策略切换配置。"""
    # 环境变化触发
    regime_change_threshold: float = 0.3  # 环境置信度阈值
    regime_confirm_days: int = 3  # 环境确认天数

    # 绩效触发
    performance_lookback: int = 20  # 绩效回看天数
    performance_threshold: float = -0.5  # Sharpe低于此值触发
    performance_gap_threshold: float = 0.3  # 策略间Sharpe差距阈值

    # 切换成本
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0002
    max_switching_cost_pct: float = 0.01  # 最大切换成本占比

    # 冷却期
    cooldown_days: int = 5  # 切换后冷却天数

    # 权重过渡
    transition_days: int = 3  # 权重过渡天数


class StrategySwitchEngine:
    """
    智能策略切换引擎。

    综合考虑市场环境转变信号强度、当前策略绩效表现及潜在交易成本，
    决定是否进行策略切换。
    """

    def __init__(self, library: StrategyLibrary, config: Optional[SwitchConfig] = None):
        self.library = library
        self.config = config or SwitchConfig()
        self._decision_log: List[SwitchDecision] = []
        self._last_switch_date: Optional[str] = None
        self._current_strategy: Optional[str] = None
        self._current_regime: Optional[MarketRegime] = None

    @property
    def decision_log(self) -> List[SwitchDecision]:
        return self._decision_log

    def estimate_switching_cost(self, position_value: float, has_position: bool) -> float:
        """估算策略切换成本。"""
        if not has_position:
            return 0.0
        rate = self.config.commission_rate + self.config.slippage_rate
        return position_value * rate * 2  # 平仓+开仓

    def check_cooldown(self, current_date: str) -> bool:
        """检查是否在冷却期内。"""
        if self._last_switch_date is None:
            return False
        last = pd.Timestamp(self._last_switch_date)
        current = pd.Timestamp(current_date)
        return (current - last).days < self.config.cooldown_days

    def evaluate_regime_change(self, current_regime: MarketRegime,
                                new_regime: MarketRegime,
                                confidence: float) -> Tuple[bool, float]:
        """
        评估环境变化是否触发切换。

        Returns:
            (是否触发, 信号强度)
        """
        if current_regime == new_regime:
            return False, 0.0

        signal_strength = confidence
        triggered = signal_strength >= self.config.regime_change_threshold
        return triggered, signal_strength

    def evaluate_performance(self, current_sharpe: float,
                              candidate_sharpe: float) -> Tuple[bool, float]:
        """
        评估绩效差异是否触发切换。

        Returns:
            (是否触发, 预期改善)
        """
        if current_sharpe < self.config.performance_threshold:
            return True, candidate_sharpe - current_sharpe

        gap = candidate_sharpe - current_sharpe
        if gap > self.config.performance_gap_threshold:
            return True, gap

        return False, gap

    def decide(self, current_date: str, current_regime: MarketRegime,
               regime_confidence: float, current_sharpe: float,
               position_value: float, has_position: bool) -> Optional[SwitchDecision]:
        """
        综合决策是否切换策略。

        Args:
            current_date: 当前日期
            current_regime: 当前市场环境
            regime_confidence: 环境识别置信度
            current_sharpe: 当前策略Sharpe
            position_value: 当前持仓市值
            has_position: 是否有持仓

        Returns:
            切换决策，若不需要切换则返回None
        """
        # 检查冷却期
        if self.check_cooldown(current_date):
            return None

        # 环境变化评估
        if self._current_regime is not None:
            regime_triggered, signal_strength = self.evaluate_regime_change(
                self._current_regime, current_regime, regime_confidence
            )
        else:
            regime_triggered = True
            signal_strength = 1.0

        if not regime_triggered:
            return None

        # 获取推荐策略
        candidates = self.library.get_strategies_for_regime(current_regime)
        if not candidates:
            return None

        # 选择最佳候选策略
        best_candidate = None
        best_sharpe = float("-inf")
        for candidate in candidates:
            sharpe = candidate.get_sharpe(current_regime)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_candidate = candidate

        if best_candidate is None:
            return None

        # 如果当前策略就是最佳候选，不需要切换
        if self._current_strategy == best_candidate.name:
            return None

        # 绩效评估
        perf_triggered, expected_improvement = self.evaluate_performance(
            current_sharpe, best_sharpe
        )

        # 切换成本评估
        switching_cost = self.estimate_switching_cost(position_value, has_position)
        max_cost = position_value * self.config.max_switching_cost_pct if position_value > 0 else 0
        cost_acceptable = switching_cost <= max_cost

        # 综合决策
        should_switch = (regime_triggered or perf_triggered) and cost_acceptable

        decision = SwitchDecision(
            timestamp=current_date,
            from_strategy=self._current_strategy or "none",
            to_strategy=best_candidate.name,
            reason=SwitchReason.REGIME_CHANGE if regime_triggered else SwitchReason.PERFORMANCE_DEGRADATION,
            regime=current_regime.value,
            confidence=regime_confidence,
            from_sharpe=current_sharpe,
            to_sharpe=best_sharpe,
            expected_improvement=expected_improvement,
            switching_cost=switching_cost,
            approved=should_switch,
            details={
                "signal_strength": signal_strength,
                "perf_triggered": perf_triggered,
                "cost_acceptable": cost_acceptable,
                "regime_triggered": regime_triggered,
            }
        )

        if should_switch:
            self._decision_log.append(decision)
            self._current_strategy = best_candidate.name
            self._current_regime = current_regime
            self._last_switch_date = current_date

        return decision

    def compute_transition_weights(self, days_since_switch: int) -> Tuple[float, float]:
        """
        计算权重过渡。

        在切换后的transition_days天内，逐步从旧策略过渡到新策略。

        Returns:
            (旧策略权重, 新策略权重)
        """
        if days_since_switch >= self.config.transition_days:
            return 0.0, 1.0
        if days_since_switch <= 0:
            return 1.0, 0.0

        progress = days_since_switch / self.config.transition_days
        new_weight = progress
        old_weight = 1.0 - progress
        return old_weight, new_weight

    def get_current_strategy(self) -> Optional[str]:
        """获取当前活跃策略。"""
        return self._current_strategy

    def set_initial_strategy(self, strategy_name: str, regime: MarketRegime):
        """设置初始策略。"""
        self._current_strategy = strategy_name
        self._current_regime = regime

    def get_decision_summary(self) -> pd.DataFrame:
        """获取决策日志摘要。"""
        if not self._decision_log:
            return pd.DataFrame()
        rows = []
        for d in self._decision_log:
            rows.append({
                "日期": d.timestamp,
                "从策略": d.from_strategy,
                "到策略": d.to_strategy,
                "原因": d.reason.value,
                "环境": d.regime,
                "置信度": d.confidence,
                "预期改善": d.expected_improvement,
                "切换成本": d.switching_cost,
                "已执行": d.approved,
            })
        return pd.DataFrame(rows)
