"""
智能策略切换引擎（v2 - 修复设计缺陷 + 增强决策逻辑）。

根据市场环境变化、策略绩效表现和交易成本，
智能决定何时切换策略，并确保平滑过渡。

核心机制:
  - 多条件触发：环境变化 + 绩效差异 + 成本考量
  - 平滑切换：避免过度交易和持仓冲突
  - 切换冷却：基于交易日序号，防止频繁无效切换
  - 决策日志：记录每次切换的完整决策过程，支持事后绩效跟踪

⚠️ 非线程安全：可变状态（_decision_log、_last_switch_index、_current_strategy）
  未加锁保护，不应在多线程环境下并发调用。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import warnings
import uuid

import pandas as pd
import numpy as np

from core.market_regime import MarketRegime
from core.strategy_library import StrategyLibrary, StrategyProfile


class SwitchReason(Enum):
    """策略切换原因。"""

    REGIME_CHANGE = "regime_change"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MANUAL = "manual"  # 降级切换等场景


@dataclass
class SwitchConfig:
    """策略切换配置。"""

    # 环境变化触发
    regime_change_threshold: float = 0.3  # 环境置信度阈值
    regime_confirm_days: int = 3  # 环境确认天数

    # 绩效触发
    performance_lookback: int = 20  # 绩效回看天数
    performance_threshold: float = -0.5  # Sharpe低于此值触发
    performance_gap_threshold: float = 0.3  # 策略间Sharpe差距阈值（保留兼容）

    # 综合评分最低Sharpe提升阈值（避免微小差异频繁切换）
    min_sharpe_gap: float = 0.15

    # 绩效样本最低天数（不足则强制不触发绩效切换）
    min_samples: int = 10

    # 切换成本
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0002
    max_switching_cost_pct: float = 0.01  # 最大切换成本占比

    # 冷却期（基于交易日序号）— 5天冷却期
    cooldown_days: int = 5

    # 权重过渡
    transition_days: int = 3

    # 无候选策略时的默认策略
    default_strategy: str = "dual_ma"

    # 切换成本惩罚系数（在综合评分差中扣除预估成本）
    cost_penalty_factor: float = 1.0


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

    # 新增：唯一决策ID，用于精确匹配事后绩效跟踪（优化点3）
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # 新增：过渡权重（旧策略权重, 新策略权重）
    transition_weights: Tuple[float, float] = (1.0, 0.0)

    # 新增：事后绩效跟踪字段（由外部回填）
    actual_sharpe_after: Optional[float] = None
    outcome: Optional[str] = None


class StrategySwitchEngine:
    """
    智能策略切换引擎（v2）。

    综合考虑市场环境转变信号强度、当前策略绩效表现及潜在交易成本，
    决定是否进行策略切换。

    核心变更（v2）:
      - 环境评估：不再依赖 _current_regime，改为比较策略的 suitable_regimes
      - 综合评分：Sharpe + 最大回撤 + 换手率的加权评分，含 min_sharpe_gap
      - 切换成本：支持 switching_ratio 部分切换，空仓免成本
      - 交易日冷却：使用 trading_day_index 而非自然日
      - 降级保护：无候选策略时回落至默认策略
      - 过渡权重：集成到决策流程，新增 get_current_transition_weights

    ⚠️ 非线程安全。多线程使用需外部加锁。
    """

    def __init__(self, library: StrategyLibrary, config: Optional[SwitchConfig] = None):
        self.library = library
        self.config = config or SwitchConfig()
        self._decision_log: List[SwitchDecision] = []
        self._last_switch_index: Optional[int] = None  # 改为交易日序号
        self._last_switch_date: Optional[str] = None  # 保留用于过渡权重计算
        self._current_strategy: Optional[str] = None

    # ----------------------------------------------------------------
    # 公共属性
    # ----------------------------------------------------------------

    @property
    def decision_log(self) -> List[SwitchDecision]:
        return self._decision_log

    # ----------------------------------------------------------------
    # 评分与评估
    # ----------------------------------------------------------------

    @staticmethod
    def _composite_score(profile: StrategyProfile, regime: MarketRegime) -> float:
        """
        计算策略综合评分。

        score = 0.6 * Sharpe + 0.3 * (1 - max_drawdown) - 0.1 * turnover

        缺失数据处理策略：
          - 若 performance_by_regime 为空，使用 profile 默认参数计算评分，
            而非返回 -inf，确保无历史数据的新策略仍可被选中。
          - 若 perf 非空但缺少部分指标，通过 perf.get(key, default) 使用默认值：
            Sharpe→0, max_drawdown→profile.max_drawdown(0.2), avg_turnover→profile.avg_turnover(0.5)。
        """
        perf = profile.get_performance(regime)
        if not perf:
            # 无历史绩效数据时，使用 profile 默认参数计算中性评分
            # 适合该环境的策略获得额外加分
            regime_bonus = 0.1 if regime in profile.suitable_regimes else -0.1
            dd = profile.max_drawdown
            turnover = profile.avg_turnover
            score = 0.6 * 0.0 + 0.3 * (1.0 - dd) - 0.1 * turnover + regime_bonus
            return score

        sharpe = perf.get("sharpe", 0.0)
        # max_drawdown 以小数表示（如 0.2 = 20%），1 - max_drawdown 越大越好
        dd = perf.get("max_drawdown", profile.max_drawdown)
        turnover = perf.get("avg_turnover", profile.avg_turnover)

        score = 0.6 * sharpe + 0.3 * (1.0 - dd) - 0.1 * turnover
        return score

    def evaluate_regime_change(
        self,
        current_strategy_regimes: List[MarketRegime],
        new_regime: MarketRegime,
        confidence: float,
    ) -> Tuple[bool, float]:
        """
        评估环境变化是否触发切换。

        比较当前策略的适用环境与实时传入的 new_regime。
        若当前策略无指定环境，则默认始终触发环境变化。

        Args:
            current_strategy_regimes: 当前策略的 suitable_regimes
            new_regime: 实时市场环境
            confidence: 环境识别置信度

        Returns:
            (是否触发, 信号强度)
        """
        # 如果当前策略已适用于该环境，不触发
        if current_strategy_regimes and new_regime in current_strategy_regimes:
            return False, 0.0

        signal_strength = confidence
        triggered = signal_strength >= self.config.regime_change_threshold
        return triggered, signal_strength

    def evaluate_performance(
        self, current_score: float, candidate_score: float, samples_valid: bool = True
    ) -> Tuple[bool, float]:
        """
        评估绩效差异是否触发切换（基于综合评分）。

        Args:
            current_score: 当前策略综合评分
            candidate_score: 候选策略综合评分
            samples_valid: 绩效样本是否充足

        Returns:
            (是否触发, 评分差)
        """
        if not samples_valid:
            return False, 0.0

        gap = candidate_score - current_score

        # 综合评分差距必须超过 min_sharpe_gap 才触发
        if gap <= self.config.min_sharpe_gap:
            return False, gap

        return True, gap

    # ----------------------------------------------------------------
    # 成本估计
    # ----------------------------------------------------------------

    def estimate_switching_cost(
        self, position_value: float, has_position: bool, switching_ratio: float = 1.0
    ) -> float:
        """
        估算策略切换成本。

        cost = position_value * switching_ratio * (commission_rate + slippage_rate) * 2

        Args:
            position_value: 当前持仓市值
            has_position: 是否有持仓
            switching_ratio: 实际切换仓位比例（0~1），默认1.0全仓切换

        Returns:
            估算成本
        """
        # position_value <= 0 或无效，直接返回0
        if position_value <= 0 or not has_position:
            return 0.0

        switching_ratio = max(0.0, min(1.0, switching_ratio))
        rate = self.config.commission_rate + self.config.slippage_rate
        return position_value * switching_ratio * rate * 2  # 平仓+开仓

    # ----------------------------------------------------------------
    # 冷却期检查（基于交易日序号）
    # ----------------------------------------------------------------

    def check_cooldown(
        self, current_date: str, trading_day_index: Optional[int] = None
    ) -> bool:
        """
        检查是否在冷却期内。

        优先使用 trading_day_index（交易日序号），
        若未传入则回退到自然日计算并打印警告。

        Args:
            current_date: 当前日期（自然日，用于兼容回退）
            trading_day_index: 当前交易日序号（从0开始）

        Returns:
            True 表示在冷却期内
        """
        cfg = self.config

        if trading_day_index is not None and self._last_switch_index is not None:
            # 基于交易日序号
            return (trading_day_index - self._last_switch_index) < cfg.cooldown_days

        if self._last_switch_date is not None:
            # 回退：自然日
            warnings.warn(
                "check_cooldown 使用自然日而非交易日序号，"
                "建议在 decide 中传入 trading_day_index"
            )
            last = pd.Timestamp(self._last_switch_date)
            current = pd.Timestamp(current_date)
            return (current - last).days < cfg.cooldown_days

        return False

    # ----------------------------------------------------------------
    # 权重过渡
    # ----------------------------------------------------------------

    def compute_transition_weights(self, days_since_switch: int) -> Tuple[float, float]:
        """
        计算权重过渡。

        在切换后的 transition_days 天内，逐步从旧策略过渡到新策略。

        Args:
            days_since_switch: 距上次切换的交易日天数

        Returns:
            (旧策略权重, 新策略权重)
        """
        cfg = self.config
        if days_since_switch >= cfg.transition_days:
            return 0.0, 1.0
        if days_since_switch <= 0:
            return 1.0, 0.0

        progress = days_since_switch / cfg.transition_days
        new_weight = progress
        old_weight = 1.0 - progress
        return old_weight, new_weight

    def get_current_transition_weights(
        self, current_date: str, trading_day_index: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        获取当前应使用的旧/新策略权重。

        优先使用 trading_day_index（交易日序号）计算距上次切换的天数，
        若未传入则回退到自然日差值。

        Args:
            current_date: 当前日期（自然日，用于兼容回退）
            trading_day_index: 当前交易日序号（推荐传入，与冷却期一致）

        Returns:
            (旧策略权重, 新策略权重)
        """
        if self._last_switch_date is None:
            return 0.0, 1.0  # 无切换记录，全仓当前策略

        # 优化点1：优先使用交易日序号，与decide中冷却期一致
        if trading_day_index is not None and self._last_switch_index is not None:
            days_since = trading_day_index - self._last_switch_index
            return self.compute_transition_weights(days_since)

        # 回退：自然日
        warnings.warn(
            "get_current_transition_weights 使用自然日而非交易日序号，"
            "建议传入 trading_day_index 保持一致"
        )
        last = pd.Timestamp(self._last_switch_date)
        current = pd.Timestamp(current_date)
        days_since = (current - last).days
        return self.compute_transition_weights(days_since)

    # ----------------------------------------------------------------
    # 核心决策
    # ----------------------------------------------------------------

    def decide(
        self,
        current_date: str,
        current_regime: MarketRegime,
        regime_confidence: float,
        current_sharpe: float,
        position_value: float,
        has_position: bool,
        sharpe_samples: int = 0,
        switching_ratio: float = 1.0,
        trading_day_index: Optional[int] = None,
    ) -> Optional[SwitchDecision]:
        """
        综合决策是否切换策略。

        Args:
            current_date: 当前日期
            current_regime: 当前市场环境
            regime_confidence: 环境识别置信度
            current_sharpe: 当前策略Sharpe
            position_value: 当前持仓市值
            has_position: 是否有持仓
            sharpe_samples: 计算Sharpe所用的样本天数（新增）
            switching_ratio: 实际切换仓位比例 0~1（新增，默认全仓）
            trading_day_index: 当前交易日序号（新增，用于冷却期计算）

        Returns:
            切换决策，若不需要切换则返回 None
        """
        cfg = self.config

        # ---- 冷却期检查 ----
        if self.check_cooldown(current_date, trading_day_index):
            return None

        # ---- 获取当前策略信息 ----
        current_profile = (
            self.library.get_profile(self._current_strategy)
            if self._current_strategy
            else None
        )

        current_suitable_regimes = (
            current_profile.suitable_regimes if current_profile else []
        )

        # ---- 环境变化评估 ----
        # 比较当前策略的适用环境与实时传入的 current_regime
        if current_suitable_regimes:
            regime_triggered, signal_strength = self.evaluate_regime_change(
                current_suitable_regimes, current_regime, regime_confidence
            )
        else:
            # 当前策略无指定环境，默认环境变化始终触发
            regime_triggered = True
            signal_strength = 1.0

        # ---- 获取候选策略 ----
        candidates = self.library.get_strategies_for_regime(current_regime)

        # ---- 降级处理：无候选策略时，回落至默认策略 ----
        if not candidates:
            default_name = cfg.default_strategy
            if self._current_strategy != default_name:
                default_profile = self.library.get_profile(default_name)
                if default_profile is None:
                    return None
                decision = SwitchDecision(
                    timestamp=current_date,
                    from_strategy=self._current_strategy or "none",
                    to_strategy=default_name,
                    reason=SwitchReason.MANUAL,
                    regime=current_regime.value,
                    confidence=regime_confidence,
                    from_sharpe=current_sharpe,
                    to_sharpe=default_profile.get_sharpe(current_regime),
                    expected_improvement=0.0,
                    switching_cost=0.0,
                    approved=True,
                    details={
                        "signal_strength": signal_strength,
                        "perf_triggered": False,
                        "cost_acceptable": True,
                        "regime_triggered": True,
                        "fallback": "no_candidates_for_regime",
                    },
                    transition_weights=(1.0, 0.0),  # 初始过渡权重
                )
                self._commit_decision(decision, trading_day_index, current_date)
                return decision
            return None

        # ---- 选择最佳候选（基于综合评分） ----
        best_candidate = None
        best_score = float("-inf")
        for candidate in candidates:
            score = self._composite_score(candidate, current_regime)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            return None

        # 如果当前策略就是最佳候选，不需要切换
        if self._current_strategy == best_candidate.name:
            return None

        # ---- 绩效触发评估（基于综合评分差 + 样本有效性） ----
        samples_valid = sharpe_samples >= cfg.min_samples

        # 计算当前策略的综合评分
        if current_profile is not None:
            current_score = self._composite_score(current_profile, current_regime)
        else:
            current_score = 0.0

        perf_triggered, score_gap = self.evaluate_performance(
            current_score, best_score, samples_valid
        )

        # 如果环境未触发且绩效也未触发，不切换
        if not regime_triggered and not perf_triggered:
            return None

        # ---- 切换到最佳候选（确保至少有环境触发） ----
        # 注意：即使绩效不触发，环境触发仍可驱动切换
        if not regime_triggered:
            # 仅有绩效触发时才可能抵达此处（已被上方return过滤）
            pass

        # ---- 切换成本评估 ----
        switching_cost = self.estimate_switching_cost(
            position_value, has_position, switching_ratio
        )
        # 当 position_value <= 0 时，成本为0且总是可接受
        max_cost = (
            position_value * cfg.max_switching_cost_pct if position_value > 0 else 0.0
        )
        cost_acceptable = switching_cost <= max_cost if position_value > 0 else True

        # ---- 成本惩罚：从评分差中扣除预估切换成本 ----
        # 将成本转化为Sharpe等价惩罚
        if position_value > 0 and has_position:
            cost_penalty = (switching_cost / position_value) * cfg.cost_penalty_factor
        else:
            cost_penalty = 0.0
        adjusted_score_gap = score_gap - cost_penalty

        # ---- 综合决策 ----
        # 成本惩罚后评分差仍需超过 min_sharpe_gap
        should_switch = (
            (regime_triggered or perf_triggered)
            and cost_acceptable
            and adjusted_score_gap > cfg.min_sharpe_gap
        )

        # 预期改善：使用综合评分差（若有当前策略），否则退化为 Sharpe 差
        if current_profile is not None:
            expected_improvement = score_gap
        else:
            expected_improvement = (
                best_candidate.get_sharpe(current_regime) - current_sharpe
            )

        decision = SwitchDecision(
            timestamp=current_date,
            from_strategy=self._current_strategy or "none",
            to_strategy=best_candidate.name,
            reason=(
                SwitchReason.REGIME_CHANGE
                if regime_triggered
                else SwitchReason.PERFORMANCE_DEGRADATION
            ),
            regime=current_regime.value,
            confidence=regime_confidence,
            from_sharpe=current_sharpe,
            to_sharpe=best_candidate.get_sharpe(current_regime),
            expected_improvement=expected_improvement,
            switching_cost=switching_cost,
            approved=should_switch,
            details={
                "signal_strength": signal_strength,
                "perf_triggered": perf_triggered,
                "cost_acceptable": cost_acceptable,
                "regime_triggered": regime_triggered,
                "samples_valid": samples_valid,
                "current_score": current_score if current_profile else None,
                "candidate_score": best_score,
                "score_gap": score_gap,
            },
            transition_weights=(1.0, 0.0),  # 初始过渡权重
        )

        if should_switch:
            self._commit_decision(decision, trading_day_index, current_date)

        return decision

    def _commit_decision(
        self,
        decision: SwitchDecision,
        trading_day_index: Optional[int],
        current_date: str,
    ):
        """记录切换决策并更新内部状态。"""
        self._decision_log.append(decision)
        self._current_strategy = decision.to_strategy
        if trading_day_index is not None:
            self._last_switch_index = trading_day_index
        self._last_switch_date = current_date

    # ----------------------------------------------------------------
    # 事后绩效跟踪
    # ----------------------------------------------------------------

    def update_decision_outcome(
        self,
        decision_timestamp: str,
        actual_sharpe: float,
        outcome: str,
        decision_id: Optional[str] = None,
    ):
        """
        回填切换决策的事后绩效。

        外部在切换后 N 天调用，更新对应决策的 actual_sharpe_after 和 outcome。

        Args:
            decision_timestamp: 决策的时间戳（与 SwitchDecision.timestamp 匹配）
            actual_sharpe: 事后实际 Sharpe
            outcome: 结果标签（如 "improved", "degraded", "no_change"）
            decision_id: 决策唯一ID（推荐使用，避免时间戳重复时匹配错误）
        """
        # 优先按 decision_id 精确匹配（优化点3：防止同时间戳多个决策匹配错误）
        if decision_id is not None:
            for d in self._decision_log:
                if d.decision_id == decision_id:
                    d.actual_sharpe_after = actual_sharpe
                    d.outcome = outcome
                    return
            return  # decision_id 未找到，静默返回

        # 回退：按时间戳匹配
        matched = [d for d in self._decision_log if d.timestamp == decision_timestamp]
        if len(matched) > 1:
            warnings.warn(
                f"时间戳 {decision_timestamp} 匹配到 {len(matched)} 条决策，"
                "仅更新第一条。建议使用 decision_id 参数精确匹配。"
            )
        if matched:
            matched[0].actual_sharpe_after = actual_sharpe
            matched[0].outcome = outcome

    # ----------------------------------------------------------------
    # 兼容接口
    # ----------------------------------------------------------------

    def get_current_strategy(self) -> Optional[str]:
        """获取当前活跃策略。"""
        return self._current_strategy

    def set_initial_strategy(self, strategy_name: str, regime: MarketRegime):
        """
        设置初始策略。

        注意：_current_regime 已移除，仅记录策略名。
        """
        self._current_strategy = strategy_name

    def get_decision_summary(self) -> pd.DataFrame:
        """获取决策日志摘要。"""
        if not self._decision_log:
            return pd.DataFrame()
        rows = []
        for d in self._decision_log:
            rows.append(
                {
                    "日期": d.timestamp,
                    "从策略": d.from_strategy,
                    "到策略": d.to_strategy,
                    "原因": d.reason.value,
                    "环境": d.regime,
                    "置信度": d.confidence,
                    "预期改善": d.expected_improvement,
                    "切换成本": d.switching_cost,
                    "已执行": d.approved,
                    "事后Sharpe": d.actual_sharpe_after,
                    "结果": d.outcome,
                }
            )
        return pd.DataFrame(rows)
