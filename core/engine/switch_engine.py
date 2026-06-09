"""
因子打分调仓引擎（纯信号合成器）。

⚠️ 重要：根据代码审查报告整改（任务3、任务13），本类已简化为**纯信号合成器**：
  - **只输出综合得分**（-1..1），不输出仓位/方向（仓位决策委托给 PortfolioManager）
  - **不做品种轮动**（Top-N 选择委托给 PortfolioManager.select_top_by_signal）
  - **不做横截面资金分配**（委托给 PortfolioManager.allocate_weights）

核心机制（保留）:
  - 5子策略打分：趋势、期限结构、均值回归、波动率突破、复合共振
  - 横截面标准化：Z-Score 标准化，消除品种间因子值分布差异
  - 排名叠加：因子排名等权相加作为综合得分，降低极端值影响
  - 滚动IC动态权重：用因子IC动态调整权重，替代固定权重
  - 调仓日执行：仅在调仓日（每N个交易日）执行调仓

整改记录（参考 /Users/luojiutian/Downloads/代码审查报告.docx）：
  - P0-任务3：删除 score_to_position、is_symbol_selected、top_n_symbols、_current_symbol
  - P0-任务3：删除 _shared_ic_weights 全局变量
  - P0-任务3：移除 update_cross_section 中的自动 finalize（由顶层显式调用）
  - P0-任务3：compute_composite_score 改为显式接收 symbol 参数
  - P2-任务13：删除所有旧接口兼容方法

⚠️ 非线程安全。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

import pandas as pd
import numpy as np

from core.config.strategy_profiles import StrategyLibrary
from core.config import DEFAULT_FACTOR_WEIGHTS

_logger = logging.getLogger(__name__)


class RebalanceReason(Enum):
    """调仓原因。"""

    SCHEDULED = "scheduled"  # 定期调仓
    STOP_LOSS = "stop_loss"  # 止损触发
    SCORE_REVERSAL = "score_reversal"  # 得分反转


@dataclass
class ScoringConfig:
    """因子打分调仓配置（仅信号合成层，不含品种轮动配置）。"""

    # 调仓周期（交易日数）
    rebalance_days: int = 3

    # 因子权重（默认等权，统一从 config.DEFAULT_FACTOR_WEIGHTS 引用）
    factor_weights: Dict[str, float] = field(
        default_factory=DEFAULT_FACTOR_WEIGHTS.copy
    )

    # 横截面标准化开关
    use_cross_section: bool = True

    # 排名叠加开关
    use_rank_score: bool = True

    # 滚动IC动态权重开关
    use_rolling_ic: bool = True

    # 信号→仓位映射参数（保留在信号层，供执行器内联使用，方法已删除）
    entry_threshold: float = 0.05
    score_scale: float = 1.0

    # 趋势过滤开关（兼容旧版执行器）
    use_trend_filter: bool = False


@dataclass
class RebalanceDecision:
    """调仓决策（仅信号层面的决策记录，不再含 direction/position_pct）。"""

    timestamp: str
    symbol: str
    factor_scores: Dict[str, float]  # 各因子得分
    composite_score: float  # 综合得分 [-1, 1]
    reason: RebalanceReason


class FactorScoringEngine:
    """
    因子打分调仓引擎（纯信号合成器）。

    职责严格限定为**子策略信号合成**，不做任何仓位/方向/轮动决策。
    资金分配与品种轮动委托给 PortfolioManager。
    整体风控委托给 RiskController。

    调仓流程（仅信号合成）：
      1. 顶层判断是否为调仓日（外部 is_rebalance_day）
      2. 调用 update_cross_section 收集每个品种的原始因子得分
      3. 所有品种收集完毕后，**由顶层显式调用** finalize_cross_section
      4. 调用 compute_composite_score(symbol) 获取综合得分 [-1, 1]
      5. 顶层将综合得分交给 PortfolioManager 分配权重
    """

    def __init__(
        self, library: StrategyLibrary, config: Optional[ScoringConfig] = None
    ):
        self.library = library
        self.config = config or ScoringConfig()
        self._decision_log: List[RebalanceDecision] = []
        self._last_rebalance_date: Any = None
        # 滚动IC动态权重（由 RollingICEngine 注入，替代原全局变量）
        self._ic_weights: Optional[Dict[str, float]] = None
        # 横截面数据：当前调仓日收集的原始因子得分
        self._cross_section_scores: Dict[str, Dict[str, float]] = {}
        # 上一轮finalize后的标准化得分（供 compute_composite_score 读取）
        self._finalized_scores: Dict[str, Dict[str, float]] = {}
        # 排名叠加结果
        self._rank_scores: Dict[str, float] = {}
        # 激活的子策略集合：None 表示全部 5 个；列表则只取列表内的
        # 修复 register_strategies 失效 bug：runner 把 _registered_strategies
        # 同步过来，extract_factor_scores 据此过滤 indicator_map
        self._active_strategies: Optional[List[str]] = None
        # 当前横截面收集的调仓日期
        self._cross_section_date: Any = None

    def set_active_strategies(self, strategies: Optional[List[str]]) -> None:
        """
        设置激活的子策略集合。

        修复 register_strategies 失效 bug：runner 在 _run_pybroker 入口
        把 self._registered_strategies 同步过来；为 None 时表示全 5 子策略。

        Args:
            strategies: 激活的子策略名列表；None/空表示全部
        """
        self._active_strategies = list(strategies) if strategies else None

    @property
    def decision_log(self) -> List[RebalanceDecision]:
        return self._decision_log

    @staticmethod
    def extract_indicator(ctx: Any, name: str) -> Optional[float]:
        """
        从 PyBroker ExecContext 中安全提取指标值。

        Args:
            ctx: PyBroker ExecContext
            name: 指标名称

        Returns:
            指标值（float），获取失败返回 None
        """
        try:
            raw = ctx.indicator(name)
            if hasattr(raw, "iloc") and len(raw) > 0:
                val = float(raw.iloc[-1])
                if not np.isfinite(val):
                    _logger.debug("指标 %s 值为NaN/Inf", name)
                    return None
                return val
            elif hasattr(raw, "__getitem__") and len(raw) > 0:
                val = float(raw[-1])
                if not np.isfinite(val):
                    _logger.debug("指标 %s 值为NaN/Inf", name)
                    return None
                return val
        except Exception as e:
            _logger.debug("指标 %s 获取失败: %s", name, e)
        return None

    def extract_factor_scores(
        self,
        ctx: Any,
        strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """
        从 PyBroker ExecContext 中提取5子策略因子得分。

        每个子策略对应一组PyBroker指标，提取后归一化到 [-1, 1]。

        5子策略指标映射（来自 StrategyIndicatorRegistry）：
          - trend: trend_signal（趋势策略信号）
          - term_structure: term_structure_signal（期限结构策略信号）
          - mean_reversion: mean_reversion_signal（均值回归策略信号）
          - vol_breakout: vol_breakout_signal（波动率突破策略信号）
          - composite_resonance: composite_signal（复合共振策略信号）

        Args:
            ctx: PyBroker ExecContext
            strategy_params: 各策略参数（预留扩展，暂未使用）

        Returns:
            因子得分字典 {子策略名: 得分}
        """
        factor_scores: Dict[str, float] = {}

        # 从 StrategyIndicatorRegistry 获取指标名→因子名映射
        try:
            from core.engine.strategy_indicators import StrategyIndicatorRegistry

            indicator_map = StrategyIndicatorRegistry.get_indicator_to_factor_map()
        except ImportError:
            # 模块加载失败时使用默认映射（仅作为兜底，不掩盖错误）
            _logger.warning("StrategyIndicatorRegistry 未加载，使用默认指标映射")
            indicator_map = {
                "trend_signal": "trend",
                "term_structure_signal": "term_structure",
                "mean_reversion_signal": "mean_reversion",
                "vol_breakout_signal": "vol_breakout",
                "composite_signal": "composite_resonance",
            }

        for indicator_name, factor_name in indicator_map.items():
            # 修复 register_strategies 失效 bug：按 _active_strategies 过滤
            # 未注册子策略的指标被跳过，executor 只看注册过的信号
            # 双保险：strategy_params 存在时也用来推导 active set
            active = self._active_strategies
            if active is None and strategy_params:
                active = list(strategy_params.keys())
            if active is not None and factor_name not in active:
                continue
            val = self.extract_indicator(ctx, indicator_name)
            if val is not None and np.isfinite(val):
                # tanh压缩确保在 [-1, 1] 区间
                factor_scores[factor_name] = float(np.clip(val, -1.0, 1.0))

        return factor_scores

    def is_rebalance_day(self, dt: Any) -> bool:
        """
        判断给定日期是否为调仓日（仅日期模式）。

        Args:
            dt: 当前日期（datetime/date/Timestamp）

        Returns:
            是否为调仓日
        """
        if dt is None:
            return False

        from datetime import date, datetime

        date_key = dt.date() if hasattr(dt, "date") else dt

        if self._last_rebalance_date is None:
            self._last_rebalance_date = date_key
            return True

        if isinstance(date_key, (date, datetime)):
            days_diff = (date_key - self._last_rebalance_date).days
        else:
            days_diff = (
                pd.Timestamp(date_key) - pd.Timestamp(self._last_rebalance_date)
            ).days

        return days_diff >= self.config.rebalance_days

    def mark_rebalanced(self, dt: Any) -> None:
        """
        标记当前日期已完成调仓（更新 _last_rebalance_date）。

        Args:
            dt: 当前日期
        """
        if dt is None:
            return
        from datetime import date, datetime

        date_key = dt.date() if hasattr(dt, "date") else dt
        if isinstance(date_key, (date, datetime, pd.Timestamp)):
            self._last_rebalance_date = date_key

    def compute_composite_score(
        self,
        symbol: str,
        factor_scores: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        计算指定品种的综合得分（-1..1）。

        数据来源优先级：
        1. 排名叠加结果（_rank_scores）—— 当 use_rank_score=True 时
        2. 横截面标准化结果（_finalized_scores[symbol]）—— 调仓周期内已finalize后
        3. 入参 factor_scores —— 单品种/未finalize场景

        Args:
            symbol: 品种代码
            factor_scores: 备选因子得分（仅当无 finalized/rank 数据时使用）

        Returns:
            综合得分，范围 [-1, 1]
        """
        # 1. 排名叠加得分优先
        if self.config.use_rank_score and self._rank_scores:
            return float(np.clip(self._rank_scores.get(symbol, 0.0), -1.0, 1.0))

        # 2. 横截面标准化后得分次优
        if self._finalized_scores and symbol in self._finalized_scores:
            factor_scores = self._finalized_scores[symbol]
        elif factor_scores is None:
            return 0.0

        # 3. 加权合成
        weights = self._get_effective_weights()
        total_score = 0.0
        total_weight = 0.0

        for factor_name, score in factor_scores.items():
            w = weights.get(factor_name, 0.0)
            if not np.isfinite(score):
                score = 0.0
            total_score += w * score
            total_weight += w

        if total_weight > 0:
            composite = total_score / total_weight
        else:
            composite = 0.0

        return float(np.clip(composite, -1.0, 1.0))

    def _get_effective_weights(self) -> Dict[str, float]:
        """获取有效权重：优先使用滚动IC动态权重，否则使用配置权重。"""
        if self.config.use_rolling_ic and self._ic_weights:
            return self._ic_weights
        return self.config.factor_weights

    def update_cross_section(
        self,
        symbol: str,
        factor_scores: Dict[str, float],
        dt: Any = None,
    ) -> None:
        """
        收集横截面因子得分（调仓日每个品种调用一次）。

        **重要整改**：移除自动 finalize 逻辑。
        finalize_cross_section 必须由顶层（run_backtest.py 或 BacktestRunner）
        在所有品种数据收集完毕后**显式调用**，避免数据不完整时触发 finalize。

        Args:
            symbol: 品种代码
            factor_scores: 该品种的各因子得分
            dt: 当前日期
        """
        if dt is not None and self._cross_section_date is None:
            self._cross_section_date = dt

        if factor_scores:
            self._cross_section_scores[symbol] = dict(factor_scores)
        elif symbol not in self._cross_section_scores:
            self._cross_section_scores[symbol] = {}

    def finalize_cross_section(self) -> None:
        """
        横截面数据处理：标准化 + 排名叠加。

        **重要整改**：此方法**不再**在 update_cross_section 中自动调用。
        必须由顶层在所有品种的 update_cross_section 调用完毕后显式调用。

        P2-1 整改（2026-06-07）：
          Z-Score 标准化改用 FactorEvaluator.cross_sectional_standardize，
          避免重复造轮（规则17）。

        Returns:
            None
        """
        if not self._cross_section_scores:
            _logger.debug("finalize: _cross_section_scores为空")
            return

        # 过滤掉没有因子数据的品种
        valid_scores = {k: v for k, v in self._cross_section_scores.items() if v}
        if len(valid_scores) < 1:
            _logger.debug("finalize: valid_scores为0")
            return

        scores_df = pd.DataFrame(valid_scores).T
        factor_names = list(scores_df.columns)

        # P2-1 整改：横截面Z-Score标准化 — 委托给 FactorEvaluator
        if self.config.use_cross_section and len(scores_df) > 1:
            from core.factors.factor_evaluator import FactorEvaluator

            _evaluator = FactorEvaluator()
            for col in factor_names:
                col_scores = scores_df[col].to_dict()
                standardized = _evaluator.cross_sectional_standardize(col_scores)
                if standardized:
                    scores_df[col] = pd.Series(standardized)
                else:
                    scores_df[col] = 0.0

        # 排名叠加（加权版：高IC因子权重更大）
        if self.config.use_rank_score and len(scores_df) > 1:
            rank_df = scores_df.rank(pct=True) * 2 - 1
            weights = self._get_effective_weights()
            weighted_rank = pd.Series(0.0, index=rank_df.index)
            total_w = 0.0
            for col in factor_names:
                w = weights.get(col, 0.0)
                if w > 0:
                    weighted_rank += w * rank_df[col]
                    total_w += w
            if total_w > 0:
                weighted_rank = weighted_rank / total_w
            self._rank_scores = weighted_rank.to_dict()
        else:
            self._rank_scores = {}

        # 保存标准化后的得分到 _finalized_scores
        self._finalized_scores = scores_df.to_dict(orient="index")

        _logger.debug(
            "finalize: %d 个品种, %d 个因子, rank_scores非空=%s",
            len(scores_df),
            len(factor_names),
            bool(self._rank_scores),
        )

    def set_ic_weights(self, weights: Dict[str, float]) -> None:
        """设置滚动IC动态权重（替代原全局变量 _shared_ic_weights）。"""
        self._ic_weights = dict(weights)

    def clear_cross_section(self) -> None:
        """清除横截面数据（每次调仓周期结束后调用）。"""
        self._cross_section_scores.clear()
        self._rank_scores.clear()
        self._finalized_scores.clear()
        self._cross_section_date = None

    def get_decision_summary(self) -> pd.DataFrame:
        """获取调仓决策汇总。"""
        if not self._decision_log:
            return pd.DataFrame()

        rows = []
        for d in self._decision_log:
            rows.append(
                {
                    "日期": d.timestamp,
                    "品种": d.symbol,
                    "综合得分": round(d.composite_score, 4),
                    "原因": d.reason.value,
                    **{f"因子_{k}": round(v, 4) for k, v in d.factor_scores.items()},
                }
            )
        return pd.DataFrame(rows)

    def record_decision(
        self,
        timestamp: str,
        symbol: str,
        factor_scores: Dict[str, float],
        composite_score: float,
        reason: RebalanceReason = RebalanceReason.SCHEDULED,
    ) -> None:
        """记录一次调仓信号（不再含 direction/position_pct）。"""
        decision = RebalanceDecision(
            timestamp=timestamp,
            symbol=symbol,
            factor_scores=dict(factor_scores),
            composite_score=composite_score,
            reason=reason,
        )
        self._decision_log.append(decision)
