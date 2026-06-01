"""
因子打分调仓引擎。

在调仓日对每个品种计算多因子综合得分，根据得分排名决定持仓方向和仓位。
替代原有的市场环境→策略切换机制，实现纯因子打分调仓。

核心机制:
  - 多因子打分：4个因子各自输出 [-1, 1] 区间的信号
  - 横截面标准化：Z-Score 标准化，消除品种间因子值分布差异
  - 排名叠加：因子排名等权相加作为综合得分，降低极端值影响
  - 滚动IC动态权重：用因子IC动态调整权重，替代固定权重
  - 品种轮动：只持有综合得分最高的前N个品种
  - 调仓日执行：仅在调仓日（每N个交易日）执行调仓
  - 仓位分配：综合得分决定方向，得分绝对值决定仓位比例
  - 冷却期：两次调仓之间至少间隔 N 个交易日
  - 因子提取：从 PyBroker ExecContext 中提取各因子原始值并归一化

⚠️ 非线程安全。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import uuid
import logging

import pandas as pd
import numpy as np

from core.strategy_registry import StrategyLibrary, StrategyProfile
from core.config import DEFAULT_FACTOR_WEIGHTS

_logger = logging.getLogger(__name__)

# 共享滚动IC权重字典，用于跨实例通信
# PyBroker会复制闭包，导致scoring_engine和IC引擎实例不同，
# 此模块级字典作为共享状态，绕开实例隔离问题
_shared_ic_weights: Dict[str, float] = {}


class RebalanceReason(Enum):
    """调仓原因。"""

    SCHEDULED = "scheduled"           # 定期调仓
    STOP_LOSS = "stop_loss"           # 止损触发
    SCORE_REVERSAL = "score_reversal" # 得分反转


@dataclass
class ScoringConfig:
    """因子打分调仓配置。"""

    # 调仓周期（交易日数）
    rebalance_days: int = 3

    # 因子权重（默认等权，统一从 config.DEFAULT_FACTOR_WEIGHTS 引用）
    factor_weights: Dict[str, float] = field(default_factory=DEFAULT_FACTOR_WEIGHTS.copy)

    # 开仓阈值：综合得分绝对值需超过此值才开仓
    entry_threshold: float = 0.05

    # 仓位缩放：得分绝对值映射到仓位比例
    # score_scale=1.0 表示满仓对应得分=1.0
    score_scale: float = 1.0

    # 止损后冷却期（交易日）
    stop_loss_cooldown: int = 1

    # 交易成本
    commission_rate: float = 0.0001
    slippage_rate: float = 0.0001

    # 横截面标准化开关
    use_cross_section: bool = True

    # 排名叠加开关
    use_rank_score: bool = True

    # 滚动IC动态权重开关
    use_rolling_ic: bool = True

    # 品种轮动：持有综合得分最高的前N个品种
    top_n_symbols: int = 5


@dataclass
class RebalanceDecision:
    """调仓决策。"""

    timestamp: str
    symbol: str
    factor_scores: Dict[str, float]       # 各因子得分
    composite_score: float                 # 综合得分
    direction: int                         # 1=多, -1=空, 0=平
    position_pct: float                    # 建议仓位比例
    reason: RebalanceReason
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class FactorScoringEngine:
    """
    因子打分调仓引擎。

    在调仓日对每个品种计算多因子综合得分，根据得分排名决定持仓方向和仓位。
    不再依赖市场环境判断，而是通过多因子综合得分决定持仓。

    调仓流程：
      1. 判断是否为调仓日
      2. 计算各因子得分
      3. 横截面标准化（Z-Score）
      4. 排名叠加（可选）
      5. 加权合成综合得分（支持滚动IC动态权重）
      6. 品种轮动筛选（只持有Top-N）
      7. 根据得分决定方向和仓位
      8. 应用风控规则（止损冷却等）
    """

    def __init__(self, library: StrategyLibrary, config: Optional[ScoringConfig] = None):
        self.library = library
        self.config = config or ScoringConfig()
        self._decision_log: List[RebalanceDecision] = []
        self._last_rebalance_index: Optional[int] = None
        self._current_strategy: Optional[str] = None
        self._stop_loss_cooldown_until: Optional[int] = None
        self._auto_fusion_mode: bool = True
        self._last_rebalance_date: Any = None
        # 横截面数据：当前调仓日收集的原始因子得分
        self._cross_section_scores: Dict[str, Dict[str, float]] = {}
        # 上一轮finalize后的标准化得分（供调仓使用）
        self._finalized_scores: Dict[str, Dict[str, float]] = {}
        # 排名叠加结果
        self._rank_scores: Dict[str, float] = {}
        # 品种轮动：当前入选品种
        self._selected_symbols: List[str] = []
        # 滚动IC权重引擎
        self._ic_weights: Optional[Dict[str, float]] = None
        # 横截面是否已finalize（用于判断新调仓周期）
        self._cross_section_finalized: bool = False
        # 当前横截面收集的调仓日期（用于检测调仓日变化）
        self._cross_section_date: Any = None

    @property
    def decision_log(self) -> List[RebalanceDecision]:
        return self._decision_log

    @property
    def auto_fusion_mode(self) -> bool:
        return self._auto_fusion_mode

    def get_current_strategy(self) -> Optional[str]:
        """兼容旧接口：返回当前策略名（打分模式下无意义，返回 None）。"""
        return self._current_strategy

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
        从 PyBroker ExecContext 中提取各因子得分。

        将各因子的原始指标值归一化到 [-1, 1] 区间。

        Args:
            ctx: PyBroker ExecContext
            strategy_params: 各策略参数 {策略名: {参数名: 值}}

        Returns:
            因子得分字典 {因子名: 得分}
        """
        factor_scores: Dict[str, float] = {}
        params = strategy_params or {}

        # ts_momentum: N日累计收益率，tanh归一化
        mom_ret_val = self.extract_indicator(ctx, "mom_ret")
        if mom_ret_val is not None and np.isfinite(mom_ret_val):
            factor_scores["ts_momentum"] = float(np.tanh(float(mom_ret_val) * 5))

        # roll_yield: 价差偏离归一化
        roll_yield_ma_val = self.extract_indicator(ctx, "roll_yield_ma")
        close = ctx.close
        current_close = (
            close[-1] if hasattr(close, "__getitem__") and len(close) > 0 else close
        )
        if (roll_yield_ma_val is not None and np.isfinite(roll_yield_ma_val)
                and current_close is not None and roll_yield_ma_val > 0):
            spread_pct = (current_close - roll_yield_ma_val) / roll_yield_ma_val * 100
            factor_scores["roll_yield"] = float(np.tanh(-spread_pct / 3.0))

        # alpha019: 温和压缩，横截面Z-Score做主要归一化
        a019_val = self.extract_indicator(ctx, "alpha019_val")
        if a019_val is not None and np.isfinite(a019_val):
            factor_scores["alpha019"] = float(np.tanh(float(a019_val) * 0.5))

        # alpha032: 温和压缩
        a032_val = self.extract_indicator(ctx, "alpha032_val")
        if a032_val is not None and np.isfinite(a032_val):
            factor_scores["alpha032"] = float(np.tanh(float(a032_val) * 0.1))

        return factor_scores

    def is_rebalance_day(
        self,
        trading_day_index: Optional[int] = None,
        dt: Any = None,
    ) -> bool:
        """
        判断是否为调仓日。

        支持 bar 序号模式和日期模式：
        - bar 序号：trading_day_index % rebalance_days == 1 时为调仓日
        - 日期模式：基于实际日期计算，避免分钟级数据下判断错误

        Args:
            trading_day_index: 当前交易日序号（从1开始），兼容旧调用
            dt: 当前日期（datetime对象），优先使用

        Returns:
            是否为调仓日
        """
        if dt is not None:
            from datetime import date, datetime
            date_key = dt.date() if hasattr(dt, 'date') else dt
            last_key = self._last_rebalance_date
            if last_key is None:
                self._last_rebalance_date = date_key
                return True
            if isinstance(date_key, (date, datetime)):
                days_diff = (date_key - last_key).days
            else:
                days_diff = (pd.Timestamp(date_key) - pd.Timestamp(last_key)).days
            if days_diff >= self.config.rebalance_days:
                # 只在日期变更时更新，同一天内多个品种调用不重复更新
                if date_key != last_key:
                    self._last_rebalance_date = date_key
                return True
            # 同一天内重复调用也返回True（确保所有品种都能收集横截面数据）
            if date_key == last_key and days_diff == 0:
                return True
            return False

        if trading_day_index is None or trading_day_index < 1:
            return False

        return (trading_day_index - 1) % self.config.rebalance_days == 0

    def compute_composite_score(
        self,
        factor_scores: Dict[str, float],
    ) -> float:
        """
        计算多因子综合得分。

        根据配置选择计算模式：
        - 排名叠加模式：因子排名等权相加
        - 加权模式：score = Σ(weight_i × score_i)

        Args:
            factor_scores: {因子名: 得分}，得分范围 [-1, 1]

        Returns:
            综合得分，范围 [-1, 1]
        """
        if self.config.use_rank_score and self._rank_scores:
            return self._rank_scores.get(
                self._current_symbol, 0.0
            )

        # 横截面标准化后得分优先（使用上一轮finalize的结果）
        if (self._finalized_scores
                and self._current_symbol in self._finalized_scores):
            factor_scores = self._finalized_scores[self._current_symbol]

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

        return max(-1.0, min(1.0, composite))

    def _get_effective_weights(self) -> Dict[str, float]:
        """获取有效权重：优先使用滚动IC动态权重，否则使用配置权重。"""
        global _shared_ic_weights
        if self.config.use_rolling_ic and _shared_ic_weights:
            return dict(_shared_ic_weights)
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

        机制：
        - 同一调仓日，各品种依次调用，收集因子得分
        - 新调仓日到来时，先finalize上一轮数据，再开始新收集

        Args:
            symbol: 品种代码
            factor_scores: 该品种的各因子得分
            dt: 当前日期，用于检测调仓日变化
        """
        # 检测调仓日变化
        if dt is not None and self._cross_section_date is not None and dt != self._cross_section_date:
            # 新调仓日到来：先finalize上一轮数据
            if self._cross_section_scores and not self._cross_section_finalized:
                self.finalize_cross_section()
            # 清除当前收集数据，开始新周期
            # 注意：不清除 _finalized_scores 和 _rank_scores，保留上一轮结果供当前调仓日使用
            self._cross_section_scores.clear()
            self._cross_section_finalized = False
            self._cross_section_date = dt

        if dt is not None and self._cross_section_date is None:
            self._cross_section_date = dt

        if factor_scores:
            self._cross_section_scores[symbol] = dict(factor_scores)
        elif symbol not in self._cross_section_scores:
            self._cross_section_scores[symbol] = {}
        self._current_symbol = symbol

    def finalize_cross_section(self) -> None:
        """
        横截面数据处理：标准化 + 排名叠加 + 品种轮动。

        在所有品种的因子得分收集完毕后调用。
        """
        if not self._cross_section_scores:
            _logger.debug("finalize: _cross_section_scores为空")
            return

        _logger.info("finalize: 收集到 %d 个品种: %s", len(self._cross_section_scores), 
                     {k: len(v) for k, v in self._cross_section_scores.items()})

        # 过滤掉没有因子数据的品种
        valid_scores = {k: v for k, v in self._cross_section_scores.items() if v}
        if len(valid_scores) < 1:
            _logger.debug("finalize: valid_scores为0, scores=%s", list(self._cross_section_scores.keys()))
            self._cross_section_finalized = True
            return

        scores_df = pd.DataFrame(valid_scores).T
        factor_names = list(scores_df.columns)

        # 横截面Z-Score标准化
        if self.config.use_cross_section and len(scores_df) > 1:
            for col in factor_names:
                col_mean = scores_df[col].mean()
                col_std = scores_df[col].std()
                if col_std > 1e-8:
                    scores_df[col] = (scores_df[col] - col_mean) / col_std
                else:
                    scores_df[col] = 0.0

        # 排名叠加
        if self.config.use_rank_score and len(scores_df) > 1:
            rank_df = scores_df.rank(pct=True) * 2 - 1
            self._rank_scores = rank_df.mean(axis=1).to_dict()
        else:
            self._rank_scores = {}

        # 品种轮动：选择综合得分最高的Top-N
        if self.config.top_n_symbols > 0 and len(scores_df) > self.config.top_n_symbols:
            weights = self._get_effective_weights()
            composite = {}
            for sym in scores_df.index:
                total = 0.0
                for col in factor_names:
                    w = weights.get(col, 0.0)
                    total += w * scores_df.loc[sym, col]
                composite[sym] = total

            sorted_syms = sorted(composite.items(), key=lambda x: abs(x[1]), reverse=True)
            self._selected_symbols = [s[0] for s in sorted_syms[:self.config.top_n_symbols]]
        else:
            self._selected_symbols = list(scores_df.index)

        # 保存标准化后的得分到 _finalized_scores（供调仓使用）
        self._finalized_scores = scores_df.to_dict(orient="index")
        self._cross_section_finalized = True

    def is_symbol_selected(self, symbol: str) -> bool:
        """判断品种是否入选当前调仓周期的持仓池。"""
        if not self._selected_symbols:
            return True
        return symbol in self._selected_symbols

    def set_ic_weights(self, weights: Dict[str, float]) -> None:
        """设置滚动IC动态权重。"""
        self._ic_weights = dict(weights)

    def clear_cross_section(self) -> None:
        """清除横截面数据（每次调仓周期结束后调用）。"""
        self._cross_section_scores.clear()
        self._rank_scores.clear()

    def score_to_position(self, composite_score: float) -> Tuple[int, float]:
        """
        将综合得分映射为方向和仓位比例。

        Args:
            composite_score: 综合得分 [-1, 1]

        Returns:
            (direction, position_pct)
            direction: 1=多, -1=空, 0=平仓
            position_pct: 仓位比例 [0, 1]
        """
        threshold = self.config.entry_threshold

        if abs(composite_score) < threshold:
            return 0, 0.0

        direction = 1 if composite_score > 0 else -1
        position_pct = min(abs(composite_score) / self.config.score_scale, 1.0)

        return direction, position_pct

    def decide(
        self,
        current_date: str,
        current_regime=None,
        regime_confidence: float = 0.5,
        current_sharpe: float = 0.0,
        position_value: float = 0.0,
        has_position: bool = False,
        sharpe_samples: int = 0,
        switching_ratio: float = 1.0,
        trading_day_index: Optional[int] = None,
        factor_scores: Optional[Dict[str, float]] = None,
        symbol: str = "",
    ) -> Optional[RebalanceDecision]:
        """
        调仓决策。

        兼容旧接口签名，新增 factor_scores 和 symbol 参数。

        Args:
            current_date: 当前日期
            factor_scores: 各因子得分 {因子名: 得分}
            symbol: 品种代码
            trading_day_index: 交易日序号
            其余参数兼容旧接口，不再使用

        Returns:
            调仓决策，非调仓日返回 None
        """
        if not self.is_rebalance_day(trading_day_index):
            return None

        if factor_scores is None:
            factor_scores = {}

        composite = self.compute_composite_score(factor_scores)
        direction, position_pct = self.score_to_position(composite)

        if self._stop_loss_cooldown_until is not None and trading_day_index is not None:
            if trading_day_index < self._stop_loss_cooldown_until:
                direction = 0
                position_pct = 0.0

        decision = RebalanceDecision(
            timestamp=current_date,
            symbol=symbol,
            factor_scores=dict(factor_scores),
            composite_score=composite,
            direction=direction,
            position_pct=position_pct,
            reason=RebalanceReason.SCHEDULED,
        )

        self._decision_log.append(decision)
        self._last_rebalance_index = trading_day_index
        return decision

    def notify_stop_loss(self, trading_day_index: Optional[int] = None):
        """
        通知止损事件，启动冷却期。

        Args:
            trading_day_index: 当前交易日序号
        """
        if trading_day_index is not None:
            self._stop_loss_cooldown_until = (
                trading_day_index + self.config.stop_loss_cooldown
            )

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
                    "方向": "多" if d.direction == 1 else ("空" if d.direction == -1 else "平"),
                    "仓位比例": round(d.position_pct, 4),
                    "原因": d.reason.value,
                    **{f"因子_{k}": round(v, 4) for k, v in d.factor_scores.items()},
                }
            )
        return pd.DataFrame(rows)

    # ── 旧接口兼容方法（不再有实际功能） ──

    def evaluate_regime_change(self, *args, **kwargs) -> Tuple[bool, float]:
        return False, 0.0

    def evaluate_performance(self, *args, **kwargs) -> Tuple[bool, float]:
        return False, 0.0

    def estimate_switching_cost(self, *args, **kwargs) -> float:
        return 0.0

    def check_cooldown(self, *args, **kwargs) -> bool:
        return False

    def compute_transition_weights(self, *args, **kwargs) -> Tuple[float, float]:
        return 0.0, 1.0

    def get_current_transition_weights(self, *args, **kwargs) -> Tuple[float, float]:
        return 0.0, 1.0


# ── 兼容别名 ──
StrategySwitchEngine = FactorScoringEngine
