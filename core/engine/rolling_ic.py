"""
滚动IC加权引擎。

用滚动信息系数（IC）替代固定权重：
  1. 对每个因子，计算因子得分与前瞻收益的滚动相关系数（IC）
  2. 用 IC 动态调整因子权重：weight_i = |IC_i| / Σ|IC_j|
  3. EMA 平滑避免突变
  4. IC 数据不足时回退到固定权重

位置: core/engine/rolling_ic.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np

from core.config import DEFAULT_FACTOR_WEIGHTS

logger = logging.getLogger(__name__)


@dataclass
class RollingICConfig:
    """滚动IC配置。"""

    # 滚动窗口（交易日数）
    window: int = 60

    # 前瞻收益周期（交易日数）
    forward_period: int = 5

    # IC权重EMA平滑系数（0=不更新, 1=直接用最新IC）
    ema_alpha: float = 0.1

    # IC 最低绝对值阈值（低于此值的因子权重清零）
    min_abs_ic: float = 0.02

    # 最少观测数（不足时回退固定权重）
    min_observations: int = 30


class RollingICWeightEngine:
    """
    滚动IC加权引擎。

    用因子得分与前瞻收益的相关系数作为权重依据，
    替代固定权重配置。

    使用方式:
        engine = RollingICWeightEngine()
        engine.update(factor_scores, forward_returns)
        weights = engine.get_dynamic_weights()
    """

    def __init__(self, config: Optional[RollingICConfig] = None):
        self.config = config or RollingICConfig()
        # 按品种独立追踪：{symbol: {factor_name: [scores]}}
        self._symbol_score_history: Dict[str, Dict[str, List[float]]] = {}
        # 按品种独立追踪收益：{symbol: [returns]}
        self._symbol_return_history: Dict[str, List[float]] = {}
        self._current_ic: Dict[str, float] = {}
        self._current_weights: Dict[str, float] = dict(DEFAULT_FACTOR_WEIGHTS)
        self._observation_count: int = 0
        self._ic_history: List[Dict[str, float]] = []

    @property
    def current_ic(self) -> Dict[str, float]:
        """当前各因子IC值。"""
        return dict(self._current_ic)

    @property
    def current_weights(self) -> Dict[str, float]:
        """当前动态权重。"""
        return dict(self._current_weights)

    @property
    def ic_history(self) -> pd.DataFrame:
        """IC历史时间序列。"""
        if not self._ic_history:
            return pd.DataFrame()
        return pd.DataFrame(self._ic_history)

    def update(self, factor_scores: Dict[str, float], forward_return: float, symbol: str = ""):
        """
        更新滚动窗口数据。

        Args:
            factor_scores: 当前各因子得分
            forward_return: 对应前瞻收益
            symbol: 品种代码（按品种独立追踪）
        """
        if symbol not in self._symbol_score_history:
            self._symbol_score_history[symbol] = {}
        if symbol not in self._symbol_return_history:
            self._symbol_return_history[symbol] = []

        sym_scores = self._symbol_score_history[symbol]
        for name, score in factor_scores.items():
            if name not in sym_scores:
                sym_scores[name] = []
            sym_scores[name].append(float(score))

        self._symbol_return_history[symbol].append(float(forward_return))
        self._observation_count += 1

        # 滚动窗口限制
        window = self.config.window
        for sym in self._symbol_score_history:
            sym_sc = self._symbol_score_history[sym]
            for name in sym_sc:
                if len(sym_sc[name]) > window:
                    sym_sc[name] = sym_sc[name][-window:]
            sym_ret = self._symbol_return_history[sym]
            if len(sym_ret) > window:
                self._symbol_return_history[sym] = sym_ret[-window:]

        # 数据足够时重新计算IC
        if self._observation_count >= self.config.min_observations:
            self._recompute_ic()

    def _recompute_ic(self):
        """重新计算各因子滚动IC并更新权重（合并所有品种数据）。"""
        # 收集所有因子名称
        all_factor_names: set = set()
        for sym_scores in self._symbol_score_history.values():
            all_factor_names.update(sym_scores.keys())

        new_ic: Dict[str, float] = {}

        for name in all_factor_names:
            # 合并所有品种的该因子得分和对应收益
            all_scores: List[float] = []
            all_returns: List[float] = []
            for sym, sym_scores in self._symbol_score_history.items():
                if name not in sym_scores:
                    continue
                sym_ret = self._symbol_return_history.get(sym, [])
                scores = sym_scores[name]
                # 对齐：取该品种因子得分和收益的较短长度
                min_len = min(len(scores), len(sym_ret))
                if min_len < 2:
                    continue
                all_scores.extend(scores[-min_len:])
                all_returns.extend(sym_ret[-min_len:])

            if len(all_scores) < self.config.min_observations:
                continue

            arr = np.array(all_scores)
            ret = np.array(all_returns)
            if np.std(arr) < 1e-10 or np.std(ret) < 1e-10:
                new_ic[name] = 0.0
                continue
            corr = np.corrcoef(arr, ret)[0, 1]
            new_ic[name] = 0.0 if np.isnan(corr) else float(corr)

        # EMA 平滑IC
        if self._current_ic:
            alpha = self.config.ema_alpha
            for name in new_ic:
                prev = self._current_ic.get(name, 0.0)
                new_ic[name] = alpha * new_ic[name] + (1 - alpha) * prev

        self._current_ic = new_ic
        self._ic_history.append(dict(new_ic))

        # 更新动态权重
        self._update_weights_from_ic()

    def _update_weights_from_ic(self):
        """根据IC绝对值计算动态权重。"""
        min_ic = self.config.min_abs_ic
        abs_ic = {name: max(abs(ic), 0.0) for name, ic in self._current_ic.items()}

        # 低于阈值的因子权重清零
        for name in list(abs_ic.keys()):
            if abs_ic[name] < min_ic:
                abs_ic[name] = 0.0

        total_ic = sum(abs_ic.values())
        if total_ic > 0:
            weights = {name: v / total_ic for name, v in abs_ic.items()}
        else:
            # 全部清零时回退到默认等权
            weights = dict(DEFAULT_FACTOR_WEIGHTS)

        # EMA平滑权重
        if self._current_weights:
            alpha = self.config.ema_alpha
            smoothed: Dict[str, float] = {}
            all_factors = set(weights.keys()) | set(self._current_weights.keys())
            for name in all_factors:
                prev = self._current_weights.get(name, 0.0)
                new_w = weights.get(name, 0.0)
                smoothed[name] = alpha * new_w + (1 - alpha) * prev
            weights = smoothed

        self._current_weights = weights

    def get_dynamic_weights(self) -> Dict[str, float]:
        """
        获取当前动态权重。

        Returns:
            {因子名: 权重}，所有权重之和为 1.0
        """
        return dict(self._current_weights)

    def compute_forward_returns(self, close_prices: pd.Series) -> pd.Series:
        """
        计算前瞻收益率。

        Args:
            close_prices: 收盘价序列（按时间排序）

        Returns:
            前瞻收益率序列
        """
        period = self.config.forward_period
        fwd_ret = close_prices.shift(-period) / close_prices - 1.0
        return fwd_ret

    def reset(self):
        """重置所有状态。"""
        self._symbol_score_history.clear()
        self._symbol_return_history.clear()
        self._current_ic.clear()
        self._current_weights = dict(DEFAULT_FACTOR_WEIGHTS)
        self._observation_count = 0
        self._ic_history.clear()

    def get_ic_summary(self) -> Dict[str, Dict[str, float]]:
        """获取IC统计摘要。
        
        Returns:
            {因子名: {mean, std, ir, current}} 字典
        """
        if not self._ic_history:
            return {}

        df = pd.DataFrame(self._ic_history)
        summary: Dict[str, Dict[str, float]] = {}
        
        for col in df.columns:
            mean_ic = float(df[col].mean())
            std_ic = float(df[col].std())
            ir = mean_ic / (std_ic if abs(std_ic) > 1e-10 else 1.0)
            current_ic = self._current_ic.get(col, 0.0)
            current_weight = self._current_weights.get(col, 0.0)
            
            summary[col] = {
                "mean": mean_ic,
                "std": std_ic,
                "ir": ir,
                "current": current_ic,
                "weight": current_weight,
            }
        
        return summary