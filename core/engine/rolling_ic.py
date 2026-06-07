"""
滚动IC加权引擎（DEPRECATED 兼容层）。

⚠️ P0-4整改（2026-06-07）：
  本文件已合并到 core/factors/factor_evaluator.py 的 FactorEvaluator.compute_ic_weights() 方法。
  保留本文件仅作为向后兼容层：
    - 旧导入路径仍可使用
    - 实际计算全部委托给 FactorEvaluator
    - 旧实例的 update()/get_dynamic_weights() 方法转为薄包装

新代码请直接使用:
    from core.factors.factor_evaluator import FactorEvaluator
    evaluator = FactorEvaluator(...)
    weights = evaluator.compute_ic_weights(...)

位置: core/engine/rolling_ic.py（仅作兼容层）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import DEFAULT_FACTOR_WEIGHTS
from core.factors.factor_evaluator import FactorEvaluator

logger = logging.getLogger(__name__)


@dataclass
class RollingICConfig:
    """滚动IC配置（兼容层）。"""

    window: int = 60
    forward_period: int = 5
    ema_alpha: float = 0.1
    min_abs_ic: float = 0.02
    min_observations: int = 30


class RollingICWeightEngine:
    """
    滚动IC加权引擎（DEPRECATED 兼容层）。

    ⚠️ P0-4整改：核心逻辑已迁移到 FactorEvaluator.compute_ic_weights()。
    本类仅做兼容转发，行为完全等价。

    迁移指南：
        # 旧用法
        engine = RollingICWeightEngine()
        engine.update(scores, ret, symbol)
        weights = engine.get_dynamic_weights()

        # 新用法
        evaluator = FactorEvaluator(
            forward_period=5, ic_window=60, min_observations=30,
        )
        weights = evaluator.compute_ic_weights(
            factor_scores_history={...},
            forward_returns=np.array([...]),
            min_abs_ic=0.02,
            ema_alpha=0.1,
            default_weights=DEFAULT_FACTOR_WEIGHTS,
            prev_weights=None,
        )
    """

    def __init__(self, config: Optional[RollingICConfig] = None):
        self.config = config or RollingICConfig()
        # 委托给 FactorEvaluator
        self._evaluator = FactorEvaluator(
            forward_period=self.config.forward_period,
            ic_window=self.config.window,
            min_observations=self.config.min_observations,
        )
        # 状态保持不变（旧接口兼容）
        self._symbol_score_history: Dict[str, Dict[str, List[float]]] = {}
        self._symbol_return_history: Dict[str, List[float]] = {}
        self._current_ic: Dict[str, float] = {}
        self._current_weights: Dict[str, float] = dict(DEFAULT_FACTOR_WEIGHTS)
        self._observation_count: int = 0
        self._ic_history: List[Dict[str, float]] = []

    @property
    def current_ic(self) -> Dict[str, float]:
        return dict(self._current_ic)

    @property
    def current_weights(self) -> Dict[str, float]:
        return dict(self._current_weights)

    @property
    def ic_history(self) -> pd.DataFrame:
        if not self._ic_history:
            return pd.DataFrame()
        return pd.DataFrame(self._ic_history)

    def update(
        self,
        factor_scores: Dict[str, float],
        forward_return: float,
        symbol: str = "",
    ) -> None:
        """
        更新滚动窗口数据（兼容方法）。

        P0-4整改：内部调用 FactorEvaluator.compute_ic_weights() 计算权重。
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
            if len(sym_scores[name]) > self.config.window:
                sym_scores[name] = sym_scores[name][-self.config.window:]

        self._symbol_return_history[symbol].append(float(forward_return))
        if len(self._symbol_return_history[symbol]) > self.config.window:
            self._symbol_return_history[symbol] = self._symbol_return_history[symbol][-self.config.window:]
        self._observation_count += 1

        if self._observation_count < self.config.min_observations:
            return

        # 合并所有品种的因子历史与收益历史，调用 FactorEvaluator
        merged_scores: Dict[str, List[float]] = {}
        merged_returns: List[float] = []
        for sym, sym_scores in self._symbol_score_history.items():
            sym_ret = self._symbol_return_history.get(sym, [])
            for name, scores in sym_scores.items():
                if name not in merged_scores:
                    merged_scores[name] = []
                n = min(len(scores), len(sym_ret))
                if n > 0:
                    merged_scores[name].extend(scores[-n:])
            if sym_ret:
                merged_returns.extend(sym_ret)

        arr_scores = {
            name: np.array(s, dtype=float) for name, s in merged_scores.items()
        }
        arr_returns = np.array(merged_returns, dtype=float)

        # P0-4整改：完全委托给 FactorEvaluator 计算 IC（含 EMA 平滑）
        new_ic = self._evaluator.compute_ic_per_factor(arr_scores, arr_returns)
        # EMA 平滑 _current_ic（仅保留历史 IC 兼容旧接口）
        if self._current_ic:
            alpha = self.config.ema_alpha
            for name in new_ic:
                prev = self._current_ic.get(name, 0.0)
                new_ic[name] = alpha * new_ic[name] + (1 - alpha) * prev
        self._current_ic = new_ic

        new_weights = self._evaluator.compute_ic_weights(
            factor_scores_history=arr_scores,
            forward_returns=arr_returns,
            min_abs_ic=self.config.min_abs_ic,
            ema_alpha=self.config.ema_alpha,
            default_weights=dict(DEFAULT_FACTOR_WEIGHTS),
            prev_weights=self._current_weights if self._current_weights else None,
        )
        self._current_weights = new_weights
        self._ic_history.append(dict(self._current_ic))
        logger.debug("RollingICWeightEngine: 权重更新 %s", new_weights)

    def get_dynamic_weights(self) -> Dict[str, float]:
        return dict(self._current_weights)

    def compute_forward_returns(self, close_prices: pd.Series) -> pd.Series:
        period = self.config.forward_period
        return close_prices.shift(-period) / close_prices - 1.0

    def reset(self) -> None:
        self._symbol_score_history.clear()
        self._symbol_return_history.clear()
        self._current_ic.clear()
        self._current_weights = dict(DEFAULT_FACTOR_WEIGHTS)
        self._observation_count = 0
        self._ic_history.clear()

    def get_ic_summary(self) -> Dict[str, Dict[str, float]]:
        if not self._ic_history:
            return {}
        df = pd.DataFrame(self._ic_history)
        summary: Dict[str, Dict[str, float]] = {}
        for col in df.columns:
            mean_ic = float(df[col].mean())
            std_ic = float(df[col].std())
            ir = mean_ic / (std_ic if abs(std_ic) > 1e-10 else 1.0)
            summary[col] = {
                "mean": mean_ic,
                "std": std_ic,
                "ir": ir,
                "current": self._current_ic.get(col, 0.0),
                "weight": self._current_weights.get(col, 0.0),
            }
        return summary
