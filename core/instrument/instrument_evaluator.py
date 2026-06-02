"""
品种评估器。

评估品种的交易适合度，基于以下维度：
  - 流动性：日均成交量、日均成交额
  - 波动率：历史波动率、ATR
  - 趋势性：ADX指标
  - 成本：手续费、滑点占比

规则14要求：品种选择基于流动性+趋势性+成本三维度评分。
"""

from dataclasses import dataclass
from typing import Dict, Optional
import logging

import numpy as np

from utils.indicators import compute_adx as _compute_adx

logger = logging.getLogger(__name__)


@dataclass
class InstrumentEvalResult:
    """品种评估结果。"""

    symbol: str = ""
    # 流动性评分（0~1）
    liquidity_score: float = 0.0
    # 波动率评分（0~1）
    volatility_score: float = 0.0
    # 趋势性评分（0~1）
    trend_score: float = 0.0
    # 成本评分（0~1）
    cost_score: float = 0.0
    # 综合评分（0~1）
    total_score: float = 0.0
    # 原始指标
    avg_volume: float = 0.0
    avg_turnover: float = 0.0
    hv_20d: float = 0.0
    adx: float = 0.0

    def summary(self) -> str:
        """返回评估摘要。"""
        return (
            f"[{self.symbol}] 综合={self.total_score:.2f} "
            f"流动性={self.liquidity_score:.2f} 波动率={self.volatility_score:.2f} "
            f"趋势性={self.trend_score:.2f} 成本={self.cost_score:.2f}"
        )


class InstrumentEvaluator:
    """
    品种评估器。

    基于流动性、波动率、趋势性和成本四个维度评估品种。

    用法:
        evaluator = InstrumentEvaluator()
        result = evaluator.evaluate(
            symbol="rb2401",
            close=close_prices,
            volume=volume_data,
            high=high_prices,
            low=low_prices,
        )
    """

    def __init__(
        self,
        volume_min: float = 10000,
        turnover_min: float = 1e8,
        hv_ideal_range: tuple = (0.10, 0.30),
        adx_trend_threshold: float = 25.0,
        cost_max_pct: float = 0.002,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        初始化品种评估器。

        Args:
            volume_min: 最低日均成交量
            turnover_min: 最低日均成交额
            hv_ideal_range: 理想波动率范围（年化）
            adx_trend_threshold: ADX趋势阈值
            cost_max_pct: 最大成本占比
            weights: 各维度权重 {liquidity, volatility, trend, cost}
        """
        self.volume_min = volume_min
        self.turnover_min = turnover_min
        self.hv_ideal_range = hv_ideal_range
        self.adx_trend_threshold = adx_trend_threshold
        self.cost_max_pct = cost_max_pct
        self.weights = weights or {
            "liquidity": 0.3,
            "volatility": 0.2,
            "trend": 0.3,
            "cost": 0.2,
        }

    def _score_liquity(self, avg_volume: float, avg_turnover: float) -> float:
        """流动性评分。"""
        vol_score = (
            min(avg_volume / self.volume_min, 1.0) if self.volume_min > 0 else 0.0
        )
        turn_score = (
            min(avg_turnover / self.turnover_min, 1.0) if self.turnover_min > 0 else 0.0
        )
        return (vol_score + turn_score) / 2.0

    def _score_volatility(self, hv: float) -> float:
        """波动率评分：在理想范围内得分最高。"""
        low, high = self.hv_ideal_range
        if low <= hv <= high:
            return 1.0
        elif hv < low:
            return max(0.0, hv / low)
        else:
            return max(0.0, high / hv)

    def _score_trend(self, adx: float) -> float:
        """趋势性评分：ADX越高趋势越明显。"""
        if adx >= self.adx_trend_threshold:
            return min(1.0, adx / 50.0)
        else:
            return adx / self.adx_trend_threshold * 0.5

    def _score_cost(self, cost_pct: float) -> float:
        """成本评分：成本越低越好。"""
        if cost_pct <= 0:
            return 1.0
        return max(0.0, 1.0 - cost_pct / self.cost_max_pct)

    def evaluate(
        self,
        symbol: str,
        close: np.ndarray,
        volume: Optional[np.ndarray] = None,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        cost_pct: float = 0.001,
    ) -> InstrumentEvalResult:
        """
        评估品种。

        Args:
            symbol: 品种代码
            close: 收盘价序列
            volume: 成交量序列
            high: 最高价序列
            low: 最低价序列
            cost_pct: 交易成本占比

        Returns:
            InstrumentEvalResult 评估结果
        """
        c = np.asarray(close, dtype=float)

        # 流动性
        avg_volume = 0.0
        avg_turnover = 0.0
        if volume is not None:
            v = np.asarray(volume, dtype=float)
            avg_volume = float(np.mean(v[-20:])) if len(v) >= 20 else float(np.mean(v))
            avg_turnover = (
                avg_volume * float(np.mean(c[-20:]))
                if len(c) >= 20
                else avg_volume * float(np.mean(c))
            )

        liquidity_score = self._score_liquity(avg_volume, avg_turnover)

        # 波动率
        hv = 0.0
        if len(c) > 21:
            returns = np.diff(c[-21:]) / c[-21:-1]
            hv = float(np.std(returns) * np.sqrt(252))

        volatility_score = self._score_volatility(hv)

        # 趋势性（委托公共ADX函数）
        adx = 0.0
        if high is not None and low is not None:
            adx, _, _ = _compute_adx(high, low, close, period=14)

        trend_score = self._score_trend(adx)

        # 成本
        cost_score = self._score_cost(cost_pct)

        # 综合评分
        total = (
            self.weights["liquidity"] * liquidity_score
            + self.weights["volatility"] * volatility_score
            + self.weights["trend"] * trend_score
            + self.weights["cost"] * cost_score
        )

        return InstrumentEvalResult(
            symbol=symbol,
            liquidity_score=liquidity_score,
            volatility_score=volatility_score,
            trend_score=trend_score,
            cost_score=cost_score,
            total_score=total,
            avg_volume=avg_volume,
            avg_turnover=avg_turnover,
            hv_20d=hv,
            adx=adx,
        )
