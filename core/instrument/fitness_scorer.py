"""
适应度评分器。

基于历史回测表现对品种进行适应度评分：
  - Sharpe比率
  - 最大回撤
  - 胜率
  - 盈亏比
  - Calmar比率

规则14要求：品种适应度评分基于历史回测表现。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FitnessResult:
    """适应度评分结果。"""

    symbol: str = ""
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    calmar: float = 0.0
    fitness_score: float = 0.0

    def summary(self) -> str:
        """返回评分摘要。"""
        return (
            f"[{self.symbol}] Fitness={self.fitness_score:.2f} "
            f"Sharpe={self.sharpe:.2f} MDD={self.max_drawdown:.1%} "
            f"WR={self.win_rate:.1%} PLR={self.profit_loss_ratio:.2f}"
        )


class FitnessScorer:
    """
    品种适应度评分器。

    基于历史回测表现对品种进行综合评分。

    用法:
        scorer = FitnessScorer()
        result = scorer.score(
            symbol="rb2401",
            returns=daily_returns,
        )
    """

    def __init__(
        self,
        sharpe_weight: float = 0.3,
        drawdown_weight: float = 0.25,
        win_rate_weight: float = 0.2,
        plr_weight: float = 0.15,
        calmar_weight: float = 0.1,
        min_trades: int = 10,
    ):
        """
        初始化适应度评分器。

        Args:
            sharpe_weight: Sharpe权重
            drawdown_weight: 最大回撤权重
            win_rate_weight: 胜率权重
            plr_weight: 盈亏比权重
            calmar_weight: Calmar权重
            min_trades: 最少交易次数
        """
        self.sharpe_weight = sharpe_weight
        self.drawdown_weight = drawdown_weight
        self.win_rate_weight = win_rate_weight
        self.plr_weight = plr_weight
        self.calmar_weight = calmar_weight
        self.min_trades = min_trades

    def _compute_sharpe(self, returns: np.ndarray) -> float:
        """计算年化Sharpe。"""
        if len(returns) < 2:
            return 0.0
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        if std_r < 1e-10:
            return 0.0
        return float(mean_r / std_r * np.sqrt(252))

    def _compute_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """计算最大回撤。"""
        if len(equity_curve) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return float(-np.min(drawdown))

    def _compute_win_rate(self, returns: np.ndarray) -> float:
        """计算胜率。"""
        if len(returns) == 0:
            return 0.0
        return float(np.mean(returns > 0))

    def _compute_plr(self, returns: np.ndarray) -> float:
        """计算盈亏比。"""
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        if avg_loss < 1e-10:
            return 10.0
        return float(avg_win / avg_loss)

    def _compute_calmar(self, returns: np.ndarray, max_dd: float) -> float:
        """计算Calmar比率。"""
        if max_dd < 1e-6:
            return 0.0
        annual_return = np.mean(returns) * 252
        return float(annual_return / max_dd)

    def score(
        self,
        symbol: str,
        returns: np.ndarray,
    ) -> FitnessResult:
        """
        计算品种适应度评分。

        Args:
            symbol: 品种代码
            returns: 日收益率序列

        Returns:
            FitnessResult 评分结果
        """
        ret = np.asarray(returns, dtype=float)

        if len(ret) < self.min_trades:
            return FitnessResult(symbol=symbol, fitness_score=0.0)

        sharpe = self._compute_sharpe(ret)

        # 构建净值曲线
        equity = np.cumprod(1 + ret)
        max_dd = self._compute_max_drawdown(equity)

        win_rate = self._compute_win_rate(ret)
        plr = self._compute_plr(ret)
        calmar = self._compute_calmar(ret, max_dd)

        # 各维度归一化评分（0~1）
        sharpe_score = min(max(sharpe / 2.0, 0.0), 1.0)
        dd_score = max(1.0 - max_dd / 0.3, 0.0)
        wr_score = min(win_rate * 1.5, 1.0)
        plr_score = min(plr / 3.0, 1.0)
        calmar_score = min(max(calmar / 2.0, 0.0), 1.0)

        fitness = (
            self.sharpe_weight * sharpe_score
            + self.drawdown_weight * dd_score
            + self.win_rate_weight * wr_score
            + self.plr_weight * plr_score
            + self.calmar_weight * calmar_score
        )

        return FitnessResult(
            symbol=symbol,
            sharpe=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_loss_ratio=plr,
            calmar=calmar,
            fitness_score=fitness,
        )

    def rank(
        self,
        results: List[FitnessResult],
        top_n: Optional[int] = None,
    ) -> List[FitnessResult]:
        """
        按适应度评分排名。

        Args:
            results: 评分结果列表
            top_n: 取前N名，None表示全部

        Returns:
            排名后的列表
        """
        ranked = sorted(results, key=lambda r: r.fitness_score, reverse=True)
        if top_n is not None:
            return ranked[:top_n]
        return ranked
