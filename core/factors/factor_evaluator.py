"""
因子评估框架。

对因子进行全面评估，输出 IC、IR、多周期稳定性等指标。
IC（信息系数）：因子得分与前瞻收益的相关系数
IR（信息比率）：IC均值 / IC标准差，衡量因子预测稳定性
多周期稳定性：1M/3M/6M/12M IC衰减曲线

规则9要求：IC > 0.03 且 IR > 0.5 的因子方可保留。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 评估周期映射（交易日数）
EVAL_PERIODS: Dict[str, int] = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "12M": 252,
}

# 因子保留阈值（规则9）
IC_THRESHOLD = 0.03
IR_THRESHOLD = 0.5
CORRELATION_REDUNDANCY = 0.7


@dataclass
class FactorEvalResult:
    """单因子评估结果。"""

    name: str
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ir: float = 0.0
    ic_by_period: Dict[str, float] = field(default_factory=dict)
    ic_decay_rate: float = 0.0
    is_valid: bool = False
    reject_reason: str = ""

    def summary(self) -> str:
        """返回评估摘要字符串。"""
        status = "✅ 有效" if self.is_valid else f"❌ {self.reject_reason}"
        periods_str = " | ".join(
            f"{k}: {v:.4f}" for k, v in self.ic_by_period.items()
        )
        return (
            f"[{self.name}] IC={self.ic_mean:.4f} IR={self.ir:.4f} "
            f"衰减={self.ic_decay_rate:.2%} {status} | {periods_str}"
        )


class FactorEvaluator:
    """
    因子评估器。

    对因子得分序列与前瞻收益进行相关性分析，
    输出 IC、IR、多周期稳定性等评估指标。

    用法:
        evaluator = FactorEvaluator(forward_period=5)
        result = evaluator.evaluate(
            factor_name="ts_momentum",
            factor_scores=scores_array,
            forward_returns=returns_array,
        )
    """

    def __init__(
        self,
        forward_period: int = 5,
        ic_window: int = 60,
        min_observations: int = 30,
    ):
        """
        初始化因子评估器。

        Args:
            forward_period: 前瞻收益周期（交易日）
            ic_window: 滚动IC计算窗口（交易日）
            min_observations: 最少观测数
        """
        self.forward_period = forward_period
        self.ic_window = ic_window
        self.min_observations = min_observations

    def evaluate(
        self,
        factor_name: str,
        factor_scores: np.ndarray,
        forward_returns: np.ndarray,
        dates: Optional[np.ndarray] = None,
    ) -> FactorEvalResult:
        """
        评估单个因子。

        Args:
            factor_name: 因子名称
            factor_scores: 因子得分序列
            forward_returns: 前瞻收益序列
            dates: 日期序列（可选，用于多周期分析）

        Returns:
            FactorEvalResult 评估结果
        """
        scores = np.asarray(factor_scores, dtype=float)
        returns = np.asarray(forward_returns, dtype=float)

        # 对齐长度
        min_len = min(len(scores), len(returns))
        if min_len < self.min_observations:
            return FactorEvalResult(
                name=factor_name,
                is_valid=False,
                reject_reason=f"观测数不足({min_len}<{self.min_observations})",
            )

        scores = scores[:min_len]
        returns = returns[:min_len]

        # 去除NaN
        valid_mask = ~(np.isnan(scores) | np.isnan(returns))
        scores_clean = scores[valid_mask]
        returns_clean = returns[valid_mask]

        if len(scores_clean) < self.min_observations:
            return FactorEvalResult(
                name=factor_name,
                is_valid=False,
                reject_reason=f"有效观测数不足({len(scores_clean)})",
            )

        # 计算整体IC
        ic_mean, ic_std, ir = self._compute_ic_stats(scores_clean, returns_clean)

        # 多周期IC
        ic_by_period = self._compute_multi_period_ic(
            scores_clean, returns_clean, dates, min_len
        )

        # IC衰减率（1M IC vs 12M IC 的衰减幅度）
        ic_1m = ic_by_period.get("1M", ic_mean)
        ic_12m = ic_by_period.get("12M", 0.0)
        ic_decay_rate = 0.0
        if abs(ic_1m) > 1e-6:
            ic_decay_rate = max(0.0, 1.0 - abs(ic_12m) / abs(ic_1m))

        # 判定有效性（规则9）
        is_valid = True
        reject_reason = ""
        if abs(ic_mean) < IC_THRESHOLD:
            is_valid = False
            reject_reason = f"IC={ic_mean:.4f}<{IC_THRESHOLD}"
        elif abs(ir) < IR_THRESHOLD:
            is_valid = False
            reject_reason = f"IR={ir:.4f}<{IR_THRESHOLD}"

        return FactorEvalResult(
            name=factor_name,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ir,
            ic_by_period=ic_by_period,
            ic_decay_rate=ic_decay_rate,
            is_valid=is_valid,
            reject_reason=reject_reason,
        )

    def evaluate_batch(
        self,
        factor_scores_dict: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
        dates: Optional[np.ndarray] = None,
    ) -> Dict[str, FactorEvalResult]:
        """
        批量评估多个因子。

        Args:
            factor_scores_dict: {因子名: 得分序列}
            forward_returns: 前瞻收益序列
            dates: 日期序列

        Returns:
            {因子名: FactorEvalResult}
        """
        results: Dict[str, FactorEvalResult] = {}
        for name, scores in factor_scores_dict.items():
            results[name] = self.evaluate(name, scores, forward_returns, dates)
            logger.info(results[name].summary())
        return results

    def compute_correlation_matrix(
        self,
        factor_scores_dict: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        计算因子间相关系数矩阵。

        用于冗余检测（规则9：相关系数 > 0.7 视为冗余）。

        Args:
            factor_scores_dict: {因子名: 得分序列}

        Returns:
            相关系数矩阵 DataFrame
        """
        df = pd.DataFrame(factor_scores_dict)
        return df.corr()

    def _compute_ic_stats(
        self,
        scores: np.ndarray,
        returns: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        计算IC统计量：均值、标准差、IR。

        使用滚动窗口计算IC序列，再取统计量。

        Returns:
            (ic_mean, ic_std, ir)
        """
        window = self.ic_window
        n = len(scores)

        if n < window:
            # 数据不足一个窗口，直接计算整体相关系数
            if np.std(scores) < 1e-10 or np.std(returns) < 1e-10:
                return 0.0, 0.0, 0.0
            ic = float(np.corrcoef(scores, returns)[0, 1])
            if np.isnan(ic):
                return 0.0, 0.0, 0.0
            return ic, 0.0, 0.0 if abs(ic) < 1e-10 else 1.0

        # 滚动IC序列
        ic_series: List[float] = []
        for i in range(window, n + 1):
            s = scores[i - window : i]
            r = returns[i - window : i]
            if np.std(s) < 1e-10 or np.std(r) < 1e-10:
                ic_series.append(0.0)
                continue
            corr = float(np.corrcoef(s, r)[0, 1])
            ic_series.append(0.0 if np.isnan(corr) else corr)

        ic_arr = np.array(ic_series)
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr))
        ir = ic_mean / ic_std if abs(ic_std) > 1e-10 else 0.0

        return ic_mean, ic_std, ir

    def _compute_multi_period_ic(
        self,
        scores: np.ndarray,
        returns: np.ndarray,
        dates: Optional[np.ndarray],
        total_len: int,
    ) -> Dict[str, float]:
        """
        计算多周期IC（1M/3M/6M/12M）。

        每个周期取最近N个交易日的数据计算IC。
        """
        ic_by_period: Dict[str, float] = {}

        for period_name, period_days in EVAL_PERIODS.items():
            if total_len < period_days:
                ic_by_period[period_name] = 0.0
                continue

            # 取最近period_days的数据
            s = scores[-period_days:]
            r = returns[-period_days:]

            if np.std(s) < 1e-10 or np.std(r) < 1e-10:
                ic_by_period[period_name] = 0.0
                continue

            corr = float(np.corrcoef(s, r)[0, 1])
            ic_by_period[period_name] = 0.0 if np.isnan(corr) else corr

        return ic_by_period
