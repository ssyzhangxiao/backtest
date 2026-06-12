"""
因子评估框架。

对因子进行全面评估，输出 IC、IR、多周期稳定性等指标。
IC（信息系数）：因子得分与前瞻收益的相关系数
IR（信息比率）：IC均值 / IC标准差，衡量因子预测稳定性
多周期稳定性：1M/3M/6M/12M IC衰减曲线

规则9要求：IC > 0.03 且 IR > 0.5 的因子方可保留。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
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
        periods_str = " | ".join(f"{k}: {v:.4f}" for k, v in self.ic_by_period.items())
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
            factor_name="trend",
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
            s_valid = (
                scores[~np.isnan(scores)]
                if np.issubdtype(scores.dtype, np.floating)
                else scores
            )
            r_valid = (
                returns[~np.isnan(returns)]
                if np.issubdtype(returns.dtype, np.floating)
                else returns
            )
            if len(s_valid) < 2 or len(r_valid) < 2:
                return 0.0, 0.0, 0.0
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if np.std(s_valid) < 1e-10 or np.std(r_valid) < 1e-10:
                    return 0.0, 0.0, 0.0
            with np.errstate(invalid="ignore"):
                try:
                    ic = float(np.corrcoef(s_valid, r_valid)[0, 1])
                except Exception:
                    ic = float("nan")
            if np.isnan(ic):
                return 0.0, 0.0, 0.0
            return ic, 0.0, 0.0 if abs(ic) < 1e-10 else 1.0

        # 滚动IC序列
        ic_series: List[float] = []
        for i in range(window, n + 1):
            s = scores[i - window : i]
            r = returns[i - window : i]
            # 2026-06-12：先做 NaN 过滤，避免 np.std 触发 RuntimeWarning 风暴
            s_valid = s[~np.isnan(s)] if np.issubdtype(s.dtype, np.floating) else s
            r_valid = r[~np.isnan(r)] if np.issubdtype(r.dtype, np.floating) else r
            if len(s_valid) < 2 or len(r_valid) < 2:
                ic_series.append(0.0)
                continue
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                s_std = float(np.std(s_valid))
                r_std = float(np.std(r_valid))
            if (
                not np.isfinite(s_std)
                or not np.isfinite(r_std)
                or s_std < 1e-10
                or r_std < 1e-10
            ):
                ic_series.append(0.0)
                continue
            with np.errstate(invalid="ignore"):
                try:
                    corr = float(np.corrcoef(s_valid, r_valid)[0, 1])
                except Exception:
                    corr = float("nan")
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

    # ────────────────────────────────────────────────────────────
    # P0-4整改：合并 rolling_ic.py 的功能
    # ────────────────────────────────────────────────────────────
    def compute_ic_per_factor(
        self,
        factor_scores_history: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        逐因子计算 IC（带符号的 Pearson 相关系数）。

        P0-4整改：暴露给 rolling_ic.py 使用，替代其内部重复 IC 计算代码。
        数据不足或零方差时返回 0.0。

        Args:
            factor_scores_history: {因子名: 因子得分时间序列}
            forward_returns: 前瞻收益时间序列

        Returns:
            {因子名: IC值（带符号）}
        """
        ic: Dict[str, float] = {}
        for name, scores in factor_scores_history.items():
            n = min(len(scores), len(forward_returns))
            if n < 2:
                continue
            arr = np.asarray(scores[-n:], dtype=float)
            ret = np.asarray(forward_returns[-n:], dtype=float)
            if np.std(arr) < 1e-10 or np.std(ret) < 1e-10:
                ic[name] = 0.0
                continue
            corr = float(np.corrcoef(arr, ret)[0, 1])
            ic[name] = 0.0 if np.isnan(corr) else float(corr)
        return ic

    def compute_ic_weights(
        self,
        factor_scores_history: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
        min_abs_ic: float = 0.02,
        ema_alpha: float = 0.1,
        default_weights: Optional[Dict[str, float]] = None,
        prev_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        基于滚动IC动态计算因子权重。

        P0-4整改：从 rolling_ic.py 合并而来，作为 FactorEvaluator 的方法。
        公式：weight_i = |IC_i| / Σ|IC_j|，EMA平滑避免突变。

        Args:
            factor_scores_history: {因子名: 因子得分时间序列}
            forward_returns: 前瞻收益时间序列
            min_abs_ic: 最低绝对IC阈值（低于此值清零）
            ema_alpha: EMA平滑系数
            default_weights: 全部清零时回退的默认权重
            prev_weights: 上次权重（用于EMA平滑）

        Returns:
            {因子名: 动态权重}，所有权重之和为 1.0
        """
        if not factor_scores_history or len(forward_returns) == 0:
            return dict(default_weights or {})

        # 1) 计算每个因子的整体IC
        abs_ic: Dict[str, float] = {}
        for name, scores in factor_scores_history.items():
            if len(scores) < self.min_observations:
                continue
            # 对齐长度
            n = min(len(scores), len(forward_returns))
            if n < 2:
                continue
            arr = np.asarray(scores[-n:], dtype=float)
            ret = np.asarray(forward_returns[-n:], dtype=float)
            if np.std(arr) < 1e-10 or np.std(ret) < 1e-10:
                abs_ic[name] = 0.0
                continue
            corr = float(np.corrcoef(arr, ret)[0, 1])
            abs_ic[name] = 0.0 if np.isnan(corr) else abs(corr)

        # 2) 低于阈值的清零
        for name in list(abs_ic.keys()):
            if abs_ic[name] < min_abs_ic:
                abs_ic[name] = 0.0

        # 3) 归一化权重
        total = sum(abs_ic.values())
        if total > 0:
            weights = {name: v / total for name, v in abs_ic.items()}
        else:
            weights = dict(default_weights or {})

        # 4) EMA平滑权重
        if prev_weights:
            smoothed: Dict[str, float] = {}
            all_names = set(weights.keys()) | set(prev_weights.keys())
            for name in all_names:
                prev = prev_weights.get(name, 0.0)
                new = weights.get(name, 0.0)
                smoothed[name] = ema_alpha * new + (1 - ema_alpha) * prev
            weights = smoothed

        return weights

    def cross_sectional_standardize(
        self,
        scores_by_symbol: Dict[str, float],
    ) -> Dict[str, float]:
        """
        横截面 Z-Score 标准化。

        P2-1整改：从 FactorScoringEngine.finalize_cross_section 提取，
        统一到 FactorEvaluator。

        Args:
            scores_by_symbol: {品种代码: 因子得分}

        Returns:
            {品种代码: 标准化后的得分}（均值0、方差1）
        """
        if not scores_by_symbol:
            return {}
        values = np.array(list(scores_by_symbol.values()), dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values))
        if abs(std) < 1e-10:
            # 常数列：原样返回
            return dict(scores_by_symbol)
        return {sym: float((v - mean) / std) for sym, v in scores_by_symbol.items()}

    def cross_sectional_rank(
        self,
        scores_by_symbol: Dict[str, float],
    ) -> Dict[str, float]:
        """
        横截面排名叠加。

        P2-1整改：从 FactorScoringEngine.finalize_cross_section 提取，
        统一到 FactorEvaluator。

        Args:
            scores_by_symbol: {品种代码: 因子得分}

        Returns:
            {品种代码: 排名叠加得分}（按绝对值排名后归一化到 [-1, 1]）
        """
        if not scores_by_symbol:
            return {}
        # 按 |score| 降序排序
        sorted_items = sorted(
            scores_by_symbol.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        n = len(sorted_items)
        if n == 0:
            return {}
        # 排名叠加：最大正向=1, 最大负向=-1
        result: Dict[str, float] = {}
        for rank, (sym, score) in enumerate(sorted_items, start=1):
            # 排名分位 ∈ (0, 1]
            percentile = (n - rank + 1) / n
            # 保留方向
            sign = 1.0 if score >= 0 else -1.0
            result[sym] = sign * percentile
        return result

    # ────────────────────────────────────────────────────────────
    # P1-2整改：DataFrame 级横截面方法（多品种宽表）
    # ────────────────────────────────────────────────────────────
    def standardize_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        对多品种宽表做横截面 Z-Score 标准化。

        行为：按行（每个交易日）做 Z-Score 标准化，
        即 df.iloc[i, j] = (df.iloc[i, j] - mean_i) / std_i。
        常数列（std≈0）置 0。

        Args:
            df: 宽表（index=日期, columns=品种/因子）

        Returns:
            与原表形状相同的标准化 DataFrame
        """
        if df is None or df.empty:
            return df.copy() if df is not None else pd.DataFrame()
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        safe_std = std.where(std > 1e-10, 1.0)
        standardized = df.sub(mean, axis=0).div(safe_std, axis=0)
        # std ≈ 0 的行填 0
        small = std <= 1e-10
        if small.any():
            standardized.loc[small] = 0.0
        return standardized

    def rank_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        对每行做横截面排名并映射到 [-1, 1]。

        公式：rank_dataframe.iloc[i, j] = sign(value) * percentile，
        其中 percentile ∈ (0, 1]。

        Args:
            df: 宽表（index=日期, columns=品种/因子）

        Returns:
            与原表形状相同的排名 DataFrame
        """
        if df is None or df.empty:
            return df.copy() if df is not None else pd.DataFrame()
        n = df.shape[1]
        if n <= 1:
            return df.copy()

        # 按 |value| 排名（行内），分数越高排名越大
        abs_vals = df.abs()
        # ascending=False → 大值排前
        ranks_desc = abs_vals.rank(axis=1, ascending=False, method="average")
        # 排名分位 ∈ (0, 1]
        percentile = (n + 1 - ranks_desc) / n
        # 保留方向
        sign = np.sign(df)
        return (sign * percentile).clip(lower=-1.0, upper=1.0)

    # ────────────────────────────────────────────────────────────
    # P1-3整改：合并 factor_decay.py 的 detect_decay 能力
    # ────────────────────────────────────────────────────────────
    def detect_decay(
        self,
        ic_history: Dict[str, List[float]],
        trend_window: int = 40,
        ic_healthy_threshold: float = 0.03,
        ic_dead_threshold: float = 0.01,
        max_consecutive_decline: int = 5,
        decay_slope_threshold: float = -0.001,
    ) -> Dict[str, Dict[str, Any]]:
        """
        检测因子IC的衰减状态。

        P1-3整改：合并自 core/engine/factor_decay.py 的核心逻辑。
        规则3要求原模块被废弃后，调用方迁移到本方法。

        Args:
            ic_history: {因子名: IC值历史序列}
            trend_window: 趋势检测窗口
            ic_healthy_threshold: IC 健康阈值（绝对值）
            ic_dead_threshold: IC 死区阈值（绝对值）
            max_consecutive_decline: 连续下降次数告警阈值
            decay_slope_threshold: 衰减斜率阈值（负斜率绝对值）

        Returns:
            {因子名: {
                "status": "healthy" | "warning" | "decaying" | "dead",
                "current_ic": float,
                "trend_slope": float,
                "consecutive_decline": int,
            }}
        """
        result: Dict[str, Dict[str, Any]] = {}
        for name, ic_series in ic_history.items():
            if len(ic_series) < trend_window:
                continue
            recent = ic_series[-trend_window:]
            current_ic = float(recent[-1])
            abs_ic = abs(current_ic)

            # 1. 状态判断
            if abs_ic < ic_dead_threshold:
                status = "dead"
            elif abs_ic < ic_healthy_threshold:
                if len(recent) >= 10:
                    x = np.arange(len(recent))
                    slope = float(np.polyfit(x, recent, 1)[0])
                else:
                    slope = 0.0
                if slope < decay_slope_threshold:
                    status = "decaying"
                else:
                    status = "warning"
            else:
                status = "healthy"

            # 2. 连续下降
            consecutive = 0
            for i in range(len(recent) - 1, 0, -1):
                if recent[i] < recent[i - 1]:
                    consecutive += 1
                else:
                    break
            if consecutive >= max_consecutive_decline and status == "healthy":
                status = "warning"

            # 3. 趋势斜率
            if len(recent) >= 10:
                x = np.arange(len(recent))
                trend_slope = float(np.polyfit(x, recent, 1)[0])
            else:
                trend_slope = 0.0

            result[name] = {
                "status": status,
                "current_ic": round(current_ic, 6),
                "trend_slope": round(trend_slope, 6),
                "consecutive_decline": consecutive,
            }
        return result
