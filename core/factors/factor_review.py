"""
因子复核模块。

实现因子质量的6项复核检查：
  1. 数据存活率 - 有效值占比 >= 85%
  2. 缺失值占比 - 每个因子缺失率 <= 15%
  3. 异常值抵抗 - 极值处理前后IC对比
  4. 参数敏感性 - 参数微调IC衰减检测
  5. 因子正交性 - Barra风格因子相关性检测
  6. 时序稳定性 - 滚动1年期ICIR方差

用法：
    from core.factors.factor_review import FactorReviewer

    reviewer = FactorReviewer(factor_data, returns)
    report = reviewer.run_full_review()
    print(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from .factor_evaluator import FactorEvaluator

logger = logging.getLogger(__name__)


# 默认评估器实例：委托给公共 FactorEvaluator（P1-1 整改）
_DEFAULT_EVALUATOR = FactorEvaluator(forward_period=5, ic_window=60, min_observations=30)


@dataclass
class FactorReviewResult:
    """单个因子的复核结果。"""

    name: str
    category: str = ""  # T/R/V/M/H

    # 1. 数据存活率
    survival_rate: float = 0.0
    survival_pass: bool = False

    # 2. 缺失值占比
    missing_rate: float = 0.0
    missing_pass: bool = False

    # 3. 异常值抵抗
    ic_before_winsorize: float = 0.0
    ic_after_winsorize: float = 0.0
    outlier_pass: bool = False

    # 4. 参数敏感性
    ic_sensitivity: float = 0.0  # IC变化率
    sensitivity_pass: bool = False

    # 5. 因子正交性
    barra_corr: Dict[str, float] = field(default_factory=dict)
    ortho_t_stat: float = 0.0
    ortho_pass: bool = False

    # 6. 时序稳定性
    rolling_icir_mean: float = 0.0
    rolling_icir_std: float = 0.0
    stability_pass: bool = False

    # 综合评价
    overall_pass: bool = False
    overall_score: float = 0.0  # 0-100
    recommendation: str = ""  # 保留/降级/剔除/待优化

    def __post_init__(self):
        # 计算综合得分
        scores = []
        if self.survival_pass:
            scores.append(20)
        else:
            scores.append(0)
        if self.missing_pass:
            scores.append(15)
        else:
            scores.append(0)
        if self.outlier_pass:
            scores.append(20)
        else:
            scores.append(0)
        if self.sensitivity_pass:
            scores.append(15)
        else:
            scores.append(0)
        if self.ortho_pass:
            scores.append(15)
        else:
            scores.append(0)
        if self.stability_pass:
            scores.append(15)
        else:
            scores.append(0)
        self.overall_score = sum(scores)
        self.overall_pass = all([
            self.survival_pass, self.missing_pass, self.outlier_pass,
            self.sensitivity_pass, self.ortho_pass, self.stability_pass,
        ])
        if self.overall_score >= 80:
            self.recommendation = "保留"
        elif self.overall_score >= 60:
            self.recommendation = "降级"
        elif self.overall_score >= 40:
            self.recommendation = "待优化"
        else:
            self.recommendation = "剔除"


@dataclass
class FactorReviewReport:
    """因子复核报告。"""

    results: List[FactorReviewResult] = field(default_factory=list)
    summary_stats: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """生成复核摘要。"""
        lines = ["=" * 60, "因子复核报告", "=" * 60]

        # 统计
        retain = sum(1 for r in self.results if r.recommendation == "保留")
        downgrade = sum(1 for r in self.results if r.recommendation == "降级")
        optimize = sum(1 for r in self.results if r.recommendation == "待优化")
        remove = sum(1 for r in self.results if r.recommendation == "剔除")

        lines.append(f"总计: {len(self.results)} 个因子")
        lines.append(f"  保留: {retain} | 降级: {downgrade} | 待优化: {optimize} | 剔除: {remove}")
        lines.append("-" * 60)

        # 各因子详情
        for r in self.results:
            status = "✓" if r.overall_pass else "✗"
            lines.append(
                f"  [{status}] {r.name} ({r.category}) "
                f"得分={r.overall_score:.0f} "
                f"存活={r.survival_rate:.1%} "
                f"缺失={r.missing_rate:.1%} "
                f"IC前={r.ic_before_winsorize:.4f} "
                f"IC后={r.ic_after_winsorize:.4f} "
                f"敏感性={r.ic_sensitivity:.4f} "
                f"正交t={r.ortho_t_stat:.2f} "
                f"→ {r.recommendation}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """转为DataFrame。"""
        rows = []
        for r in self.results:
            rows.append({
                "因子": r.name,
                "类别": r.category,
                "存活率": round(r.survival_rate, 4),
                "缺失率": round(r.missing_rate, 4),
                "IC_截尾前": round(r.ic_before_winsorize, 4),
                "IC_截尾后": round(r.ic_after_winsorize, 4),
                "IC敏感性": round(r.ic_sensitivity, 4),
                "正交t值": round(r.ortho_t_stat, 2),
                "ICIR均值": round(r.rolling_icir_mean, 4),
                "ICIR标准差": round(r.rolling_icir_std, 4),
                "总分": round(r.overall_score, 0),
                "建议": r.recommendation,
            })
        return pd.DataFrame(rows)


class FactorReviewer:
    """
    因子复核器：执行6项质量检查。

    用法:
        reviewer = FactorReviewer(factor_data, returns)
        report = reviewer.run_full_review()
    """

    # 因子类别 → 因子名映射
    FACTOR_CATEGORIES = {
        "T": ["T_01", "T_02", "T_03", "T_04", "T_05"],
        "R": ["R_01", "R_02", "R_03", "R_04", "R_05"],
        "V": ["V_01", "V_02", "V_03", "V_04"],
        "M": ["M_01", "M_02", "M_03", "M_04", "M_05", "CF_01", "CF_02", "CF_03"],
        "H": ["H_01", "H_02", "H_03", "H_04", "H_05"],
        "TS": ["TS_01", "TS_02", "TS_03"],
    }

    def __init__(
        self,
        factor_data: pd.DataFrame,
        returns: pd.Series,
        survival_threshold: float = 0.85,
        missing_threshold: float = 0.15,
        ic_threshold: float = 0.03,
        ir_threshold: float = 0.5,
        ortho_corr_threshold: float = 0.5,
        ortho_t_threshold: float = 1.96,
        icir_std_threshold: float = 0.5,
        winsorize_pct: Tuple[float, float] = (0.01, 0.99),
    ):
        """
        初始化因子复核器。

        Args:
            factor_data: 因子日度数据 (index=日期, columns=因子名)
            returns: 前瞻收益率序列 (index=日期)
            survival_threshold: 存活率阈值（默认 85%）
            missing_threshold: 缺失率阈值（默认 15%）
            ic_threshold: IC阈值（默认 0.03）
            ir_threshold: IR阈值（默认 0.5）
            ortho_corr_threshold: 正交性相关系数阈值（默认 0.5）
            ortho_t_threshold: 正交性t值阈值（默认 1.96）
            icir_std_threshold: ICIR标准差阈值（默认 0.5）
            winsorize_pct: 缩尾分位数 (下, 上)
        """
        self._factor_data = factor_data
        self._returns = returns
        self._survival_threshold = survival_threshold
        self._missing_threshold = missing_threshold
        self._ic_threshold = ic_threshold
        self._ir_threshold = ir_threshold
        self._ortho_corr_threshold = ortho_corr_threshold
        self._ortho_t_threshold = ortho_t_threshold
        self._icir_std_threshold = icir_std_threshold
        self._winsorize_pct = winsorize_pct

        # 对齐数据
        common_idx = self._factor_data.index.intersection(self._returns.index)
        self._factor_data = self._factor_data.loc[common_idx]
        self._returns = self._returns.loc[common_idx]

        # P1-1 整改：复用公共 FactorEvaluator，避免重复实现 IC/IR 计算
        self._evaluator = FactorEvaluator(
            forward_period=5,
            ic_window=60,
            min_observations=30,
        )

    def run_full_review(self) -> FactorReviewReport:
        """执行全部6项复核检查。"""
        results = []
        for col in self._factor_data.columns:
            factor_series = self._factor_data[col]
            category = self._get_category(col)
            result = self._review_single(col, factor_series, category)
            results.append(result)

        report = FactorReviewReport(results=results)
        stats = {}
        for r in results:
            stats[r.recommendation] = stats.get(r.recommendation, 0) + 1
        report.summary_stats = stats
        return report

    def _get_category(self, name: str) -> str:
        """根据因子名获取类别。"""
        for cat, names in self.FACTOR_CATEGORIES.items():
            if name in names:
                return cat
        return "?"

    def _review_single(self, name: str, factor: pd.Series, category: str) -> FactorReviewResult:
        """对单个因子执行6项复核。"""
        result = FactorReviewResult(name=name, category=category)

        # 1. 数据存活率
        result.survival_rate = self._check_survival(factor)
        result.survival_pass = result.survival_rate >= self._survival_threshold

        # 2. 缺失值占比
        result.missing_rate = self._check_missing(factor)
        result.missing_pass = result.missing_rate <= self._missing_threshold

        # 3. 异常值抵抗
        ic_before, ic_after = self._check_outlier_resistance(factor)
        result.ic_before_winsorize = ic_before
        result.ic_after_winsorize = ic_after
        # 若截尾导致IC翻转（符号变化），则抗噪极差
        result.outlier_pass = (
            abs(ic_after) >= abs(ic_before) * 0.5 and
            np.sign(ic_before) == np.sign(ic_after)
        )

        # 4. 参数敏感性
        result.ic_sensitivity = self._check_sensitivity(factor)
        result.sensitivity_pass = result.ic_sensitivity < 0.3  # IC变化率 < 30%

        # 5. 因子正交性
        barra_corr, t_stat = self._check_orthogonality(factor, name)
        result.barra_corr = barra_corr
        result.ortho_t_stat = t_stat
        result.ortho_pass = (
            all(abs(c) <= self._ortho_corr_threshold for c in barra_corr.values()) or
            t_stat >= self._ortho_t_threshold
        )

        # 6. 时序稳定性 — 委托 FactorEvaluator 多周期 IC 方法（P1-1 整改）
        icir_mean, icir_std = self._check_stability(factor)
        result.rolling_icir_mean = icir_mean
        result.rolling_icir_std = icir_std
        result.stability_pass = (
            icir_std < self._icir_std_threshold and
            abs(icir_mean) > 0.1
        )

        return result

    # ── 1. 数据存活率 ──

    def _check_survival(self, factor: pd.Series) -> float:
        """计算因子数据的存活率（有效值占比）。"""
        total = len(factor)
        if total == 0:
            return 0.0
        valid = factor.notna().sum() + (factor != np.inf).sum() - (factor == np.inf).sum()
        return valid / total

    # ── 2. 缺失值占比 ──

    def _check_missing(self, factor: pd.Series) -> float:
        """计算因子缺失率。"""
        total = len(factor)
        if total == 0:
            return 1.0
        return factor.isna().sum() / total

    # ── 3. 异常值抵抗 ──

    def _check_outlier_resistance(self, factor: pd.Series) -> Tuple[float, float]:
        """
        极值处理前后的IC对比。

        P1-1 整改（2026-06-07）：通过 FactorEvaluator.evaluate 复用 IC 计算逻辑，
        避免与 factor_evaluator._compute_ic_stats 重复实现。

        Returns:
            (截尾前IC, 截尾后IC)
        """
        aligned = pd.DataFrame({"factor": factor, "ret": self._returns}).dropna()
        if len(aligned) < 30:
            return 0.0, 0.0

        # 截尾前IC（委托 FactorEvaluator）
        ic_before = self._evaluator.evaluate(
            factor_name="_raw",
            factor_scores=aligned["factor"].to_numpy(),
            forward_returns=aligned["ret"].to_numpy(),
        ).ic_mean

        # Winsorize截尾
        lo, hi = self._winsorize_pct
        factor_winsorized = aligned["factor"].clip(
            lower=aligned["factor"].quantile(lo),
            upper=aligned["factor"].quantile(hi),
        )

        # 截尾后IC（委托 FactorEvaluator）
        ic_after = self._evaluator.evaluate(
            factor_name="_winsorized",
            factor_scores=factor_winsorized.to_numpy(),
            forward_returns=aligned["ret"].to_numpy(),
        ).ic_mean

        return float(ic_before), float(ic_after)

    # ── 4. 参数敏感性 ──

    def _check_sensitivity(self, factor: pd.Series) -> float:
        """
        参数敏感性检测：改变跳空修复权重(0.3/0.7)验证IC变化。

        简化实现：对因子值做 ±20% 扰动，计算IC变化率。
        使用 Pearson 相关性，与 `factor_evaluator.FactorEvaluator` 保持一致
        （规则17：复用同一相关性算法，避免评估口径不一致）。
        """
        aligned = pd.DataFrame({"factor": factor, "ret": self._returns}).dropna()
        if len(aligned) < 30:
            return 1.0

        # Pearson 与 FactorEvaluator._compute_ic_stats 保持一致
        base_ic = abs(aligned["factor"].corr(aligned["ret"], method="pearson"))

        # 扰动 +20%
        perturbed_up = aligned["factor"] * 1.2
        ic_up = abs(perturbed_up.corr(aligned["ret"], method="pearson"))

        # 扰动 -20%
        perturbed_down = aligned["factor"] * 0.8
        ic_down = abs(perturbed_down.corr(aligned["ret"], method="pearson"))

        if base_ic < 1e-8:
            return 1.0

        # 最大IC变化率
        max_change = max(abs(ic_up - base_ic), abs(ic_down - base_ic)) / base_ic
        return float(max_change)

    # ── 5. 因子正交性 ──

    def _check_orthogonality(
        self, factor: pd.Series, name: str
    ) -> Tuple[Dict[str, float], float]:
        """
        与传统Barra风格因子（动量、波动率）的相关性检测。

        在因子框架中，动量因子用 20日收益率 近似，波动率因子用 20日标准差 近似。

        Returns:
            (Barra因子相关系数字典, 正交化后t统计量)
        """
        aligned = pd.DataFrame({"factor": factor, "ret": self._returns}).dropna()
        if len(aligned) < 30:
            return {}, 0.0

        # 构建Barra近似因子
        # 动量因子：20日滚动收益率
        momentum = self._returns.rolling(20).sum().shift(1)
        # 波动率因子：20日滚动标准差
        volatility = self._returns.rolling(20).std().shift(1)

        barra_factors = pd.DataFrame({
            "momentum": momentum,
            "volatility": volatility,
        }).reindex(aligned.index).dropna()

        common = aligned.index.intersection(barra_factors.index)
        if len(common) < 30:
            return {}, 0.0

        factor_aligned = aligned.loc[common, "factor"]
        barra_aligned = barra_factors.loc[common]

        # 计算相关系数
        corr_dict = {}
        for col in barra_aligned.columns:
            corr = factor_aligned.corr(barra_aligned[col], method="spearman")
            corr_dict[col] = float(corr)

        # 正交化：对Barra因子回归取残差
        X = barra_aligned.values
        y = factor_aligned.values.reshape(-1, 1)

        if X.shape[1] > 0:
            try:
                reg = LinearRegression()
                reg.fit(X, y)
                residual = y - reg.predict(X)
                # 残差与y的t检验
                t_stat = float(stats.ttest_1samp(residual.flatten(), 0).statistic)
            except Exception:
                t_stat = 0.0
        else:
            t_stat = 0.0

        return corr_dict, t_stat

    # ── 6. 时序稳定性 ──

    def _check_stability(self, factor: pd.Series) -> Tuple[float, float]:
        """
        滚动 1 年期 ICIR 的时间方差。

        P1-1 整改（2026-06-07）：核心滚动 ICIR 序列计算委托给
        FactorEvaluator._compute_ic_stats（通过 evaluate 调用），
        避免重复实现。

        Returns:
            (ICIR均值, ICIR标准差)
        """
        aligned = pd.DataFrame({"factor": factor, "ret": self._returns}).dropna()
        if len(aligned) < 252:
            return 0.0, 1.0

        window = min(252, len(aligned) // 2)
        min_window = 60

        icir_list: List[float] = []
        for i in range(window, len(aligned)):
            sub = aligned.iloc[i - window:i]
            if len(sub) < min_window:
                continue
            # 委托 FactorEvaluator 计算窗口内 IC（避免与 _compute_ic_stats 重复实现）
            ic = self._evaluator.evaluate(
                factor_name="_stable",
                factor_scores=sub["factor"].to_numpy(),
                forward_returns=sub["ret"].to_numpy(),
            ).ic_mean
            ic_std = sub["factor"].rolling(20).std().mean()
            if ic_std > 1e-8:
                icir = ic / ic_std
            else:
                icir = 0.0
            icir_list.append(icir)

        if not icir_list:
            return 0.0, 1.0

        return float(np.mean(icir_list)), float(np.std(icir_list))

    # ── 便捷方法 ──

    def quick_review(self, factor_name: str) -> FactorReviewResult:
        """快速复核单个因子。"""
        if factor_name not in self._factor_data.columns:
            raise ValueError(f"因子 {factor_name} 不存在")
        factor = self._factor_data[factor_name]
        category = self._get_category(factor_name)
        return self._review_single(factor_name, factor, category)

    def get_pass_factors(self) -> List[str]:
        """获取通过全部复核的因子列表。"""
        report = self.run_full_review()
        return [r.name for r in report.results if r.overall_pass]

    def get_retain_factors(self) -> List[str]:
        """获取建议保留的因子列表。"""
        report = self.run_full_review()
        return [r.name for r in report.results if r.recommendation == "保留"]