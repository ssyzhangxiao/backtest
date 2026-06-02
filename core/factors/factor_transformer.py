"""
因子变换器。

对原始因子进行非线性变换和交叉项构造，提升因子预测力。

变换类型：
  - 对数变换：log(|f| + 1)，压缩极端值
  - 指数变换：sign(f) * (exp(|f|) - 1)，放大信号
  - 幂函数：sign(f) * |f|^power，减少偏度
  - 因子乘积：f1 * f2，捕捉共振信号
  - 因子比率：f1 / f2，风险调整信号
  - 条件组合：if condition then f1 else f2

规则9要求：变换后IC提升 > 20% 或 IR提升 > 30% 方为有效。
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)

# 变换有效性阈值（规则9）
IC_IMPROVEMENT_THRESHOLD = 0.20
IR_IMPROVEMENT_THRESHOLD = 0.30


@dataclass
class TransformResult:
    """变换结果。"""

    name: str
    values: np.ndarray
    transform_type: str
    source_factors: List[str]
    ic_before: float = 0.0
    ic_after: float = 0.0
    ic_improvement: float = 0.0
    is_effective: bool = False

    def summary(self) -> str:
        """返回变换摘要。"""
        status = "✅ 有效" if self.is_effective else "❌ 无效"
        return (
            f"[{self.name}] {self.transform_type} "
            f"IC: {self.ic_before:.4f}→{self.ic_after:.4f} "
            f"提升={self.ic_improvement:.1%} {status}"
        )


class FactorTransformer:
    """
    因子变换器。

    对原始因子进行非线性变换和交叉项构造。
    每种变换都会自动评估IC变化，仅保留有效变换。

    用法:
        transformer = FactorTransformer()
        results = transformer.apply_all(
            factor_scores={"ts_momentum": arr1, "roll_yield": arr2},
            forward_returns=ret_arr,
        )
    """

    def __init__(
        self,
        powers: Optional[List[float]] = None,
        ic_evaluator: Optional[Callable] = None,
    ):
        """
        初始化因子变换器。

        Args:
            powers: 幂函数指数列表，默认 [0.5, 0.33, 2.0]
            ic_evaluator: IC计算函数，默认用 np.corrcoef
        """
        self.powers = powers or [0.5, 0.33, 2.0]
        self._compute_ic = ic_evaluator or self._default_ic

    @staticmethod
    def _default_ic(scores: np.ndarray, returns: np.ndarray) -> float:
        """默认IC计算：Pearson相关系数。"""
        valid = ~(np.isnan(scores) | np.isnan(returns))
        s, r = scores[valid], returns[valid]
        if len(s) < 10 or np.std(s) < 1e-10 or np.std(r) < 1e-10:
            return 0.0
        corr = float(np.corrcoef(s, r)[0, 1])
        return 0.0 if np.isnan(corr) else corr

    # ── 非线性变换 ──

    def log_transform(self, factor: np.ndarray, name: str = "") -> np.ndarray:
        """
        对数变换：log(|f| + 1) * sign(f)

        压缩极端值，保留方向信息。
        """
        arr = np.asarray(factor, dtype=float)
        return np.sign(arr) * np.log1p(np.abs(arr))

    def exp_transform(self, factor: np.ndarray, name: str = "") -> np.ndarray:
        """
        指数变换：sign(f) * (exp(|f|) - 1)

        放大信号强度，对大幅因子值放大更多。
        """
        arr = np.asarray(factor, dtype=float)
        # 裁剪避免溢出（|f| > 10 时 exp 爆炸）
        clipped = np.clip(np.abs(arr), 0, 10)
        return np.sign(arr) * (np.exp(clipped) - 1)

    def power_transform(
        self, factor: np.ndarray, power: float = 0.5, name: str = ""
    ) -> np.ndarray:
        """
        幂函数变换：sign(f) * |f|^power

        power < 1 压缩极端值，power > 1 放大极端值。
        """
        arr = np.asarray(factor, dtype=float)
        return np.sign(arr) * np.power(np.abs(arr), power)

    # ── 交叉项构造 ──

    def factor_product(
        self, f1: np.ndarray, f2: np.ndarray
    ) -> np.ndarray:
        """
        因子乘积：f1 * f2

        捕捉两个因子的共振信号。同号放大，异号抵消。
        """
        return np.asarray(f1, dtype=float) * np.asarray(f2, dtype=float)

    def factor_ratio(
        self, f1: np.ndarray, f2: np.ndarray, eps: float = 1e-6
    ) -> np.ndarray:
        """
        因子比率：f1 / f2

        风险调整信号。如动量/波动率 = 风险调整动量。
        """
        a = np.asarray(f1, dtype=float)
        b = np.asarray(f2, dtype=float)
        # 避免除零
        b_safe = np.where(np.abs(b) < eps, eps * np.sign(b + eps), b)
        return a / b_safe

    def conditional_combine(
        self,
        condition: np.ndarray,
        f_true: np.ndarray,
        f_false: np.ndarray,
    ) -> np.ndarray:
        """
        条件组合：if condition > 0 then f_true else f_false

        根据条件选择不同因子，如趋势向上用动量因子，向下用均值回归因子。
        """
        cond = np.asarray(condition, dtype=float)
        t = np.asarray(f_true, dtype=float)
        f = np.asarray(f_false, dtype=float)
        return np.where(cond > 0, t, f)

    # ── 批量变换与评估 ──

    def apply_nonlinear(
        self,
        factor_name: str,
        factor_scores: np.ndarray,
        forward_returns: np.ndarray,
    ) -> List[TransformResult]:
        """
        对单个因子应用所有非线性变换并评估效果。

        Args:
            factor_name: 因子名称
            factor_scores: 因子得分序列
            forward_returns: 前瞻收益序列

        Returns:
            变换结果列表
        """
        results: List[TransformResult] = []
        ic_before = abs(self._compute_ic(factor_scores, forward_returns))

        # 对数变换
        log_vals = self.log_transform(factor_scores)
        ic_log = abs(self._compute_ic(log_vals, forward_returns))
        improvement = self._calc_improvement(ic_before, ic_log)
        results.append(TransformResult(
            name=f"{factor_name}_log",
            values=log_vals,
            transform_type="log",
            source_factors=[factor_name],
            ic_before=ic_before,
            ic_after=ic_log,
            ic_improvement=improvement,
            is_effective=improvement >= IC_IMPROVEMENT_THRESHOLD,
        ))

        # 指数变换
        exp_vals = self.exp_transform(factor_scores)
        ic_exp = abs(self._compute_ic(exp_vals, forward_returns))
        improvement = self._calc_improvement(ic_before, ic_exp)
        results.append(TransformResult(
            name=f"{factor_name}_exp",
            values=exp_vals,
            transform_type="exp",
            source_factors=[factor_name],
            ic_before=ic_before,
            ic_after=ic_exp,
            ic_improvement=improvement,
            is_effective=improvement >= IC_IMPROVEMENT_THRESHOLD,
        ))

        # 幂函数变换
        for p in self.powers:
            pow_vals = self.power_transform(factor_scores, p)
            ic_pow = abs(self._compute_ic(pow_vals, forward_returns))
            improvement = self._calc_improvement(ic_before, ic_pow)
            results.append(TransformResult(
                name=f"{factor_name}_pow{p}",
                values=pow_vals,
                transform_type=f"power({p})",
                source_factors=[factor_name],
                ic_before=ic_before,
                ic_after=ic_pow,
                ic_improvement=improvement,
                is_effective=improvement >= IC_IMPROVEMENT_THRESHOLD,
            ))

        return results

    def apply_cross(
        self,
        factor_scores_dict: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
    ) -> List[TransformResult]:
        """
        对因子对应用交叉项构造并评估效果。

        Args:
            factor_scores_dict: {因子名: 得分序列}
            forward_returns: 前瞻收益序列

        Returns:
            变换结果列表
        """
        results: List[TransformResult] = []
        names = list(factor_scores_dict.keys())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                f1 = np.asarray(factor_scores_dict[n1], dtype=float)
                f2 = np.asarray(factor_scores_dict[n2], dtype=float)

                # 因子乘积
                prod = self.factor_product(f1, f2)
                ic_prod = abs(self._compute_ic(prod, forward_returns))
                ic_max = max(
                    abs(self._compute_ic(f1, forward_returns)),
                    abs(self._compute_ic(f2, forward_returns)),
                )
                improvement = self._calc_improvement(ic_max, ic_prod)
                results.append(TransformResult(
                    name=f"{n1}_x_{n2}",
                    values=prod,
                    transform_type="product",
                    source_factors=[n1, n2],
                    ic_before=ic_max,
                    ic_after=ic_prod,
                    ic_improvement=improvement,
                    is_effective=improvement >= IC_IMPROVEMENT_THRESHOLD,
                ))

                # 因子比率
                ratio = self.factor_ratio(f1, f2)
                ic_ratio = abs(self._compute_ic(ratio, forward_returns))
                improvement = self._calc_improvement(ic_max, ic_ratio)
                results.append(TransformResult(
                    name=f"{n1}_div_{n2}",
                    values=ratio,
                    transform_type="ratio",
                    source_factors=[n1, n2],
                    ic_before=ic_max,
                    ic_after=ic_ratio,
                    ic_improvement=improvement,
                    is_effective=improvement >= IC_IMPROVEMENT_THRESHOLD,
                ))

        return results

    def apply_all(
        self,
        factor_scores_dict: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
    ) -> List[TransformResult]:
        """
        应用所有变换（非线性 + 交叉项）。

        Args:
            factor_scores_dict: {因子名: 得分序列}
            forward_returns: 前瞻收益序列

        Returns:
            所有变换结果列表
        """
        all_results: List[TransformResult] = []

        # 非线性变换
        for name, scores in factor_scores_dict.items():
            results = self.apply_nonlinear(name, scores, forward_returns)
            all_results.extend(results)

        # 交叉项
        cross_results = self.apply_cross(factor_scores_dict, forward_returns)
        all_results.extend(cross_results)

        # 日志输出
        effective = [r for r in all_results if r.is_effective]
        logger.info(
            f"因子变换完成：共{len(all_results)}种变换，"
            f"{len(effective)}种有效（IC提升>{IC_IMPROVEMENT_THRESHOLD:.0%}）"
        )
        for r in effective:
            logger.info(f"  {r.summary()}")

        return all_results

    @staticmethod
    def _calc_improvement(ic_before: float, ic_after: float) -> float:
        """计算IC提升比例。"""
        if abs(ic_before) < 1e-8:
            return 1.0 if abs(ic_after) > 1e-8 else 0.0
        return (ic_after - ic_before) / ic_before
