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

P2-2 整改（2026-06-07）：apply_cross 限制最大组合数（MAX_CROSS_COMBINATIONS=200），
  因子数较多时防止 O(N²) 组合爆炸。
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)

# 变换有效性阈值（规则9）
IC_IMPROVEMENT_THRESHOLD = 0.20
IR_IMPROVEMENT_THRESHOLD = 0.30

# P2-2 整改：交叉项最大组合数（超过则按IC降序截断，避免O(N²)爆炸）
MAX_CROSS_COMBINATIONS = 200


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
            factor_scores={"trend": arr1, "term_structure": arr2},
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

    def exp_transform(
        self,
        factor: np.ndarray,
        name: str = "",
        clip_threshold: float = 10.0,
    ) -> np.ndarray:
        """
        指数变换：sign(f) * (exp(|f|) - 1)

        放大信号强度，对大幅因子值放大更多。

        Args:
            factor: 因子值序列
            name: 因子名（用于日志）
            clip_threshold: |f| 裁剪上限（默认 10.0）。
                            裁剪可避免 exp 数值溢出（exp(700) 即达 float64 上限）。
                            调大可保留更多极端信号；设为 None 表示不裁剪（极端值会爆inf）。
                            推荐区间 [3, 10]：
                              - 3：保守，避免极端值主导排序
                              - 10：激进，保留尾部信号但已对超 10 部分饱和
        """
        arr = np.asarray(factor, dtype=float)
        if clip_threshold is None:
            return np.sign(arr) * (np.exp(np.abs(arr)) - 1)
        clipped = np.clip(np.abs(arr), 0, clip_threshold)
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
        max_combinations: int = MAX_CROSS_COMBINATIONS,
    ) -> List[TransformResult]:
        """
        对因子对应用交叉项构造并评估效果。

        P2-2 整改（2026-06-07）：增加 max_combinations 参数限制最大组合数。
        当因子数 N 较多时，组合数 N*(N-1)/2 会快速爆炸（如 N=30 → 435 组合）。
        超过 max_combinations 时：
          1. 先按 |IC| 降序排序所有因子
          2. 取 top-K 因子进行组合，保证最有价值的因子对先被处理
          3. 超出 max_combinations 时记录 warning 日志

        Args:
            factor_scores_dict: {因子名: 得分序列}
            forward_returns: 前瞻收益序列
            max_combinations: 最多生成的交叉组合数，默认 200

        Returns:
            变换结果列表
        """
        results: List[TransformResult] = []
        names = list(factor_scores_dict.keys())
        total_pairs = len(names) * (len(names) - 1) // 2

        # P2-2：组合数控制
        if total_pairs > max_combinations:
            # 按 |IC| 降序排序，截取 top-K 因子（K 满足 K*(K-1)/2 <= max_combinations）
            ic_values: List[Tuple[str, float]] = []
            for n in names:
                arr = np.asarray(factor_scores_dict[n], dtype=float)
                ic_val = abs(self._compute_ic(arr, forward_returns))
                ic_values.append((n, ic_val))
            ic_values.sort(key=lambda kv: kv[1], reverse=True)

            # 求解最大 K 使 K*(K-1)/2 <= max_combinations
            k = 1
            while k * (k - 1) // 2 <= max_combinations:
                k += 1
            k = max(2, k - 1)  # 至少保留 2 个因子

            original_count = len(names)
            names = [n for n, _ in ic_values[:k]]
            logger.warning(
                "apply_cross 组合数=%d 超过 max_combinations=%d，"
                "已按 |IC| 降序截取 top-%d 因子（原始 %d 个）",
                total_pairs, max_combinations, k, original_count,
            )

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
