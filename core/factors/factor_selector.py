"""
因子筛选器。

通过 IC 检验、相关性分析和冗余度检测确定最终因子集。

筛选流程（规则9）：
  1. IC 检验：IC > 0.03 且 IR > 0.5 的因子保留
  2. 相关性去冗余：因子间相关系数 > 0.7 时，保留 IC 更高的因子
  3. IR 排序：按 IR 降序排列，优先保留预测稳定的因子

验证标准：最终因子集平均 IC > 0.04，最大互相关 < 0.6
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import logging

import numpy as np
import pandas as pd

from .factor_evaluator import FactorEvaluator, FactorEvalResult, IC_THRESHOLD, IR_THRESHOLD, CORRELATION_REDUNDANCY

logger = logging.getLogger(__name__)

# 最终因子集验证标准
AVG_IC_THRESHOLD = 0.04
MAX_CROSS_CORR = 0.6


@dataclass
class SelectionResult:
    """因子筛选结果。"""

    selected: List[str]
    rejected: Dict[str, str]
    correlation_matrix: Optional[pd.DataFrame] = None
    avg_ic: float = 0.0
    max_cross_corr: float = 0.0
    is_valid: bool = False

    def summary(self) -> str:
        """返回筛选摘要。"""
        status = "✅ 通过" if self.is_valid else "⚠️ 未达标"
        return (
            f"筛选结果：选中{len(self.selected)}个因子 "
            f"平均IC={self.avg_ic:.4f} 最大互相关={self.max_cross_corr:.4f} {status}\n"
            f"  选中: {', '.join(self.selected)}\n"
            f"  淘汰: {', '.join(f'{k}({v})' for k, v in self.rejected.items())}"
        )


class FactorSelector:
    """
    因子筛选器。

    三步筛选：IC检验 → 相关性去冗余 → IR排序。

    用法:
        selector = FactorSelector()
        result = selector.select(
            eval_results={"ts_momentum": result1, "roll_yield": result2},
            factor_scores_dict={"ts_momentum": arr1, "roll_yield": arr2},
        )
    """

    def __init__(
        self,
        ic_threshold: float = IC_THRESHOLD,
        ir_threshold: float = IR_THRESHOLD,
        corr_threshold: float = CORRELATION_REDUNDANCY,
        avg_ic_threshold: float = AVG_IC_THRESHOLD,
        max_cross_corr: float = MAX_CROSS_CORR,
    ):
        """
        初始化因子筛选器。

        Args:
            ic_threshold: IC绝对值阈值
            ir_threshold: IR绝对值阈值
            corr_threshold: 相关性冗余阈值
            avg_ic_threshold: 最终因子集平均IC阈值
            max_cross_corr: 最终因子集最大互相关阈值
        """
        self.ic_threshold = ic_threshold
        self.ir_threshold = ir_threshold
        self.corr_threshold = corr_threshold
        self.avg_ic_threshold = avg_ic_threshold
        self.max_cross_corr = max_cross_corr

    def select(
        self,
        eval_results: Dict[str, FactorEvalResult],
        factor_scores_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> SelectionResult:
        """
        执行三步因子筛选。

        Args:
            eval_results: {因子名: FactorEvalResult}
            factor_scores_dict: {因子名: 得分序列}，用于计算相关矩阵

        Returns:
            SelectionResult 筛选结果
        """
        rejected: Dict[str, str] = {}

        # ── 第1步：IC检验 ──
        ic_passed: List[str] = []
        for name, result in eval_results.items():
            if result.is_valid:
                ic_passed.append(name)
            else:
                rejected[name] = result.reject_reason or "IC/IR不达标"

        logger.info(f"IC检验：{len(ic_passed)}/{len(eval_results)}通过")

        if not ic_passed:
            return SelectionResult(
                selected=[],
                rejected=rejected,
                avg_ic=0.0,
                max_cross_corr=0.0,
                is_valid=False,
            )

        # ── 第2步：相关性去冗余 ──
        corr_matrix = None
        if factor_scores_dict:
            evaluator = FactorEvaluator()
            corr_matrix = evaluator.compute_correlation_matrix(factor_scores_dict)

        deduped = self._deduplicate(
            ic_passed, eval_results, corr_matrix, rejected
        )

        # ── 第3步：IR排序 ──
        sorted_factors = sorted(
            deduped,
            key=lambda n: abs(eval_results[n].ir),
            reverse=True,
        )

        # ── 验证最终因子集 ──
        avg_ic = np.mean([abs(eval_results[n].ic_mean) for n in sorted_factors]) if sorted_factors else 0.0

        max_corr = 0.0
        if corr_matrix is not None and len(sorted_factors) > 1:
            sub_corr = corr_matrix.loc[sorted_factors, sorted_factors]
            # 取上三角（不含对角线）的最大值
            mask = np.triu(np.ones(sub_corr.shape, dtype=bool), k=1)
            max_corr = float(np.abs(sub_corr.values[mask]).max())

        is_valid = avg_ic >= self.avg_ic_threshold and max_corr <= self.max_cross_corr

        result = SelectionResult(
            selected=sorted_factors,
            rejected=rejected,
            correlation_matrix=corr_matrix,
            avg_ic=avg_ic,
            max_cross_corr=max_corr,
            is_valid=is_valid,
        )

        logger.info(result.summary())
        return result

    def _deduplicate(
        self,
        factors: List[str],
        eval_results: Dict[str, FactorEvalResult],
        corr_matrix: Optional[pd.DataFrame],
        rejected: Dict[str, str],
    ) -> List[str]:
        """
        相关性去冗余。

        因子间相关系数 > 阈值时，保留IC更高的因子。
        使用贪心算法：按IC降序遍历，与已保留因子高相关则淘汰。
        """
        if corr_matrix is None or len(factors) <= 1:
            return list(factors)

        # 按IC绝对值降序排列
        sorted_by_ic = sorted(
            factors,
            key=lambda n: abs(eval_results[n].ic_mean),
            reverse=True,
        )

        kept: List[str] = []

        for name in sorted_by_ic:
            is_redundant = False
            for kept_name in kept:
                try:
                    corr = abs(corr_matrix.loc[name, kept_name])
                except KeyError:
                    corr = 0.0
                if corr > self.corr_threshold:
                    is_redundant = True
                    rejected[name] = f"与{kept_name}相关={corr:.2f}>{self.corr_threshold}"
                    logger.debug(
                        f"冗余淘汰：{name} 与 {kept_name} "
                        f"相关系数={corr:.4f}，保留IC更高的{kept_name}"
                    )
                    break

            if not is_redundant:
                kept.append(name)

        removed_count = len(factors) - len(kept)
        if removed_count > 0:
            logger.info(f"去冗余：淘汰{removed_count}个冗余因子，保留{len(kept)}个")

        return kept
