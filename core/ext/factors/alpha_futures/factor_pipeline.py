"""
因子Pipeline：统一编排评估、变换、筛选、复核全流程。

流程：
  1. FactorEngine 计算全部因子
  2. FactorEvaluator 评估 IC/IR/多周期稳定性
  3. FactorTransformer 生成非线性变换 + 交叉项
  4. FactorSelector 三步筛选：IC检验 → 去冗余 → IR排序
  5. FactorReviewer 6项质量复核

用法：
    from core.factors.alpha_futures.factor_pipeline import FactorPipeline

    pipeline = FactorPipeline(config)
    pipeline.run(raw_data, forward_returns)
    print(pipeline.report())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from .factor_engine import FactorEngine
from ..evaluator import FactorEvaluator, FactorEvalResult
from ..transformer import FactorTransformer, TransformResult
from ..selector import FactorSelector, SelectionResult
from ..review import FactorReviewer, FactorReviewReport

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Pipeline完整运行结果。"""

    # 因子计算结果
    factor_scores: Dict[str, np.ndarray] = field(default_factory=dict)

    # 评估结果
    eval_results: Dict[str, FactorEvalResult] = field(default_factory=dict)

    # 变换结果
    transform_results: List[TransformResult] = field(default_factory=list)

    # 筛选结果
    selection_result: Optional[SelectionResult] = None

    # 复核报告
    review_report: Optional[FactorReviewReport] = None

    def summary(self) -> str:
        """生成完整摘要。"""
        lines = ["=" * 70, "因子Pipeline运行报告", "=" * 70]

        # 因子计算
        lines.append(f"\n【因子计算】共 {len(self.factor_scores)} 个因子")

        # 评估
        if self.eval_results:
            valid = sum(1 for r in self.eval_results.values() if r.is_valid)
            lines.append(f"\n【因子评估】{valid}/{len(self.eval_results)} 通过IC/IR检验")
            for name, r in sorted(self.eval_results.items()):
                status = "✅" if r.is_valid else "❌"
                lines.append(f"  {status} {name}: IC={r.ic_mean:.4f} IR={r.ir:.4f} 衰减={r.ic_decay_rate:.1%}")

        # 变换
        if self.transform_results:
            effective = [t for t in self.transform_results if t.is_effective]
            lines.append(f"\n【因子变换】{len(effective)}/{len(self.transform_results)} 种变换有效")
            for t in effective:
                lines.append(f"  ✅ {t.name}: {t.transform_type} IC {t.ic_before:.4f}→{t.ic_after:.4f} (+{t.ic_improvement:.1%})")

        # 筛选
        if self.selection_result:
            lines.append(f"\n【因子筛选】{self.selection_result.summary()}")

        # 复核
        if self.review_report:
            lines.append(f"\n【因子复核】{self.review_report.summary()}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """转为评估DataFrame（含因子得分 + 评估指标）。"""
        rows = []
        for name, scores in self.factor_scores.items():
            row = {
                "因子": name,
                "均值": round(float(np.nanmean(scores)), 4),
                "标准差": round(float(np.nanstd(scores)), 4),
                "NaN占比": f"{np.isnan(scores).mean():.1%}",
            }
            if name in self.eval_results:
                r = self.eval_results[name]
                row["IC"] = round(r.ic_mean, 4)
                row["IR"] = round(r.ir, 4)
                row["有效"] = "是" if r.is_valid else "否"
            rows.append(row)
        return pd.DataFrame(rows)


class FactorPipeline:
    """
    因子Pipeline：一站式因子计算、评估、变换、筛选、复核。

    用法:
        pipeline = FactorPipeline(config)
        result = pipeline.run(raw_data, forward_returns)
        print(result.summary())

        # 获取精选因子
        selected = pipeline.get_selected_factors()
        # 获取有效变换
        transforms = pipeline.get_effective_transforms()
    """

    def __init__(
        self,
        config: Any,
        factor_names: Optional[List[str]] = None,
        evaluator_kwargs: Optional[Dict] = None,
        transformer_kwargs: Optional[Dict] = None,
        selector_kwargs: Optional[Dict] = None,
        reviewer_kwargs: Optional[Dict] = None,
    ):
        """
        初始化Pipeline。

        Args:
            config: 全局配置对象（AlphaFuturesConfig）
            factor_names: 要计算的因子列表，None=全部已注册因子
            evaluator_kwargs: FactorEvaluator 参数
            transformer_kwargs: FactorTransformer 参数
            selector_kwargs: FactorSelector 参数
            reviewer_kwargs: FactorReviewer 参数
        """
        self.config = config
        self.factor_names = factor_names

        # 各组件
        self._engine = FactorEngine(config, factor_names)
        self._evaluator = FactorEvaluator(**(evaluator_kwargs or {}))
        self._transformer = FactorTransformer(**(transformer_kwargs or {}))
        self._selector = FactorSelector(**(selector_kwargs or {}))
        self._reviewer_kwargs = reviewer_kwargs or {}

        # 缓存
        self._last_result: Optional[PipelineResult] = None

    def run(
        self,
        raw_data: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
        run_review: bool = True,
    ) -> PipelineResult:
        """
        执行完整的因子Pipeline。

        Args:
            raw_data: 原始数据字典（同 FactorEngine.compute_all）
            forward_returns: 前瞻收益序列
            run_review: 是否执行6项复核（较耗时）

        Returns:
            PipelineResult
        """
        result = PipelineResult()

        # 1. 计算因子
        logger.info("步骤1/5: 计算因子...")
        result.factor_scores = self._engine.compute_all(raw_data)
        logger.info(f"  完成：{len(result.factor_scores)} 个因子")

        # 2. 评估因子
        logger.info("步骤2/5: 评估因子...")
        result.eval_results = self._evaluator.evaluate_batch(
            result.factor_scores, forward_returns
        )
        valid_count = sum(1 for r in result.eval_results.values() if r.is_valid)
        logger.info(f"  完成：{valid_count}/{len(result.eval_results)} 通过")

        # 3. 变换因子
        logger.info("步骤3/5: 变换因子...")
        result.transform_results = self._transformer.apply_all(
            result.factor_scores, forward_returns
        )
        effective_count = sum(1 for t in result.transform_results if t.is_effective)
        logger.info(f"  完成：{effective_count}/{len(result.transform_results)} 有效")

        # 4. 筛选因子
        logger.info("步骤4/5: 筛选因子...")
        result.selection_result = self._selector.select(
            result.eval_results, result.factor_scores
        )
        logger.info(f"  完成：选中 {len(result.selection_result.selected)} 个")

        # 5. 复核因子
        if run_review:
            logger.info("步骤5/5: 复核因子...")
            result.review_report = self._run_review(result.factor_scores, forward_returns)
            if result.review_report:
                stats = result.review_report.summary_stats
                logger.info(f"  完成：保留{stats.get('保留',0)} 降级{stats.get('降级',0)} 待优化{stats.get('待优化',0)} 剔除{stats.get('剔除',0)}")

        self._last_result = result
        return result

    def _run_review(
        self,
        factor_scores: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
    ) -> Optional[FactorReviewReport]:
        """
        执行6项因子复核。

        FactorReviewer 需要 pd.DataFrame (index=日期, columns=因子名) 和 pd.Series (returns)。
        这里将 numpy 数组转为 DataFrame。

        Args:
            factor_scores: 因子得分字典
            forward_returns: 前瞻收益序列

        Returns:
            FactorReviewReport 或 None
        """
        try:
            # 对齐长度，取最短
            min_len = min(
                min(len(v) for v in factor_scores.values()),
                len(forward_returns),
            )

            factor_df = pd.DataFrame({
                name: arr[:min_len]
                for name, arr in factor_scores.items()
            })
            returns_series = pd.Series(forward_returns[:min_len])

            reviewer = FactorReviewer(factor_df, returns_series, **self._reviewer_kwargs)
            return reviewer.run_full_review()
        except Exception as e:
            logger.warning("因子复核失败: %s", e)
            return None

    # ── 便捷方法 ──

    def get_selected_factors(self) -> List[str]:
        """获取筛选后保留的因子列表。"""
        if self._last_result and self._last_result.selection_result:
            return self._last_result.selection_result.selected
        return []

    def get_effective_transforms(self) -> List[TransformResult]:
        """获取有效的变换结果。"""
        if self._last_result:
            return [t for t in self._last_result.transform_results if t.is_effective]
        return []

    def get_valid_factors(self) -> List[str]:
        """获取通过IC/IR检验的因子。"""
        if self._last_result:
            return [n for n, r in self._last_result.eval_results.items() if r.is_valid]
        return []

    def get_retain_factors(self) -> List[str]:
        """获取建议保留的因子（复核通过）。"""
        if self._last_result and self._last_result.review_report:
            return [r.name for r in self._last_result.review_report.results
                    if r.recommendation == "保留"]
        return []

    def report(self) -> str:
        """输出当前结果摘要。"""
        if self._last_result:
            return self._last_result.summary()
        return "尚未运行Pipeline"

    @property
    def last_result(self) -> Optional[PipelineResult]:
        """获取最近一次运行结果。"""
        return self._last_result