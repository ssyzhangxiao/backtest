"""因子池管理器（规则21 — 因子池扩展）。

功能：
  - IC 阈值过滤：仅保留 IC > threshold 的因子
  - 相关性去冗余：因子间相关系数 > max_corr 时，保留 IC 更高的因子
  - 权重优化：按 IR 加权或等权分配
  - 衰减监控：委托 FactorEvaluator 检测因子 IC 衰减

复用约束（规则21.4）：
  - 必须复用 core.ext.factors.evaluator.FactorEvaluator 的 IC 计算
  - 必须复用 core.ext.factors.selector.FactorSelector 的筛选逻辑
  - 不得重写 IC / IR / 相关性计算
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from core.ext.factors.evaluator import FactorEvaluator, FactorEvalResult
from core.ext.factors.selector import FactorSelector

logger = logging.getLogger(__name__)


@dataclass
class FactorPoolConfig:
    """因子池配置。"""

    ic_threshold: float = 0.03
    ir_threshold: float = 0.5
    max_correlation: float = 0.7
    weight_method: str = "ir"  # "ir" | "equal" | "ic"
    decay_window: int = 60
    forward_period: int = 5
    min_observations: int = 30


@dataclass
class FactorInfo:
    """因子池中单个因子的元信息。"""

    name: str
    weight: float = 0.0
    ic_mean: float = 0.0
    ir: float = 0.0
    is_active: bool = True
    decay_status: str = "healthy"  # healthy | warning | decaying | dead


class FactorPoolManager:
    """因子池管理器。

    管理因子的生命周期：入池评估 → 权重分配 → 衰减监控 → 淘汰。

    用法::

        pool = FactorPoolManager(config=FactorPoolConfig())
        pool.evaluate_and_add(
            factor_scores={"trend": scores1, "momentum": scores2},
            forward_returns=returns,
        )
        weights = pool.get_weights()
        active = pool.get_active_factors()
    """

    def __init__(self, config: Optional[FactorPoolConfig] = None) -> None:
        self.config = config or FactorPoolConfig()
        self._factors: Dict[str, FactorInfo] = {}
        self._evaluator = FactorEvaluator(
            forward_period=self.config.forward_period,
            ic_window=self.config.decay_window,
            min_observations=self.config.min_observations,
        )
        self._selector = FactorSelector(
            ic_threshold=self.config.ic_threshold,
            ir_threshold=self.config.ir_threshold,
            max_correlation=self.config.max_correlation,
        )

    def evaluate_and_add(
        self,
        factor_scores: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
    ) -> List[str]:
        """评估因子并添加到池中。

        Args:
            factor_scores: {因子名: 得分数组}
            forward_returns: 前瞻收益数组

        Returns:
            入池因子名列表
        """
        # 1. 评估每个因子
        eval_results: Dict[str, FactorEvalResult] = {}
        for name, scores in factor_scores.items():
            result = self._evaluator.evaluate(
                factor_name=name,
                factor_scores=scores,
                forward_returns=forward_returns,
            )
            eval_results[name] = result

        # 2. 筛选（传 eval_results + factor_scores_dict，与 FactorSelector.select 签名对齐）
        selection = self._selector.select(
            eval_results=eval_results,
            factor_scores_dict=factor_scores,
        )

        # 3. 更新池
        for name in selection.selected:
            er = eval_results.get(name)
            self._factors[name] = FactorInfo(
                name=name,
                ic_mean=er.ic_mean if er else 0.0,
                ir=er.ir if er else 0.0,
                is_active=True,
            )

        # 4. 重新分配权重
        self._reweight()

        added = selection.selected
        if selection.rejected:
            logger.info(
                "因子池筛选淘汰: %s",
                ", ".join(f"{k}({v})" for k, v in selection.rejected.items()),
            )
        return added

    def get_weights(self) -> Dict[str, float]:
        """获取当前活跃因子的权重。"""
        return {
            name: info.weight
            for name, info in self._factors.items()
            if info.is_active
        }

    def get_active_factors(self) -> List[str]:
        """获取当前活跃因子名列表。"""
        return [
            name for name, info in self._factors.items()
            if info.is_active
        ]

    def check_decay(
        self,
        factor_scores: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
    ) -> Dict[str, str]:
        """检查因子衰减状态。

        Args:
            factor_scores: {因子名: 得分数组}
            forward_returns: 前瞻收益数组

        Returns:
            {因子名: 衰减状态}
        """
        decay_map: Dict[str, str] = {}
        for name, scores in factor_scores.items():
            if name not in self._factors:
                continue
            result = self._evaluator.evaluate(
                factor_name=name,
                factor_scores=scores,
                forward_returns=forward_returns,
            )
            if result.ic_mean < 0.01:
                status = "dead"
            elif result.ic_mean < 0.02:
                status = "decaying"
            elif result.ic_mean < self.config.ic_threshold:
                status = "warning"
            else:
                status = "healthy"

            self._factors[name].ic_mean = result.ic_mean
            self._factors[name].ir = result.ir
            self._factors[name].decay_status = status

            if status == "dead":
                self._factors[name].is_active = False
                logger.warning("因子 %s 已死亡，自动停用", name)

            decay_map[name] = status

        self._reweight()
        return decay_map

    def remove_factor(self, name: str) -> None:
        """手动移除因子。"""
        if name in self._factors:
            del self._factors[name]
            self._reweight()
            logger.info("因子 %s 已从池中移除", name)

    def summary(self) -> str:
        """返回因子池摘要。"""
        active = self.get_active_factors()
        lines = [
            f"因子池: {len(active)}/{len(self._factors)} 活跃",
            f"权重分配: {self.config.weight_method}",
        ]
        for name in active:
            info = self._factors[name]
            lines.append(
                f"  {name}: IC={info.ic_mean:.4f} IR={info.ir:.4f} "
                f"W={info.weight:.2%} [{info.decay_status}]"
            )
        return "\n".join(lines)

    def _reweight(self) -> None:
        """根据权重方法重新分配权重。"""
        active = {
            name: info
            for name, info in self._factors.items()
            if info.is_active
        }
        if not active:
            return

        method = self.config.weight_method
        if method == "equal":
            w = 1.0 / len(active)
            for info in active.values():
                info.weight = w
        elif method == "ir":
            total_ir = sum(max(info.ir, 0.0) for info in active.values())
            if total_ir > 0:
                for info in active.values():
                    info.weight = max(info.ir, 0.0) / total_ir
            else:
                w = 1.0 / len(active)
                for info in active.values():
                    info.weight = w
        elif method == "ic":
            total_ic = sum(max(info.ic_mean, 0.0) for info in active.values())
            if total_ic > 0:
                for info in active.values():
                    info.weight = max(info.ic_mean, 0.0) / total_ic
            else:
                w = 1.0 / len(active)
                for info in active.values():
                    info.weight = w
        else:
            raise ValueError(f"未知权重方法: {method}")


__all__ = [
    "FactorPoolConfig",
    "FactorInfo",
    "FactorPoolManager",
]
