"""
策略权重动态调整模块。

基于滚动Sharpe比率动态分配策略权重：
  - 基准权重：等权
  - 调整因子：weight_i = base_weight_i * (rolling_sharpe_i / avg_rolling_sharpe)
  - 归一化：确保权重之和 = 1

规则12要求：单次调整幅度≤20%，避免频繁大幅调整。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)

# 单次最大调整幅度（规则12）
MAX_ADJUSTMENT = 0.20


@dataclass
class WeightAdjustmentResult:
    """权重调整结果。"""

    old_weights: Dict[str, float] = field(default_factory=dict)
    new_weights: Dict[str, float] = field(default_factory=dict)
    adjustments: Dict[str, float] = field(default_factory=dict)
    is_capped: Dict[str, bool] = field(default_factory=dict)


class DynamicWeightAllocator:
    """
    策略权重动态调整器。

    根据各策略的滚动Sharpe比率动态分配权重。
    单次调整幅度受限（≤20%），避免突变。

    用法:
        allocator = DynamicWeightAllocator(base_weights={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        new_weights = allocator.adjust(rolling_sharpes={"A": 0.5, "B": 0.3, "C": -0.1, "D": 0.8})
    """

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        max_adjustment: float = MAX_ADJUSTMENT,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ):
        """
        初始化权重调整器。

        Args:
            base_weights: 基准权重，默认等权
            max_adjustment: 单次最大调整幅度（0~1）
            min_weight: 单策略最小权重
            max_weight: 单策略最大权重
        """
        self.base_weights = base_weights or {}
        self.max_adjustment = max_adjustment
        self.min_weight = min_weight
        self.max_weight = max_weight
        self._current_weights: Dict[str, float] = dict(self.base_weights)

    @property
    def current_weights(self) -> Dict[str, float]:
        """当前权重。"""
        return dict(self._current_weights)

    def adjust(
        self, rolling_sharpes: Dict[str, float]
    ) -> WeightAdjustmentResult:
        """
        根据滚动Sharpe调整权重。

        算法：
          1. 计算各策略Sharpe占比
          2. 生成目标权重
          3. 限制单次调整幅度
          4. 归一化

        Args:
            rolling_sharpes: {策略名: 滚动Sharpe}

        Returns:
            WeightAdjustmentResult 调整结果
        """
        if not rolling_sharpes:
            return WeightAdjustmentResult()

        strategies = list(rolling_sharpes.keys())
        n = len(strategies)

        # 基准权重（等权或自定义）
        if not self.base_weights:
            base = {s: 1.0 / n for s in strategies}
        else:
            base = {}
            for s in strategies:
                base[s] = self.base_weights.get(s, 1.0 / n)

        # 计算目标权重
        # 使用Sharpe的绝对值（负Sharpe的策略权重趋近于0）
        abs_sharpes = {s: max(0.0, rolling_sharpes.get(s, 0.0)) for s in strategies}
        total_sharpe = sum(abs_sharpes.values())

        if total_sharpe < 1e-8:
            # 所有Sharpe≤0时回退到基准权重
            target = dict(base)
        else:
            # 按Sharpe比例分配，但保留基准权重的影响
            target = {}
            for s in strategies:
                sharpe_ratio = abs_sharpes[s] / total_sharpe
                # 混合：70%按Sharpe分配 + 30%基准权重
                target[s] = 0.7 * sharpe_ratio + 0.3 * base[s]

        # 限制单次调整幅度（规则12）
        old_weights = dict(self._current_weights) if self._current_weights else dict(base)
        adjustments: Dict[str, float] = {}
        is_capped: Dict[str, bool] = {}

        for s in strategies:
            old_w = old_weights.get(s, base[s])
            new_w = target[s]
            change = new_w - old_w

            if abs(change) > self.max_adjustment:
                # 裁剪调整幅度
                capped_w = old_w + self.max_adjustment * np.sign(change)
                target[s] = capped_w
                is_capped[s] = True
                adjustments[s] = self.max_adjustment * np.sign(change)
            else:
                is_capped[s] = False
                adjustments[s] = change

        # 归一化
        total = sum(target.values())
        if total > 1e-8:
            target = {s: w / total for s, w in target.items()}
        else:
            target = dict(base)

        # 范围限制
        for s in strategies:
            target[s] = max(self.min_weight, min(self.max_weight, target[s]))

        # 再次归一化
        total = sum(target.values())
        if total > 1e-8:
            target = {s: w / total for s, w in target.items()}

        self._current_weights = target

        result = WeightAdjustmentResult(
            old_weights=old_weights,
            new_weights=target,
            adjustments=adjustments,
            is_capped=is_capped,
        )

        capped_strategies = [s for s, c in is_capped.items() if c]
        if capped_strategies:
            logger.debug(
                f"权重调整被裁剪：{capped_strategies} "
                f"（单次调整幅度>{self.max_adjustment:.0%}）"
            )

        return result

    def reset(self):
        """重置到基准权重。"""
        self._current_weights = dict(self.base_weights)
