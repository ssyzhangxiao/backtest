"""
策略权重管理模块。

委托 core/engine/rolling_ic.py 和 core/position/dynamic_weight.py，
不重新实现权重计算逻辑。
"""

import copy
from typing import Any, Dict, List

from loguru import logger


def extract_optimized_weights(e11_result: Dict[str, Any]) -> Dict[str, float]:
    """
    从 E11 因子分析结果中提取优化后的权重。

    取所有品种权重的均值作为全局优化权重。

    Args:
        e11_result: E11 因子分析结果字典

    Returns:
        归一化后的因子权重字典
    """
    if not e11_result:
        return {}

    all_weights: Dict[str, List[float]] = {}
    for symbol, data in e11_result.items():
        if isinstance(data, dict) and "final_weights" in data:
            for factor, weight in data["final_weights"].items():
                if factor not in all_weights:
                    all_weights[factor] = []
                all_weights[factor].append(float(weight))

    if not all_weights:
        return {}

    optimized: Dict[str, float] = {}
    for factor, weights in all_weights.items():
        optimized[factor] = round(sum(weights) / len(weights), 4)

    # 归一化
    total = sum(optimized.values())
    if total > 0:
        optimized = {k: round(v / total, 4) for k, v in optimized.items()}

    return optimized


def apply_ic_weights_to_config(
    config: Dict[str, Any],
    ic_weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    将 IC 优化权重应用到配置字典。

    Args:
        config: 原始配置字典
        ic_weights: IC 优化权重

    Returns:
        更新后的配置字典（深拷贝）
    """
    enhanced = copy.deepcopy(config)
    enhanced["factor_weights"] = ic_weights
    return enhanced
