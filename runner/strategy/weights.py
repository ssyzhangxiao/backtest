"""
策略权重管理模块。

提供 E11 因子分析结果到全局因子权重的提取、归一化，
以及 IC 优化权重到配置字典的注入。
"""

import copy
from typing import Any, Dict, List

from loguru import logger


def extract_optimized_weights(e11_result: Dict[str, Any]) -> Dict[str, float]:
    """
    从 E11 因子分析结果中提取优化后的全局因子权重。

    输入结构预期（与 run_e11_factor_analysis 返回格式一致）：
        {
            "<symbol>": {
                "final_weights": {"<factor>": float, ...},
                # ... 其他 E11 字段（如 ic_per_factor、stability）
            },
            ...
        }
    提取每个 symbol 的 final_weights，对同一因子跨品种取算术平均，
    最后归一化使所有权重之和为 1.0。

    Args:
        e11_result: E11 因子分析结果字典，{symbol: {"final_weights": {factor: weight}}}

    Returns:
        归一化后的因子权重字典 {factor: weight}；输入为空或无 final_weights 字段时返回 {}
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
