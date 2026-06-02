"""
样本外优先选择模块。

P0-C: 目标函数 = OOS_Sharpe - penalty_weight × max(0, IS_Sharpe - OOS_Sharpe)
"""

from typing import Any, Dict, List, Tuple

import pandas as pd
from loguru import logger

from runner.common.utils import safe_float

_OOF_PENALTY_WEIGHT = 0.5


def compute_oos_priority_score(
    is_sharpe: float,
    oos_sharpe: float,
    penalty_weight: float = _OOF_PENALTY_WEIGHT,
) -> float:
    """
    计算样本外优先的综合评分。

    公式：Score = OOS_Sharpe - penalty_weight × max(0, IS_Sharpe - OOS_Sharpe)

    当 IS >> OOS 时（过拟合），惩罚项增大，降低总分。
    当 OOS >= IS 时（样本外更好），惩罚为0，直接用 OOS。

    Args:
        is_sharpe: 样本内 Sharpe
        oos_sharpe: 样本外 Sharpe
        penalty_weight: 过拟合惩罚权重

    Returns:
        综合评分
    """
    overfit_gap = max(0.0, is_sharpe - oos_sharpe)
    return oos_sharpe - penalty_weight * overfit_gap


def select_best_by_oos_priority(
    strategy_name: str,
    grid_df: pd.DataFrame,
    oos_results_map: Dict[str, Dict[str, Any]],
    param_keys: List[str],
) -> Tuple[Dict[str, Any], float]:
    """
    按样本外优先原则选择最优参数。

    对 Top 10 样本内参数组合，计算样本外优先评分。

    Args:
        strategy_name: 策略名
        grid_df: 网格搜索结果
        oos_results_map: {参数签名: 样本外KPI}
        param_keys: 参数键名列表

    Returns:
        (最优参数, 最优评分)
    """
    best_score = -float("inf")
    best_params = {}

    for _, row in grid_df.head(10).iterrows():
        params = {k: row[k] for k in param_keys}
        param_sig = str(sorted(params.items()))

        is_sharpe = safe_float(row.get("sharpe", 0))
        oos_kpi = oos_results_map.get(param_sig, {})
        oos_sharpe = safe_float(oos_kpi.get("sharpe", 0))

        score = compute_oos_priority_score(is_sharpe, oos_sharpe)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score
