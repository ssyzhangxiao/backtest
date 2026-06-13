"""
验证层：样本外验证与鲁棒性测试。

提供训练/测试分割、蒙特卡洛、Bootstrap、因子IC稳定性四种验证方法。
委托 core/validation/ 和 core/engine/ 的公共接口，不重复实现。
"""

from typing import Callable, Dict

from runner.validation.train_test import task2_train_test_split
from runner.validation.monte_carlo import task3_monte_carlo
from runner.validation.bootstrap import run_bootstrap_validation
from runner.validation.factor_stability import factor_ic_stability_analysis
from runner.validation.factor_alpha24 import factor_alpha24_screening
from runner.validation.factor_alpha24 import factor_combo_ic_validation
from runner.validation.factor_review import factor_review_validation
from runner.validation.cross_sectional import cross_sectional_validation
from runner.validation.factor_adf import factor_adf_validation
from runner.validation.factor_prf import factor_prf_validation
from runner.validation.event_study import factor_event_study_validation
from runner.validation.standard_report import factor_standard_report_validation

_VALIDATOR_MAP: Dict[str, Callable] = {
    "train_test": task2_train_test_split,
    "monte_carlo": task3_monte_carlo,
    "bootstrap": run_bootstrap_validation,
    "factor_ic": factor_ic_stability_analysis,
    "factor_alpha24": factor_alpha24_screening,
    "factor_combo_ic": factor_combo_ic_validation,
    "factor_review": factor_review_validation,
    "cross_sectional": cross_sectional_validation,
    # 5 段式因子验证（规则 28 阶段 A 扩展）
    "factor_adf": factor_adf_validation,
    "factor_prf": factor_prf_validation,
    "event_study": factor_event_study_validation,
    "standard_report": factor_standard_report_validation,
}


def get_validator(method: str) -> Callable:
    """
    获取验证方法对应的函数。

    Args:
        method: 验证方法名称

    Returns:
        验证函数

    Raises:
        ValueError: 验证方法不存在
    """
    method_lower = method.lower()
    if method_lower not in _VALIDATOR_MAP:
        raise ValueError(f"未知验证方法: {method}，可用: {list(_VALIDATOR_MAP.keys())}")
    return _VALIDATOR_MAP[method_lower]


__all__ = [
    "task2_train_test_split",
    "task3_monte_carlo",
    "run_bootstrap_validation",
    "factor_ic_stability_analysis",
    "factor_alpha24_screening",
    "factor_review_validation",
    "cross_sectional_validation",
    "factor_adf_validation",
    "factor_prf_validation",
    "factor_event_study_validation",
    "factor_standard_report_validation",
    "run_signal_fusion",
    "run_parameter_plateau",
    "run_walk_forward",
    "get_validator",
]
