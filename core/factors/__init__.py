"""
因子模块。

提供因子评估、变换、筛选和新因子实现的统一框架。
"""

from .factor_evaluator import FactorEvaluator, FactorEvalResult
from .factor_transformer import FactorTransformer
from .factor_selector import FactorSelector
from .capital_flow import CapitalFlowFactor
from .term_structure import TermStructureFactor

__all__ = [
    "FactorEvaluator",
    "FactorEvalResult",
    "FactorTransformer",
    "FactorSelector",
    "CapitalFlowFactor",
    "TermStructureFactor",
]
