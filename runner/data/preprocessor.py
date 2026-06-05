"""
数据预处理模块。

因子得分计算已迁移到 core/factors/basic_factors.py，
此处保留兼容导入。
"""

# 委托 core/factors/basic_factors，消除重复实现
from core.factors.basic_factors import compute_factor_scores_from_ohlcv  # noqa: F401

__all__ = ["compute_factor_scores_from_ohlcv"]
