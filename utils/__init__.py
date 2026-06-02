import pandas as pd

from .plots import PlotManager
from .metrics import MetricsCalculator
from .indicators import (
    compute_adx,
    compute_adx_series,
    compute_adx_components,
    compute_true_range,
)


def safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """安全获取DataFrame列，自动处理单列DataFrame的情况。"""
    result = df[col]
    if isinstance(result, pd.DataFrame):
        return result.iloc[:, 0]
    return result
