import pandas as pd

from .plots import PlotManager
from .metrics import MetricsCalculator


def safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """安全获取DataFrame列，自动处理单列DataFrame的情况。"""
    result = df[col]
    if isinstance(result, pd.DataFrame):
        return result.iloc[:, 0]
    return result
