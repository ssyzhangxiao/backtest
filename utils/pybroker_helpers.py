import pandas as pd


def safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    result = df[col]
    if isinstance(result, pd.DataFrame):
        return result.iloc[:, 0]
    return result