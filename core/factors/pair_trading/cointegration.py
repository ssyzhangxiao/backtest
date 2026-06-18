"""
滚动协整检验与对冲比估计。

核心：
  - rolling_ols_hedge_ratio: 滚动 OLS 回归估计 hedge ratio β
  - adf_pvalue: 单次 ADF 检验，返回 p-value
  - rolling_cointegration: 滚动窗口同时输出 β 和 ADF p-value
  - CointegrationResult: 标准化结果

性能注意：
  - 调仓日才重算（rebalance_interval 默认 20 bar ≈ 1 月）
  - 缓存：self._cache[(symbol_a, symbol_b)] = (hedge_ratio, pvalue)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    from statsmodels.tsa.stattools import adfuller
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


@dataclass
class CointegrationResult:
    """单对配对的协整检验结果。"""
    symbol_a: str
    symbol_b: str
    hedge_ratio: float      # β: spread = price_a - β * price_b
    pvalue: float           # ADF 检验 p-value
    valid: bool             # p-value < threshold (默认 0.05)
    n_obs: int              # 实际用于回归的样本数

    def __repr__(self) -> str:
        return (
            f"<CointegrationResult {self.symbol_a}/{self.symbol_b} "
            f"β={self.hedge_ratio:.3f} p={self.pvalue:.3f} "
            f"{'✓' if self.valid else '✗'}>"
        )


def rolling_ols_hedge_ratio(
    y: np.ndarray,
    x: np.ndarray,
    window: int,
) -> np.ndarray:
    """滚动 OLS 估计 hedge ratio β，其中 y = α + β * x + ε。

    Args:
        y: 价格序列 A（被解释变量）
        x: 价格序列 B（解释变量）
        window: 滚动窗口（默认 60-90）

    Returns:
        np.ndarray，与 y 等长。前 (window-1) 个位置为 NaN。
    """
    n = len(y)
    beta = np.full(n, np.nan, dtype=float)
    if n < window or not _HAS_STATSMODELS:
        return beta
    for i in range(window - 1, n):
        y_w = y[i - window + 1: i + 1]
        x_w = x[i - window + 1: i + 1]
        if np.any(np.isnan(y_w)) or np.any(np.isnan(x_w)):
            continue
        try:
            x_const = add_constant(x_w)
            model = OLS(y_w, x_const).fit()
            beta[i] = float(model.params[1])
        except Exception:
            beta[i] = np.nan
    return beta


def adf_pvalue(spread: np.ndarray, maxlag: int = 0) -> float:
    """对 spread 序列做 ADF 检验，返回 p-value。

    Args:
        spread: 价差序列
        maxlag: ADF 滞后阶数，0=自动选择

    Returns:
        p-value，若 statsmodels 不可用或检验失败返回 1.0（视为不协整）
    """
    if not _HAS_STATSMODELS:
        return 1.0
    s = spread[~np.isnan(spread)]
    if len(s) < 20:
        return 1.0
    try:
        result = adfuller(s, maxlag=maxlag, autolag="AIC")
        return float(result[1])  # 第二个返回值是 p-value
    except Exception:
        return 1.0


def rolling_cointegration(
    y: np.ndarray,
    x: np.ndarray,
    window: int,
    pvalue_threshold: float = 0.05,
) -> CointegrationResult:
    """单点协整检验：使用最近 window 个 bar 估计 β + ADF 检验。

    Returns:
        CointegrationResult（valid 字段由 pvalue < threshold 决定）
    """
    if len(y) < window or len(x) < window:
        return CointegrationResult(
            symbol_a="?", symbol_b="?",
            hedge_ratio=1.0, pvalue=1.0, valid=False, n_obs=len(y),
        )
    y_w = y[-window:]
    x_w = x[-window:]
    if np.any(np.isnan(y_w)) or np.any(np.isnan(x_w)):
        return CointegrationResult(
            symbol_a="?", symbol_b="?",
            hedge_ratio=1.0, pvalue=1.0, valid=False, n_obs=0,
        )
    # 1) OLS 估计 β
    beta = 1.0
    if _HAS_STATSMODELS:
        try:
            x_const = add_constant(x_w)
            model = OLS(y_w, x_const).fit()
            beta = float(model.params[1])
        except Exception:
            beta = 1.0
    # 2) ADF 检验
    spread = y_w - beta * x_w
    pval = adf_pvalue(spread)
    return CointegrationResult(
        symbol_a="?", symbol_b="?",
        hedge_ratio=beta, pvalue=pval,
        valid=(pval < pvalue_threshold),
        n_obs=window,
    )


def batch_rolling_cointegration(
    close_df: pd.DataFrame,
    pairs: Iterable[Tuple[str, str]],
    bar_idx: int,
    window: int,
    pvalue_threshold: float = 0.05,
) -> Dict[Tuple[str, str], CointegrationResult]:
    """批量检验多个配对在 bar_idx 时刻的协整性。

    Args:
        close_df: 列=品种，索引=bar 的 close 矩阵
        pairs: 待检验的配对列表 [(A, B), ...]
        bar_idx: 当前 bar 索引（取 bar_idx-window+1 到 bar_idx 的窗口）
        window: 滚动窗口
        pvalue_threshold: p-value 阈值

    Returns:
        {(A, B): CointegrationResult}
    """
    if bar_idx < window:
        return {}
    results: Dict[Tuple[str, str], CointegrationResult] = {}
    for a, b in pairs:
        if a not in close_df.columns or b not in close_df.columns:
            continue
        y = close_df[a].iloc[: bar_idx + 1].to_numpy(dtype=float)
        x = close_df[b].iloc[: bar_idx + 1].to_numpy(dtype=float)
        res = rolling_cointegration(y, x, window, pvalue_threshold)
        res.symbol_a = a
        res.symbol_b = b
        results[(a, b)] = res
    return results
