"""仓单变化率因子 — 四因子 CTA 升级（2026-06-19）。

核心逻辑：
  change[t] = (receipt[t] - receipt[t-1]) / max(receipt[t-1], 1)
  std_change = rolling_std(change, window)
  raw = change / std_change
  signal = -clip(raw, -1, 1)   （仓单↑ → 做空；仓单↓ → 做多）

物理含义（基于库存周期）：
  - 仓单上升 → 库存累积 → 现货供应宽松 → 价格下行压力 → 做空
  - 仓单下降 → 库存去化 → 现货供应紧张 → 价格上行驱动 → 做多

与 oi_signal 的差异：
  - oi_signal 看持仓量变化（资金面/投机热度）
  - receipt_factor 看库存变化（基本面/供需）

依赖数据：
  - receipt_series: 仓单日度序列（pd.Series, 索引=日期）

Usage::

    from core.factors.receipt_factor import compute_receipt_factor, compute_receipt_factor_signal
    arr = compute_receipt_factor(receipt_array, window=20)
    series = compute_receipt_factor_signal(receipt_series, window=20)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.data.receipt_fetcher import get_receipt_change_signal


def compute_receipt_factor(
    receipt_series: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """计算仓单变化率信号（数组版）。

    Args:
        receipt_series: 仓单日度序列
        window: 滚动标准差窗口（默认 20）

    Returns:
        signal: 与 receipt 等长，前 warmup 段为 0。
        连续信号 [-1, 1]:
          >0 做多（仓单下降），<0 做空（仓单上升）。
    """
    n = len(receipt_series)
    if n == 0:
        return np.zeros(0, dtype=float)

    s = pd.Series(np.asarray(receipt_series, dtype=float))

    # 委托给模块级 get_receipt_change_signal（规则 17 不重复造轮子）
    return get_receipt_change_signal(s, window=window).to_numpy(dtype=float)


def compute_receipt_factor_signal(
    receipt_series: Optional[pd.Series],
    window: int = 20,
) -> pd.Series:
    """计算仓单变化率信号（Series 版）。

    Args:
        receipt_series: 仓单日度序列（按日期索引）
        window: 滚动标准差窗口（默认 20）

    Returns:
        signal_series: 与 receipt 等长索引，值为 [-1, 1]。
    """
    if receipt_series is None or receipt_series.empty:
        return pd.Series(dtype=float, name="receipt_factor")
    return get_receipt_change_signal(receipt_series, window=window).rename("receipt_factor")


__all__ = [
    "compute_receipt_factor",
    "compute_receipt_factor_signal",
]
