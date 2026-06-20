"""基差动量因子 — 四因子 CTA 升级（2026-06-19）。

核心逻辑：
  basis = far_close - close   （远月 - 近月；正值=backwardation，近月相对紧）
  basis_pct = basis / close   （标准化为百分比）
  basis_mom = basis_pct - basis_pct.shift(basis_window)   （基差的变化方向）
  raw = basis_mom / rolling_std(basis_mom, basis_window)  （归一化）
  signal = clip(raw, -1, 1)   （截断到 [-1, 1]）

物理含义：
  - 基差走强（basis_mom ↑）→ 近月相对更紧 → 近月价格上涨压力 → 做多
  - 基差走弱（basis_mom ↓）→ 近月相对更松 → 近月价格下跌压力 → 做空
  - 与 carry（期限结构绝对值）的差异：carry 看当前位置，basis_mom 看变化

依赖数据：
  - close: 主力合约收盘价
  - far_close: 次主力合约收盘价（无则返回零信号）

Usage::

    from core.factors.basis_momentum import compute_basis_momentum
    signal = compute_basis_momentum(close, far_close, basis_window=20)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_basis_momentum(
    close: np.ndarray,
    far_close: np.ndarray,
    basis_window: int = 20,
) -> np.ndarray:
    """计算基差动量信号。

    Args:
        close: 主力合约收盘价序列
        far_close: 次主力合约收盘价序列（无次主力时全填 np.nan）
        basis_window: 基差变化窗口（默认 20）

    Returns:
        signal: 与 close 等长，无 far_close 数据或 warmup 段为 0。
        连续信号 [-1, 1]:
          >0 做多（基差走强），<0 做空（基差走弱）。
    """
    n = len(close)
    if n == 0:
        return np.zeros(0, dtype=float)

    close_arr = np.asarray(close, dtype=float)
    far_arr = np.asarray(far_close, dtype=float) if far_close is not None else np.full(n, np.nan)

    # 远月数据缺失 → 返回零信号（品种无次主力，如已退市）
    if np.all(np.isnan(far_arr)):
        return np.zeros(n, dtype=float)

    # 用 pandas 计算（rolling/NaN 处理更稳健）
    close_s = pd.Series(close_arr)
    far_s = pd.Series(far_arr)

    # 基差 = 远月 - 近月（正值 = backwardation，近月相对紧）
    basis = far_s - close_s
    # 标准化为百分比
    basis_pct = basis / close_s.where(close_s > 0, 1.0)
    basis_pct = basis_pct.replace([np.inf, -np.inf], np.nan)

    # 基差变化：当前 vs basis_window 之前
    basis_mom = basis_pct - basis_pct.shift(basis_window)

    # 归一化：除以滚动 std
    std = basis_mom.rolling(window=basis_window, min_periods=basis_window).std()
    median = float(std.median()) if std.notna().any() else 1e-3
    if not np.isfinite(median) or median <= 0:
        median = 1e-3
    std_safe = std.replace(0, median).fillna(median)

    raw = basis_mom / std_safe
    signal = raw.clip(-1.0, 1.0).fillna(0.0)

    result = signal.to_numpy(dtype=float)
    # warmup 段置 0
    if n >= basis_window:
        result[:basis_window] = 0.0
    else:
        result[:] = 0.0

    return result


def compute_basis_momentum_series(
    close_series: pd.Series,
    far_close_series: pd.Series,
    basis_window: int = 20,
) -> pd.Series:
    """Series 输入版本（按日期对齐）。

    用于回测中调用：输入按日期索引的 Series，输出同索引的 signal Series。
    """
    if close_series is None or close_series.empty:
        return pd.Series(dtype=float)
    close_arr = close_series.values
    if far_close_series is None or far_close_series.empty:
        far_arr = np.full(len(close_arr), np.nan)
    else:
        # 按 close_series 索引对齐
        far_aligned = far_close_series.reindex(close_series.index)
        far_arr = far_aligned.values
    sig = compute_basis_momentum(close_arr, far_arr, basis_window=basis_window)
    return pd.Series(sig, index=close_series.index, name="basis_momentum")


__all__ = ["compute_basis_momentum", "compute_basis_momentum_series"]
