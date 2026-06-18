"""
持仓量衍生信号（OI Signal）— 因子模块。

核心逻辑：
  raw = sign(price_return) * (oi_change / oi_volatility)

  物理含义：
  - 价格↑ + OI↑ → 趋势加速 → 做多
  - 价格↓ + OI↑ → 趋势加速 → 做空
  - 价格↑ + OI↓ → 趋势减弱 → 反向（看反转）
  - 价格↓ + OI↓ → 趋势减弱 → 反向（看反转）

  通过 oi_volatility 归一化，使不同品种的信号可比。
  截断到 [-1, 1] 区间。

函数接口：
  compute_oi_signal(close, oi, window) → signal_array (与 close 等长)
"""

from __future__ import annotations

import numpy as np


def compute_oi_signal(
    close: np.ndarray,
    oi: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """计算持仓量衍生信号。

    Args:
        close: 收盘价序列
        oi: 持仓量序列
        window: OI 波动率滚动窗口（默认 20）

    Returns:
        signal: 与 close 等长，前 warmup 段为 0。
        连续信号 [-1, 1]:
          >0 做多趋势，<0 做空趋势。
    """
    n = len(close)
    if n == 0:
        return np.zeros(0, dtype=float)

    # 价格日收益
    price_ret = np.zeros(n, dtype=float)
    price_ret[1:] = np.diff(close) / np.where(close[:-1] != 0, close[:-1], 1.0)

    # OI 变化率
    oi_pct = np.zeros(n, dtype=float)
    oi_pct[1:] = np.diff(oi) / np.where(oi[:-1] != 0, oi[:-1], 1.0)

    # OI 滚动波动率
    s = pd_series(oi_pct)
    oi_vol = s.rolling(window=window, min_periods=window).std().to_numpy()
    # 用 oi_vol 的 60 日中位数做兜底（避免早期 oi_vol=0 时信号爆炸）
    oi_vol_safe = oi_vol.copy()
    finite_vol = oi_vol[np.isfinite(oi_vol) & (oi_vol > 1e-8)]
    if len(finite_vol) > 0:
        fallback = float(np.median(finite_vol))
    else:
        fallback = 1e-3
    oi_vol_safe = np.where(np.isfinite(oi_vol_safe) & (oi_vol_safe > 1e-8), oi_vol_safe, fallback)

    # 原始信号：价格方向 × OI 相对波动
    raw = np.sign(price_ret) * (oi_pct / oi_vol_safe)

    # 截断到 [-1, 1]
    signal = np.clip(raw, -1.0, 1.0)

    # 早期 warmup 段（无足够 oi_vol）置 0
    if n >= window:
        signal[:window] = 0.0
    else:
        signal[:] = 0.0

    return signal


def pd_series(arr: np.ndarray):
    """轻量级 pandas 包装（避免顶部 import）。"""
    import pandas as pd  # noqa: WPS433 (延迟导入)
    return pd.Series(arr)


__all__ = ["compute_oi_signal"]
