"""
配对价差 z-score 与横截面聚合。

核心：
  - compute_spread_zscore: 给定 hedge_ratio 和价差历史，返回当前 z-score
  - rolling_pair_zscore_matrix: 滚动计算所有配对在所有 bar 上的 z-score
  - aggregate_pair_zscores_to_symbols: 把配对 z-score 聚合为每品种净得分

设计原则（避免方向二教训）：
  1. z-score 输出连续值（不是 ±1 饱和）
  2. 聚合使用绝对值+符号传播（不会因为品种排名极值导致信号退化为常数）
  3. 仅使用经过 ADF 检验的配对（valid=True）
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cointegration import CointegrationResult


def compute_spread_zscore(
    price_a: np.ndarray,
    price_b: np.ndarray,
    hedge_ratio: float,
    lookback: int,
) -> float:
    """计算价差的 z-score（当前 bar 处）。

    spread_t = price_a_t - β * price_b_t
    z_t = (spread_t - mean) / std

    Args:
        price_a: 价差历史（不含未来）
        price_b: 同上
        hedge_ratio: β
        lookback: 均值/标准差窗口

    Returns:
        z-score（float）；不足 lookback 时返回 0.0
    """
    if len(price_a) < lookback or len(price_b) < lookback:
        return 0.0
    a_w = price_a[-lookback:]
    b_w = price_b[-lookback:]
    if np.any(np.isnan(a_w)) or np.any(np.isnan(b_w)):
        return 0.0
    spread = a_w - hedge_ratio * b_w
    mu = np.mean(spread)
    sd = np.std(spread, ddof=1)
    if sd < 1e-12:
        return 0.0
    return float((spread[-1] - mu) / sd)


def rolling_pair_zscore_matrix(
    close_df: pd.DataFrame,
    cointegration_results: Dict[Tuple[str, str], CointegrationResult],
    lookback: int,
    end_bar: int,
) -> Dict[Tuple[str, str], float]:
    """批量计算所有有效配对在 end_bar 时刻的 z-score。

    Args:
        close_df: close 矩阵（列=品种，索引=bar）
        cointegration_results: {(A, B): CointegrationResult}，仅 valid=True 的会被使用
        lookback: z-score 窗口
        end_bar: 当前 bar 索引

    Returns:
        {(A, B): z-score}：仅 valid=True 且数据足够的配对
    """
    out: Dict[Tuple[str, str], float] = {}
    for (a, b), res in cointegration_results.items():
        if not res.valid:
            continue
        if a not in close_df.columns or b not in close_df.columns:
            continue
        if end_bar < lookback:
            continue
        y = close_df[a].iloc[: end_bar + 1].to_numpy(dtype=float)
        x = close_df[b].iloc[: end_bar + 1].to_numpy(dtype=float)
        z = compute_spread_zscore(y, x, res.hedge_ratio, lookback)
        out[(a, b)] = z
    return out


def aggregate_pair_zscores_to_symbols(
    pair_zscores: Dict[Tuple[str, str], float],
    symbols: Iterable[str],
    clip_abs: float = 3.0,
) -> Dict[str, float]:
    """把配对 z-score 聚合为每品种的净 z-score。

    规则（金融逻辑）：
      - 配对 (A, B) 中 z > 0 → spread 偏高 → A 应做空，B 应做多
        → symbol A 收到 -z（做空信号），symbol B 收到 +z（做多信号）
      - 多个配对聚合：等权平均（保留 magnitude，与方向二教训一致）
      - 最终裁剪到 [-clip_abs, clip_abs]，避免极端值

    Args:
        pair_zscores: {(A, B): z}
        symbols: 全部品种
        clip_abs: 裁剪阈值（默认 ±3σ）

    Returns:
        {symbol: net_z_score}，无配对的品种为 0.0
    """
    sym_set = set(symbols)
    contributions: Dict[str, List[float]] = {s: [] for s in sym_set}
    for (a, b), z in pair_zscores.items():
        # 价差偏高 → 卖 A 买 B
        if a in sym_set:
            contributions[a].append(-z)
        if b in sym_set:
            contributions[b].append(+z)
    out: Dict[str, float] = {}
    for s in sym_set:
        vals = contributions[s]
        if not vals:
            out[s] = 0.0
        else:
            avg = float(np.mean(vals))
            out[s] = float(np.clip(avg, -clip_abs, clip_abs))
    return out
