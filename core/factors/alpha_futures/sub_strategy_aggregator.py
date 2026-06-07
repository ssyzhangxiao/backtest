"""
P0 整改：从 OHLCV DataFrame 直接计算 5 子策略因子得分。

替代旧的 core/factors/basic_factors.py：
  - 不再使用简化的 compute_ts_momentum / compute_roll_yield / compute_alpha019 / compute_alpha032
  - 统一通过 FactorEngine.compute_all() 获取 24 个基础因子
  - 按 SUB_STRATEGY_FACTOR_GROUPS 分组，对同组因子做均值聚合，得到5子策略信号
  - 趋势/期限结构信号通过 utils.indicators.compute_atr 归一化
  - 输出字段名与原 basic_factors.compute_factor_scores_from_ohlcv 完全一致（兼容旧调用方）

字段对应：
  trend              ←  T_01..T_05 均值（ATR 归一化）
  term_structure     ←  TS_01..TS_03 均值（ATR 归一化）
  mean_reversion     ←  M_01..M_05 均值
  vol_breakout       ←  V_01..V_04 / H_01..H_05 均值
  composite_resonance←  R_01..R_05 / CF_01..CF_03 均值
  forward_return     ←  close.shift(-5) / close - 1

位置: core/factors/alpha_futures/sub_strategy_aggregator.py
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.indicators import compute_atr

from .factor_engine import FactorEngine
from .factor_registry import get_sub_strategy_factors
from .config import AlphaFuturesConfig

DEFAULT_FACTOR_NAMES = [
    "trend", "term_structure", "mean_reversion",
    "vol_breakout", "composite_resonance",
]


def _safe_clip(series: pd.Series, lo: float = -1.0, hi: float = 1.0) -> pd.Series:
    """安全 clip 到 [-1, 1]，NaN 填 0。"""
    return series.fillna(0.0).clip(lower=lo, upper=hi)


def _aggregate_group(factor_results: Dict[str, np.ndarray], names: List[str]) -> np.ndarray:
    """对同组因子取均值（缺失跳过），返回长度一致的一维数组。"""
    arrays: List[np.ndarray] = []
    for n in names:
        v = factor_results.get(n)
        if v is None:
            continue
        arr = np.asarray(v, dtype=float)
        if arr.ndim == 0:
            continue
        arrays.append(arr)
    if not arrays:
        return np.zeros(0)
    min_len = min(len(a) for a in arrays)
    if min_len == 0:
        return np.zeros(0)
    stacked = np.vstack([a[-min_len:] for a in arrays])
    return np.nanmean(stacked, axis=0)


def _to_series(values: np.ndarray, index: pd.Index) -> pd.Series:
    """将 np.ndarray 与目标索引对齐（长度不足时右对齐补 NaN）。"""
    if len(values) == 0:
        return pd.Series(np.zeros(len(index)), index=index)
    if len(values) == len(index):
        return pd.Series(values, index=index)
    # 右对齐：values[-len(index):]
    n = len(index)
    arr = np.full(n, np.nan, dtype=float)
    arr[-len(values):] = values[-n:]
    return pd.Series(arr, index=index)


def compute_sub_strategy_scores_from_ohlcv(
    ohlcv: pd.DataFrame,
    config: Optional[AlphaFuturesConfig] = None,
    atr_period: int = 14,
    atr_scaling: float = 10.0,
) -> pd.DataFrame:
    """
    从 OHLCV DataFrame 计算 5 子策略因子得分（替代 basic_factors.compute_factor_scores_from_ohlcv）。

    实现路径（P0 整改）：
      1. 通过 FactorEngine.compute_all() 一次性计算 24 个因子
      2. 按 SUB_STRATEGY_FACTOR_GROUPS 分组聚合
      3. 趋势 / 期限结构信号用 utils.indicators.compute_atr 归一化
      4. 输出字段名与旧接口完全一致

    Args:
        ohlcv: 含 close, high, low, volume 列的 DataFrame（按日期排序）
        config: AlphaFuturesConfig 实例，None 时使用默认值
        atr_period: ATR 归一化周期
        atr_scaling: ATR 归一化缩放系数（trend / term_structure 专用）。
                     经验值 10.0，对应 1 个 ATR ≈ 10 倍归一化单位；
                     调大可降低信号幅度，调小可放大。NaN 时使用 1e-8 兜底。

    Returns:
        含以下列的 DataFrame：
          - trend, term_structure, mean_reversion, vol_breakout, composite_resonance
          - forward_return
    """
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame(columns=DEFAULT_FACTOR_NAMES + ["forward_return"])

    df = ohlcv.copy().sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(
        np.zeros(len(df)), index=df.index,
    )

    # 1) 一次性计算 24 因子
    cfg = config or AlphaFuturesConfig()
    engine = FactorEngine(cfg)
    raw = {
        "close": close.to_numpy(),
        "open_price": df["open"].astype(float).to_numpy() if "open" in df.columns else close.to_numpy(),
        "high": high.to_numpy(),
        "low": low.to_numpy(),
        "open_interest": df["open_interest"].astype(float).to_numpy()
        if "open_interest" in df.columns else np.zeros(len(df)),
        "volume": volume.to_numpy(),
    }
    factor_results = engine.compute_all(raw)

    # 2) 按子策略分组聚合
    index = df.index
    grouped: Dict[str, np.ndarray] = {}
    for sname in DEFAULT_FACTOR_NAMES:
        names = get_sub_strategy_factors(sname)
        grouped[sname] = _aggregate_group(factor_results, names)

    # 3) 归一化处理：trend / term_structure 用 ATR 缩放
    atr = compute_atr(high, low, close, period=atr_period)
    close_safe = close.replace(0, np.nan)
    mom_scale = (atr / close_safe).fillna(0.05).clip(lower=0.005, upper=0.5)

    for sname, raw_arr in grouped.items():
        ser = _to_series(raw_arr, index)
        if sname in ("trend", "term_structure"):
            # ATR 归一化后 clip 到 [-1, 1]
            ser = (ser / (mom_scale * atr_scaling + 1e-8)).pipe(_safe_clip)
        else:
            ser = _safe_clip(ser)
        df[sname] = ser

    # 4) 前瞻收益（5日）
    df["forward_return"] = close.shift(-5) / close - 1.0

    return df
