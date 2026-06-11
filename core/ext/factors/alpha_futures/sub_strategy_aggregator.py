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

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.indicators import compute_atr

from .factor_engine import FactorEngine
from .factor_registry import get_sub_strategy_factors
from .config import AlphaFuturesConfig

DEFAULT_FACTOR_NAMES = [
    "trend",
    "term_structure",
    "mean_reversion",
    "vol_breakout",
    "composite_resonance",
]


def _safe_clip(series: pd.Series, lo: float = -1.0, hi: float = 1.0) -> pd.Series:
    """安全 clip 到 [-1, 1]，NaN 填 0。"""
    return series.fillna(0.0).clip(lower=lo, upper=hi)


def _aggregate_group(
    factor_results: Dict[str, np.ndarray], names: List[str]
) -> np.ndarray:
    """对同组因子取均值（缺失跳过），返回长度一致的一维数组。

    数据缺失语义：
      - 若组内**所有因子均缺失**（None 或长度 0）→ 返回**空数组**（长度 0），
        由 _to_series 进一步处理为全 NaN 序列
      - 若组内**至少有一个因子产出**但其余为 NaN → np.nanmean 自动跳过 NaN
      - **禁止**用全 0 数组作为"无数据"标记，避免下游误判为"中性信号"
    """
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
        # 空数组语义：告诉调用方"本组无数据"，由 _to_series 填充为全 NaN
        return np.zeros(0)
    min_len = min(len(a) for a in arrays)
    if min_len == 0:
        return np.zeros(0)
    stacked = np.vstack([a[-min_len:] for a in arrays])
    return np.nanmean(stacked, axis=0)


def _to_series(values: np.ndarray, index: pd.Index) -> pd.Series:
    """将 np.ndarray 与目标索引对齐（长度不足时右对齐补 NaN）。

    空输入语义：
      - 若 values 长度为 0（来自 _aggregate_group 的"全无数据"信号），
        **返回全 NaN 序列**而非全 0，避免下游把"无数据"误判为"中性信号 0"
      - 若 values 长度等于 index → 直接对齐
      - 若 values 长度小于 index → 右对齐：values[-len(index):] 放尾部，前部补 NaN
      - 若 values 长度大于 index → 截取尾部
    """
    n = len(index)
    if len(values) == 0:
        # 全无数据 → 全 NaN（区分于"中性信号 0"）
        return pd.Series(np.full(n, np.nan, dtype=float), index=index)
    if len(values) == n:
        return pd.Series(values, index=index)
    # 右对齐：把 values 的尾部对齐到 index 末尾
    arr = np.full(n, np.nan, dtype=float)
    k = min(len(values), n)
    arr[-k:] = values[-k:]
    return pd.Series(arr, index=index)


def compute_sub_strategy_scores_from_ohlcv(
    ohlcv: pd.DataFrame,
    config: Optional[AlphaFuturesConfig] = None,
    atr_period: int = 14,
    atr_scaling: float = 10.0,
    strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
    param_window_blend: float = 0.3,
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
        strategy_params: 子策略参数 {strategy_name: params}，用于叠加参数化窗口动量
                        通道（best_params 真正生效的入口）。None 或缺失关键参数时
                        退化为纯因子库输出（与旧行为一致）。
        param_window_blend: 参数化窗口动量通道权重（0~1，0 表示关闭通道）。
                          默认 0.3（2026-06-10 选项 B 修复：从 0.1 恢复 0.3，
                          避免压过 T_01..T_05 多因子集成 + 保留 best_params
                          30% 边际贡献。算法已升级为 SMA 斜率，更鲁棒）。

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
    volume = (
        df["volume"].astype(float)
        if "volume" in df.columns
        else pd.Series(
            np.zeros(len(df)),
            index=df.index,
        )
    )

    # 1) 一次性计算 24 因子
    cfg = config or AlphaFuturesConfig()
    engine = FactorEngine(cfg)
    raw = {
        "close": close.to_numpy(),
        "open_price": df["open"].astype(float).to_numpy()
        if "open" in df.columns
        else close.to_numpy(),
        "high": high.to_numpy(),
        "low": low.to_numpy(),
        "open_interest": df["open_interest"].astype(float).to_numpy()
        if "open_interest" in df.columns
        else np.zeros(len(df)),
        "volume": volume.to_numpy(),
    }
    # 期限结构因子（TS_01/02/03）需要近月/远月价：
    #   near_price = 主力连续合约 close（已是近月）
    #   far_price  = spread_pairs 注入的远月合约收盘价（far_close）
    # 若数据源缺失 far_close，传 None 让 TS_01/02/03 返回 NaN（不污染其他策略）
    if "far_close" in df.columns and df["far_close"].notna().any():
        raw["near_price"] = close.to_numpy()
        raw["far_price"] = df["far_close"].astype(float).to_numpy()
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
        # 3.1) 参数化窗口动量叠加通道（best_params 真正生效的入口）
        # 权重 0.1 让 T_01..T_05 多因子集成占主导，窗口动量做边际增强
        ser = _apply_param_window(
            ser,
            sname,
            close,
            strategy_params,
            mom_scale,
            blend=param_window_blend,
        )
        df[sname] = ser

    # 4) 前瞻收益（5日）
    df["forward_return"] = close.shift(-5) / close - 1.0

    return df


def _apply_param_window(
    ser: pd.Series,
    sname: str,
    close: pd.Series,
    strategy_params: Optional[Dict[str, Dict[str, Any]]],
    mom_scale: pd.Series,
    blend: float = 0.3,
) -> pd.Series:
    """
    在已有子策略信号上叠加一条"参数化 SMA 斜率"通道。

    这是 best_params（trend.window / term_structure.lookback / mean_reversion.short_window
    / vol_breakout.ma_window）真正生效的入口。

    算法（2026-06-10 选项 B 升级）：
      旧版：`window_ret = close / close.shift(w) - 1`（原始动量，噪声大）
      新版：`sma_slope  = (close - SMA(w)) / SMA(w)`（价格相对 SMA 的偏离度）
      新版优势：SMA 自身做了一次平滑，信号噪声显著低于原始动量，
                且在不同品种/波动率上更稳定。

    设计要点：
      - 只在 strategy_params 显式提供窗口参数时才叠加（默认行为不变，零回归风险）
      - 归一化：sma_slope / mom_scale + tanh
      - blend 默认 0.3（2026-06-10 选项 B 修复：从 0.1 恢复 0.3，保留 best_params
        30% 边际贡献。E1 v2 实验证明 blend=0.1 会让 best_params 失效，
        avg Sharpe 反而下降 8.19%）
      - 叠加后重新 clip 到 [-1, 1]

    Args:
        ser: 现有子策略因子得分序列
        sname: 子策略名
        close: 收盘价序列
        strategy_params: 子策略参数 {strategy_name: params}
        mom_scale: ATR/close 缩放序列
        blend: 窗口 SMA 斜率通道权重（0~1，0 表示关闭通道）

    Returns:
        叠加后的子策略得分序列
    """
    if not strategy_params or blend <= 0:
        return ser
    sp = strategy_params.get(sname) or {}
    if not sp:
        return ser
    # 不同子策略的窗口参数键
    window_key_map = {
        "trend": "window",
        "term_structure": "lookback",
        "mean_reversion": "short_window",
        "vol_breakout": "ma_window",
        "composite_resonance": "window",
    }
    w = sp.get(window_key_map.get(sname, "window"))
    if w is None:
        return ser
    try:
        w = int(w)
    except (TypeError, ValueError):
        return ser
    if w < 2 or w >= len(close):
        return ser

    # SMA 斜率（2026-06-10 升级）：
    # 旧版：window_ret = (close / close.shift(w) - 1.0)            # 原始动量
    # 新版：sma_slope  = (close - sma(w)) / sma(w)                  # SMA 偏离度
    sma_w = close.rolling(window=w, min_periods=max(2, w // 2)).mean()
    sma_safe = sma_w.replace(0, np.nan)
    sma_slope = ((close - sma_w) / sma_safe).fillna(0.0)
    scaled = (sma_slope / (mom_scale + 1e-8)).clip(lower=-3.0, upper=3.0)
    # tanh 软饱和到 [-1, 1]
    window_signal = np.tanh(scaled).fillna(0.0)

    # 重新索引对齐（避免 index 不一致）
    window_signal = window_signal.reindex(ser.index, fill_value=0.0)

    # 加权混合：(1 - blend) * 原信号 + blend * SMA 斜率，再 clip
    blended = (1.0 - blend) * ser + blend * window_signal
    return blended.clip(lower=-1.0, upper=1.0).fillna(0.0)
