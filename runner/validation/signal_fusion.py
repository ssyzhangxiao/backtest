"""
多策略信号融合（固定权重 + 月度滚动夏普更新）。

组合信号 = 0.3*carry + 0.3*vol + 0.2*donchian + 0.1*momentum + 0.05*tsi + 0.05*pair

两种模式:
  - "fixed": 使用上述固定权重（默认）
  - "rolling_sharpe": 按月度滚动窗口的 Sharpe 比率动态更新权重

2026-06-13 重构：
  - 内部改用 UnifiedFactorPool 统一计算所有信号
  - 移除对 scripts/run_cta_batch 和 CTAExecutorBuilder 的依赖
  - 每品种每策略信号从 UnifiedFactorPool 获取，不再单独建 CTAExecutorBuilder
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.execution.factor_pool import UnifiedFactorPool
from core.execution.signal_abstraction import DEFAULT_CTA_WEIGHTS
from runner.common.utils import sanitize_filename

# ── 固定权重（与用户要求一致） ──
_DEFAULT_WEIGHTS: Dict[str, float] = DEFAULT_CTA_WEIGHTS.copy()

# 默认6策略及其参数
_DEFAULT_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "carry": {"lookback": 60, "entry_z": 1.2, "direction": "long_only", "ema_alpha": 0.3, "use_slope": True},
    "vol_mean_reversion": {"vol_window": 20, "lookback": 252, "entry_z": 1.2, "vol_percentile": 0.7},
    "donchian_breakout": {"entry_lookback": 20, "atr_window": 14, "atr_entry_mult": 0.5, "trend_filter_ma": 60, "momentum_factor": 0.2},
    "momentum_ma": {"rsi_window": 14, "momentum_fast": 5, "momentum_mid": 20, "momentum_slow": 60},
    "tsi_garch": {"reg_window": 60, "min_obs": 30, "t_stat_threshold": 1.5, "cache_update_freq": 5},
    "pair_trading": {"lookback": 60, "entry_z": 2.0, "ols_window": 90, "adf_interval": 20},
}

# ── 滚动夏普参数 ──
_SHARPE_WINDOW = 21  # 月度滚动（约 21 个交易日）
_SHARPE_MIN_PERIODS = 63  # 最少 3 个月数据才开始更新
_WEIGHT_SMOOTH = 0.3  # 权重平滑系数（防止突变）

# 品种相关参数（兼容旧 signal_fusion 调用方）
_WARMUP = 30
_TEST_START = "2023-01-01"


def _extract_cta_returns(
    df_pb: pd.DataFrame,
    symbol: str,
    signal_series: pd.Series,
    initial_cash: float = 1_000_000,
    entry_threshold: float = 0.05,
) -> Optional[pd.Series]:
    """从 CTA 信号序列提取日收益率（模拟简单 CTA 执行）。

    用信号方向做每日调仓，信号绝对值需超过 entry_threshold 才开仓。
    返回日收益率序列。
    """
    if signal_series is None or len(signal_series) < 20:
        return None
    # 对齐日期
    df = df_pb.copy()
    df = df.sort_values("date").reset_index(drop=True)
    # 信号右对齐到 DataFrame 尾部
    sig = signal_series.values
    if len(sig) > len(df):
        sig = sig[-len(df):]
    elif len(sig) < len(df):
        pad = np.full(len(df) - len(sig), np.nan)
        sig = np.concatenate([pad, sig])

    df["signal"] = sig
    df["position"] = np.where(df["signal"] > entry_threshold, 1,
                              np.where(df["signal"] < -entry_threshold, -1, 0))
    df["ret"] = df["close"].pct_change()
    df["strategy_ret"] = df["position"].shift(1) * df["ret"]
    df["equity"] = (1 + df["strategy_ret"].fillna(0)).cumsum()
    returns = df["strategy_ret"].dropna()
    returns.name = symbol
    return returns if len(returns) > 20 else None


def _compute_rolling_sharpe_weights(
    returns_dict: Dict[str, pd.Series],
    window: int = _SHARPE_WINDOW,
    min_periods: int = _SHARPE_MIN_PERIODS,
) -> Dict[str, np.ndarray]:
    """按月度滚动窗口计算各策略 Sharpe 并归一化为权重。"""
    dfs = []
    for name, sr in returns_dict.items():
        sr = sr.copy()
        sr.name = name
        dfs.append(sr)
    aligned = pd.concat(dfs, axis=1, join="inner").dropna()
    if len(aligned) < min_periods + window:
        return {}

    rolling_sharpe = aligned.rolling(window, min_periods=min_periods).apply(
        lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 1e-10 else 0.0,
        raw=False,
    )
    rolling_sharpe = rolling_sharpe.fillna(0.0)

    weights_dict: Dict[str, np.ndarray] = {}
    for col in rolling_sharpe.columns:
        s = rolling_sharpe[col].values
        s_pos = np.maximum(s, 0.0)
        weights_dict[col] = s_pos
    return weights_dict


def run_signal_fusion(
    symbols: List[str],
    strategies: Optional[Dict[str, Dict[str, Any]]] = None,
    weights: Optional[Dict[str, float]] = None,
    mode: str = "fixed",
    entry_threshold: float = 0.05,
    initial_cash: float = 1_000_000,
    full_start: str = "2020-01-01",
    test_start: str = "2023-01-01",
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    主入口：多策略信号融合。

    使用 UnifiedFactorPool 统一计算所有 CTA 信号，
    不再依赖旧的 CTAExecutorBuilder 和 scripts/run_cta_batch 数据加载。

    Args:
        symbols: 品种列表
        strategies: 策略名→参数 dict。默认 _DEFAULT_STRATEGIES
        weights: 固定权重。默认 _DEFAULT_WEIGHTS
        mode: "fixed" 或 "rolling_sharpe"
        entry_threshold: 入场阈值
        initial_cash: 初始资金
        full_start: 全量数据起始
        test_start: 回测起始
        output_dir: 输出目录

    Returns:
        汇总 DataFrame: 每行一个品种，含各策略指标 + 融合指标
    """
    strategies = strategies or _DEFAULT_STRATEGIES
    weights = weights or _DEFAULT_WEIGHTS

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    # 每个品种创建一次 UnifiedFactorPool（内部缓存）
    pool = UnifiedFactorPool()

    for symbol in symbols:
        logger.info(f"[信号融合] 品种: {symbol} 模式: {mode}")

        # 通过 core.data.DataLoader（规则 17 统一数据加载器）加载数据
        try:
            from core.data import DataLoader
            loader = DataLoader(data_source="tqsdk", symbols=[symbol])
            loader.load_data(show_progress=False)
            loader.identify_dominant_contracts()
            loader.build_continuous_series()
            df = loader.get_pybroker_df()
        except Exception:
            # fallback: 尝试 CSV 模式
            try:
                from core.data import DataLoader
                import os
                data_dir = os.environ.get("DATA_DIR", "data")
                loader = DataLoader(data_source="csv", data_dir=data_dir)
                loader.load_data(show_progress=False)
                df = loader.get_pybroker_df()
            except Exception:
                logger.warning(f"  {symbol}: 无法加载数据（tqsdk/csv 均失败）")
                continue

        if df is None or len(df) < 100:
            logger.warning(f"  {symbol}: 数据不足")
            continue

        # 用 UnifiedFactorPool 一次性计算所有信号
        try:
            signal_df = pool.compute_all(df, symbol, strategy_params=strategies)
        except Exception as e:
            logger.warning(f"  {symbol}: 因子计算失败 {e}")
            continue

        returns_dict: Dict[str, pd.Series] = {}
        metrics_dict: Dict[str, Dict[str, float]] = {}

        for sname in strategies:
            if sname not in signal_df.columns:
                continue
            sr = _extract_cta_returns(
                df, symbol, signal_df[sname],
                initial_cash=initial_cash,
                entry_threshold=entry_threshold,
            )
            if sr is not None and len(sr) > 20:
                returns_dict[sname] = sr
                metrics_dict[sname] = {
                    "total_return": float(sr.sum()),
                    "sharpe": float(sr.mean() / sr.std() * np.sqrt(252)) if sr.std() > 1e-10 else 0.0,
                    "vol": float(sr.std() * np.sqrt(252)),
                    "trades": len(sr),
                }

        if not returns_dict:
            logger.warning(f"  {symbol}: 无可用策略数据")
            continue

        # ── 融合 ──
        if mode == "rolling_sharpe" and len(returns_dict) > 1:
            W = _compute_rolling_sharpe_weights(returns_dict)
            if W and len(next(iter(W.values()))) > 10:
                aligned = pd.concat(
                    [sr.copy().rename(name) for name, sr in returns_dict.items()],
                    axis=1, join="inner",
                ).dropna()
                combined_rets = np.zeros(len(aligned))
                for i in range(len(aligned)):
                    total_w = 0.0
                    for name in aligned.columns:
                        w_i = W.get(name, np.zeros(len(aligned)))
                        if i < len(w_i) and w_i[i] > 0:
                            combined_rets[i] += aligned.iloc[i][name] * w_i[i]
                            total_w += w_i[i]
                    if total_w > 1e-10:
                        combined_rets[i] /= total_w
                combined_returns = pd.Series(combined_rets, index=aligned.index)
            else:
                logger.warning(f"  {symbol}: 滚动 Sharpe 数据不足，回退固定权重")
                combined_returns = _combine_fixed(returns_dict, weights)
        else:
            combined_returns = _combine_fixed(returns_dict, weights)

        # ── 融合指标 ──
        row: Dict[str, Any] = {"symbol": symbol, "mode": mode}
        for sname in strategies:
            m = metrics_dict.get(sname, {})
            row[f"{sname}_return"] = round(m.get("total_return", 0) * 100, 2)
            row[f"{sname}_sharpe"] = round(m.get("sharpe", 0), 2)

        if combined_returns is not None and len(combined_returns) > 5:
            row["fusion_return_pct"] = round(float(combined_returns.sum()) * 100, 2)
            row["fusion_sharpe"] = round(
                float(combined_returns.mean() / combined_returns.std() * np.sqrt(252)),
                2,
            ) if combined_returns.std() > 1e-10 else 0.0
            row["fusion_vol_pct"] = round(
                float(combined_returns.std() * np.sqrt(252) * 100), 2
            )
        else:
            row["fusion_return_pct"] = 0.0
            row["fusion_sharpe"] = 0.0
            row["fusion_vol_pct"] = 0.0

        all_rows.append(row)

        if output_dir:
            if combined_returns is not None and len(combined_returns) > 5:
                eq_df = pd.DataFrame({
                    "date": combined_returns.index,
                    "fusion_returns": combined_returns.values,
                    "fusion_equity": (1 + combined_returns).cumprod(),
                })
                eq_path = output_dir / f"fusion_{sanitize_filename(symbol)}.csv"
                eq_df.to_csv(eq_path, index=False)

    result_df = pd.DataFrame(all_rows)
    if output_dir:
        summary_path = output_dir / "signal_fusion_summary.csv"
        result_df.to_csv(summary_path, index=False)

    return result_df


def _combine_fixed(
    returns_dict: Dict[str, pd.Series],
    weights: Dict[str, float],
) -> pd.Series:
    """固定权重融合。"""
    aligned = pd.concat(
        [sr.copy().rename(name) for name, sr in returns_dict.items()],
        axis=1, join="inner",
    ).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)

    cols = [c for c in aligned.columns if c in weights]
    if not cols:
        return pd.Series(dtype=float)

    w_arr = np.array([weights[c] for c in cols])
    w_arr = w_arr / w_arr.sum()
    combined = aligned[cols].values @ w_arr
    return pd.Series(combined, index=aligned.index)


__all__ = ["run_signal_fusion", "_DEFAULT_WEIGHTS", "_DEFAULT_STRATEGIES"]
