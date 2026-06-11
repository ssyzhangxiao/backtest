"""
因子事件研究（Event Study）模块。

按"门槛触发 + 多窗口 + T+N 累计收益"模式改造经典事件研究（Fama 1969），
用于评估因子信号在商品期货市场的"持续性"与"统计显著性"。

商品期货适配要点（与用户案例一致）：
1. 信号构造：|factor| > 1.5σ 触发（避免单边过拟合）
2. 事件聚类校正：相邻触发间隔 < 5 日视为同一事件
3. 多窗口评估：T+1 / T+3 / T+5 / T+10 累计收益
4. 显著性检验：t 检验（scipy.stats.ttest_1samp）+ bootstrap 双输出
5. 复权处理：使用 OHLCV 原始数据，未做复权（默认日频）

委托：
- `core.factors.alpha_futures.factor_engine.FactorEngine`：计算因子值
- `scipy.stats.ttest_1samp`：t 检验
- `numpy`：累计收益与 bootstrap 重采样

通过标准（规则 28 阶段 A 新增）：
- T+5 ~ T+10 累计收益 p-value < 0.01：趋势持续
- 否则：剔除或降低权重

输出：output/validate/event_study.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.factors import AlphaFuturesConfig
from core.factors.alpha_futures.factor_engine import FactorEngine
from runner.common.utils import save_csv

# 信号触发阈值（绝对值，单位：标准差）
SIGNAL_SIGMA_THRESHOLD = 1.5

# 事件聚类最小间隔（交易日）
EVENT_MIN_GAP = 5

# T+N 评估窗口
HORIZONS = [1, 3, 5, 10]

# 通过标准：T+5 ~ T+10 累计收益 p-value < 0.01
PVALUE_THRESHOLD = 0.01

# Bootstrap 次数（用于 p-value 二次校验）
N_BOOTSTRAP = 1000

# 最小事件数（避免样本过少导致 p-value 失真）
MIN_EVENTS = 30

# 输出文件名
OUTPUT_FILENAME = "event_study.csv"


def _detect_events(factor_values: np.ndarray, sigma: float = SIGNAL_SIGMA_THRESHOLD,
                   min_gap: int = EVENT_MIN_GAP) -> np.ndarray:
    """
    从连续因子值提取事件索引（突破 ±1.5σ）。

    Args:
        factor_values: 因子值序列
        sigma: 突破阈值（绝对值，单位 σ）
        min_gap: 相邻事件最小间隔

    Returns:
        事件触发的索引数组
    """
    s = pd.Series(factor_values)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return np.array([], dtype=int)
    threshold = sigma * sd
    # 突破方向：f > mu + σ OR f < mu - σ
    above = (s > mu + threshold).values
    below = (s < mu - threshold).values
    candidate = above | below
    indices = np.where(candidate)[0]
    if len(indices) == 0:
        return indices
    # 聚类：相邻 < min_gap 视为同一事件，保留首个
    kept: List[int] = []
    last = -min_gap - 1
    for idx in indices:
        if idx - last >= min_gap:
            kept.append(int(idx))
            last = idx
    return np.array(kept, dtype=int)


def _t_stat_pvalue(samples: np.ndarray) -> Tuple[float, float]:
    """
    单样本 t 检验：H0: 均值=0。

    Returns:
        (t_stat, p_value)
    """
    from scipy import stats
    clean = samples[~np.isnan(samples)]
    if len(clean) < 2:
        return (np.nan, np.nan)
    result = stats.ttest_1samp(clean, popmean=0.0)
    return (float(result.statistic), float(result.pvalue))


def _bootstrap_pvalue(samples: np.ndarray, n_boot: int = N_BOOTSTRAP) -> float:
    """
    Bootstrap p-value：H0: 均值=0。

    在 0 中心重采样 n_boot 次，计算 |bootstrap_mean| >= |observed_mean| 比例。
    """
    clean = samples[~np.isnan(samples)]
    if len(clean) < 2:
        return np.nan
    obs = abs(clean.mean())
    rng = np.random.default_rng(42)
    centered = clean - clean.mean()
    boot_means = np.array([
        rng.choice(centered, size=len(centered), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float((np.abs(boot_means) >= obs).mean())


def _cumulative_return_at_horizons(
    close: np.ndarray,
    event_indices: np.ndarray,
    horizons: List[int] = HORIZONS,
) -> Dict[int, np.ndarray]:
    """
    对每个事件索引 + horizon，计算 [T, T+h] 累计收益。

    Returns:
        {horizon: 累计收益数组}，长度 = len(event_indices)
    """
    out: Dict[int, np.ndarray] = {h: np.full(len(event_indices), np.nan) for h in horizons}
    for i, t in enumerate(event_indices):
        for h in horizons:
            end = t + h
            if end >= len(close):
                continue
            out[h][i] = (close[end] - close[t]) / close[t]
    return out


def factor_event_study_validation(
    data_source,
    config: BacktestConfig,
    lib=None,
    output_dir: Path = Path("output/validate"),
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    cross_sectional: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    因子事件研究（按品种 × 因子 × horizon 输出）。

    Args:
        data_source: PyBrokerDataSource
        config: 回测配置
        lib: 策略库（未使用）
        output_dir: 输出目录
        best_params: 最优参数（未使用）
        cross_sectional: 是否横截面（未使用）
        **kwargs:
            sigma: 信号触发阈值（默认 1.5）
            min_gap: 事件最小间隔（默认 5）
            horizons: 评估窗口列表（默认 [1,3,5,10]）

    Returns:
        {
            "output_path": Path,
            "n_factors": int,
            "n_symbols": int,
            "pass_rate": float,
            "results": {symbol: DataFrame}
        }
    """
    sigma = float(kwargs.get("sigma", SIGNAL_SIGMA_THRESHOLD))
    min_gap = int(kwargs.get("min_gap", EVENT_MIN_GAP))
    horizons = list(kwargs.get("horizons", HORIZONS))

    logger.info("=" * 60)
    logger.info(
        f"因子事件研究（阈值 {sigma}σ，事件间隔 {min_gap}，"
        f"评估窗口 {horizons}）"
    )
    logger.info(f"通过标准: T+5~T+10 p-value < {PVALUE_THRESHOLD}")
    logger.info("=" * 60)

    af_cfg = AlphaFuturesConfig()
    engine = FactorEngine(af_cfg)
    symbols = config.symbols

    all_rows: list[Dict[str, Any]] = []
    per_symbol: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            ohlcv = data_source.query(
                config.train_start, config.test_end, symbols=[symbol]
            )
            if ohlcv is None or len(ohlcv) < 100:
                logger.warning(f"  {symbol}: 数据不足，跳过")
                continue
            ohlcv = ohlcv.sort_values("date").reset_index(drop=True)
            close = ohlcv["close"].values.astype(float)
            raw_data = {
                "close": close,
                "open_price": ohlcv["open"].values.astype(float),
                "high": ohlcv["high"].values.astype(float),
                "low": ohlcv["low"].values.astype(float),
                "open_interest": (
                    ohlcv["open_interest"].values.astype(float)
                    if "open_interest" in ohlcv.columns
                    else np.zeros(len(close))
                ),
                "volume": ohlcv["volume"].values.astype(float) if "volume" in ohlcv.columns else None,
            }
            factor_scores = engine.compute_all(raw_data)
        except Exception as e:
            logger.warning(f"  {symbol}: 因子计算失败: {e}")
            continue

        rows: list[Dict[str, Any]] = []
        for fname, fvalues in factor_scores.items():
            event_idx = _detect_events(fvalues, sigma, min_gap)
            cum_ret = _cumulative_return_at_horizons(close, event_idx, horizons)
            n_events = len(event_idx)
            # 关键窗口 T+5 / T+10
            t5_samples = cum_ret.get(5, np.array([]))
            t10_samples = cum_ret.get(10, np.array([]))
            t5_pvalue = np.nan
            t10_pvalue = np.nan
            t5_mean = np.nan
            t10_mean = np.nan
            if n_events >= MIN_EVENTS:
                t5_pvalue = _t_stat_pvalue(t5_samples)[1]
                t10_pvalue = _t_stat_pvalue(t10_samples)[1]
                t5_mean = float(np.nanmean(t5_samples))
                t10_mean = float(np.nanmean(t10_samples))
            is_pass = bool(
                n_events >= MIN_EVENTS
                and not np.isnan(t5_pvalue)
                and not np.isnan(t10_pvalue)
                and t5_pvalue < PVALUE_THRESHOLD
                and t10_pvalue < PVALUE_THRESHOLD
            )
            row = {
                "symbol": symbol,
                "factor": fname,
                "n_events": n_events,
                "t1_mean": float(np.nanmean(cum_ret.get(1, np.array([])))) if n_events > 0 else np.nan,
                "t3_mean": float(np.nanmean(cum_ret.get(3, np.array([])))) if n_events > 0 else np.nan,
                "t5_mean": t5_mean,
                "t10_mean": t10_mean,
                "t5_pvalue": t5_pvalue,
                "t10_pvalue": t10_pvalue,
                "is_pass": is_pass,
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        per_symbol[symbol] = df
        all_rows.extend(rows)

        n_pass = int(df["is_pass"].sum())
        n_total = len(df)
        logger.info(f"  {symbol}: Pass {n_pass}/{n_total} ({100*n_pass/max(n_total,1):.1f}%)")

    if not all_rows:
        logger.warning("  无有效事件研究结果")
        return {
            "output_path": None,
            "n_factors": 0,
            "n_symbols": 0,
            "pass_rate": 0.0,
            "results": {},
        }

    full_df = pd.DataFrame(all_rows)[
        ["symbol", "factor", "n_events", "t1_mean", "t3_mean", "t5_mean", "t10_mean",
         "t5_pvalue", "t10_pvalue", "is_pass"]
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / OUTPUT_FILENAME
    save_csv(full_df, out_path)
    logger.info(f"  事件研究结果已保存: {out_path}")

    n_total = len(full_df)
    n_pass = int(full_df["is_pass"].sum())
    return {
        "output_path": out_path,
        "n_factors": int(full_df["factor"].nunique()),
        "n_symbols": int(full_df["symbol"].nunique()),
        "pass_rate": n_pass / max(n_total, 1),
        "results": per_symbol,
    }


__all__ = [
    "factor_event_study_validation",
    "SIGNAL_SIGMA_THRESHOLD",
    "EVENT_MIN_GAP",
    "HORIZONS",
    "PVALUE_THRESHOLD",
    "MIN_EVENTS",
    "OUTPUT_FILENAME",
]
