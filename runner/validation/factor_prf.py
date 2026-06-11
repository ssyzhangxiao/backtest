"""
因子 PRF（Precision / Recall / Lift）分析。

将连续因子值离散化为 0/1 信号后，评估其对前瞻收益的预测力：
- Precision：信号触发后 T+5 收益为正的比例
- Recall：实际 T+5 收益为正的样本中，信号触发的比例
- Lift：信号组的正收益比例 - 全样本正收益比例（基线）
- F1：Precision 与 Recall 的调和平均

适用场景：发现"阈值效应"——连续因子 IC 弱，但突破特定阈值后预测力突显
（如用户案例：ATR 因子连续路径 IC=0.07 失败，离散路径 PRF 翻转）。

委托：
- `core.factors.alpha_futures.factor_engine.FactorEngine`：计算因子值
- `numpy.percentile`：阈值分位筛选
- `core.factors.operators`：标准分位数

通过标准（规则 28 阶段 A 新增）：
- Precision > 0.55 且 Lift > 0：离散信号有效，保留因子
- 否则：离散信号无效，回到因子变换（log/exp/cross/quantile）

输出：output/validate/factor_prf.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.factors import AlphaFuturesConfig
from core.factors.alpha_futures.factor_engine import FactorEngine
from runner.common.utils import save_csv

# 信号阈值（突破分位）：|factor| > upper_q OR |factor| < lower_q
DEFAULT_UPPER_QUANTILE = 0.80
DEFAULT_LOWER_QUANTILE = 0.20

# 前瞻收益窗口（T+5 视为短线验证窗口）
DEFAULT_FORWARD_PERIOD = 5

# 通过标准
PRECISION_THRESHOLD = 0.55  # 信号组中正收益样本占比
LIFT_THRESHOLD = 0.0  # 信号组需好于全样本基线
RECALL_MIN = 0.10  # 信号触发占比下限（避免极稀疏信号）

# 输出文件名
OUTPUT_FILENAME = "factor_prf.csv"


def _signal_from_quantile(
    values: np.ndarray,
    upper_q: float = DEFAULT_UPPER_QUANTILE,
    lower_q: float = DEFAULT_LOWER_QUANTILE,
) -> np.ndarray:
    """
    因子值 → 0/1 信号（双向突破）。

    |value| > upper_q 分位 或 |value| < lower_q 分位 → 1
    其他 → 0

    Returns:
        bool 数组（True 表示信号触发）
    """
    s = pd.Series(values)
    abs_vals = s.abs()
    upper = abs_vals.quantile(upper_q)
    lower = abs_vals.quantile(lower_q)
    return ((abs_vals >= upper) | (abs_vals <= lower)).values


def _compute_prf(signal: np.ndarray, forward_ret: np.ndarray) -> Dict[str, float]:
    """
    计算 Precision / Recall / Lift / F1 / N_triggers。

    Args:
        signal: bool 数组（True=信号触发）
        forward_ret: 前瞻收益（连续值）

    Returns:
        {precision, recall, lift, f1, n_triggers, base_rate, n_total}
    """
    sig = signal.astype(bool)
    ret = forward_ret.astype(float)
    # 去除 NaN
    valid = ~np.isnan(ret)
    sig = sig[valid]
    ret = ret[valid]
    if len(ret) == 0 or sig.sum() == 0:
        return {
            "precision": np.nan,
            "recall": np.nan,
            "lift": np.nan,
            "f1": np.nan,
            "n_triggers": 0,
            "base_rate": np.nan,
            "n_total": 0,
            "is_pass": False,
        }
    n_total = len(ret)
    base_rate = float((ret > 0).mean())
    # 信号组
    sig_pos = (ret[sig] > 0).sum()
    n_trig = int(sig.sum())
    precision = sig_pos / n_trig
    # 实际正收益样本中
    actual_pos = (ret > 0).sum()
    recall = sig_pos / actual_pos if actual_pos > 0 else 0.0
    lift = precision - base_rate
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    is_pass = bool(precision > PRECISION_THRESHOLD and lift > LIFT_THRESHOLD and recall >= RECALL_MIN)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "lift": float(lift),
        "f1": float(f1),
        "n_triggers": n_trig,
        "base_rate": float(base_rate),
        "n_total": int(n_total),
        "is_pass": is_pass,
    }


def factor_prf_validation(
    data_source,
    config: BacktestConfig,
    lib=None,
    output_dir: Path = Path("output/validate"),
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    cross_sectional: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    因子 PRF 分析（按品种 × 因子输出）。

    Args:
        data_source: PyBrokerDataSource
        config: 回测配置
        lib: 策略库（未使用）
        output_dir: 输出目录
        best_params: 最优参数（未使用）
        cross_sectional: 是否横截面（未使用）
        **kwargs:
            forward_period: 前瞻收益周期（默认 5）
            upper_quantile: 信号上分位（默认 0.80）
            lower_quantile: 信号下分位（默认 0.20）

    Returns:
        {
            "output_path": Path,
            "n_factors": int,
            "n_symbols": int,
            "pass_rate": float,
            "results": {symbol: DataFrame}
        }
    """
    forward_period = int(kwargs.get("forward_period", DEFAULT_FORWARD_PERIOD))
    upper_q = float(kwargs.get("upper_quantile", DEFAULT_UPPER_QUANTILE))
    lower_q = float(kwargs.get("lower_quantile", DEFAULT_LOWER_QUANTILE))

    logger.info("=" * 60)
    logger.info(
        f"因子 PRF 分析（前瞻 T+{forward_period}，"
        f"阈值分位 {lower_q}/{upper_q}）"
    )
    logger.info(
        f"通过标准: Precision > {PRECISION_THRESHOLD} AND Lift > {LIFT_THRESHOLD}"
    )
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
            # 前瞻收益
            forward_ret = np.full(len(close), np.nan)
            forward_ret[:-forward_period] = (
                close[forward_period:] - close[:-forward_period]
            ) / close[:-forward_period]
            factor_scores = engine.compute_all(raw_data)
        except Exception as e:
            logger.warning(f"  {symbol}: 因子计算失败: {e}")
            continue

        rows: list[Dict[str, Any]] = []
        for fname, fvalues in factor_scores.items():
            signal = _signal_from_quantile(fvalues, upper_q, lower_q)
            prf = _compute_prf(signal, forward_ret)
            prf["symbol"] = symbol
            prf["factor"] = fname
            rows.append(prf)
        df = pd.DataFrame(rows)
        per_symbol[symbol] = df
        all_rows.extend(rows)

        n_pass = int(df["is_pass"].sum())
        n_total = len(df)
        logger.info(f"  {symbol}: Pass {n_pass}/{n_total} ({100*n_pass/max(n_total,1):.1f}%)")

    if not all_rows:
        logger.warning("  无有效 PRF 结果")
        return {
            "output_path": None,
            "n_factors": 0,
            "n_symbols": 0,
            "pass_rate": 0.0,
            "results": {},
        }

    full_df = pd.DataFrame(all_rows)[
        ["symbol", "factor", "precision", "recall", "lift", "f1",
         "n_triggers", "base_rate", "n_total", "is_pass"]
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / OUTPUT_FILENAME
    save_csv(full_df, out_path)
    logger.info(f"  PRF 结果已保存: {out_path}")

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
    "factor_prf_validation",
    "PRECISION_THRESHOLD",
    "LIFT_THRESHOLD",
    "DEFAULT_FORWARD_PERIOD",
    "OUTPUT_FILENAME",
]
