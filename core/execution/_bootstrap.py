"""回测运行器 — Bootstrap 重采样与简化信号/指标计算。"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.engine._result_types import PyBrokerResult

logger = logging.getLogger(__name__)

try:
    import pybroker

    PYBROKER_AVAILABLE = True
except ImportError:
    PYBROKER_AVAILABLE = False


def bootstrap_metrics(
    runner, n_samples: Optional[int] = None
) -> Dict:
    """绩效指标 bootstrap 重采样。"""
    if runner._last_result is None:
        raise RuntimeError("请先调用 run()")

    n_samples = n_samples or runner.config.pybroker_bootstrap_samples

    if PYBROKER_AVAILABLE:
        try:
            return _bootstrap_pybroker(runner, n_samples)
        except Exception as e:
            logger.warning("PyBroker bootstrap 失败 (%s)，回退到 numpy 实现。", e)

    return _bootstrap_numpy(runner, n_samples)


def _bootstrap_pybroker(runner, _n_samples: int) -> Dict:
    """使用 PyBroker 内置 bootstrap。"""
    if not hasattr(runner, "_last_pb_result") or runner._last_pb_result is None:
        raise RuntimeError("没有可用的 PyBroker 回测结果")

    pb_result = runner._last_pb_result
    if not hasattr(pb_result, "bootstrap") or pb_result.bootstrap is None:
        raise RuntimeError(
            "回测结果中无 bootstrap 数据，请使用 calc_bootstrap=True"
        )

    bs = pb_result.bootstrap
    result = {}

    if hasattr(bs, "conf_intervals") and isinstance(
        bs.conf_intervals, pd.DataFrame
    ):
        ci = bs.conf_intervals
        for idx_val in ci.index:
            if isinstance(idx_val, tuple):
                metric_name, conf_level = idx_val
            else:
                metric_name, conf_level = str(idx_val), "value"
            row = ci.loc[idx_val]
            key = f"{metric_name} ({conf_level})"
            result[key] = {
                "ci_lower": round(float(row.get("lower", 0)), 4),
                "ci_upper": round(float(row.get("upper", 0)), 4),
            }
        return result

    for attr in ("sharpe", "drawdown", "profit_factor"):
        if hasattr(bs, attr):
            arr = getattr(bs, attr)
            if arr is not None and hasattr(arr, "__len__") and len(arr) > 0:
                arr_np = np.asarray(arr)
                result[attr] = {
                    "mean": round(float(np.mean(arr_np)), 4),
                    "std": round(float(np.std(arr_np)), 4),
                    "ci_lower": round(float(np.percentile(arr_np, 2.5)), 4),
                    "ci_upper": round(float(np.percentile(arr_np, 97.5)), 4),
                }
    return result


def _bootstrap_numpy(runner, n_samples: int) -> Dict:
    """numpy 自实现 bootstrap。"""
    equity = runner._last_result.equity_curve["equity"]
    daily_returns = equity.pct_change().dropna()

    if len(daily_returns) < 10:
        return {"error": "样本太少，无法 bootstrap"}

    n = len(daily_returns)
    rng = np.random.default_rng(42)

    metrics_samples: Dict[str, list] = {
        "sharpe": [],
        "total_return": [],
        "max_drawdown": [],
        "calmar": [],
        "win_rate": [],
    }

    actual_samples = n_samples
    if actual_samples > 50000:
        logger.info(
            "bootstrap n_samples=%d 较大，可能需要较长时间。", actual_samples
        )

    for _ in range(actual_samples):
        idx = rng.integers(0, n, size=n)
        ret_sample = daily_returns.iloc[idx].values
        eq_sample = equity.iloc[0] * np.cumprod(1 + np.insert(ret_sample, 0, 0))

        ann_factor = np.sqrt(252)
        sharpe = (np.mean(ret_sample) / max(np.std(ret_sample), 1e-8)) * ann_factor
        total_ret = (eq_sample[-1] - eq_sample[0]) / eq_sample[0]
        peak = np.maximum.accumulate(eq_sample)
        dd = np.min((eq_sample - peak) / peak)
        calmar = total_ret / abs(dd) if abs(dd) > 1e-10 else 0.0
        win_rate = np.mean(ret_sample > 0)

        metrics_samples["sharpe"].append(float(sharpe))
        metrics_samples["total_return"].append(float(total_ret))
        metrics_samples["max_drawdown"].append(float(dd))
        metrics_samples["calmar"].append(float(calmar))
        metrics_samples["win_rate"].append(float(win_rate))

    result = {}
    for key, vals in metrics_samples.items():
        arr = np.array(vals)
        result[key] = {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "ci_lower": round(float(np.percentile(arr, 2.5)), 4),
            "ci_upper": round(float(np.percentile(arr, 97.5)), 4),
        }

    runner._last_result.bootstrap_metrics = result
    return result


def generate_simple_signal(
    df: pd.DataFrame, idx: int, strategy_name: str, params: dict = None
) -> int:
    """简化信号生成（WalkForward fallback 引擎使用）。5子策略版本。"""
    if params is None:
        params = {}
    close = df["close"]
    i = idx

    if strategy_name == "trend":
        window = params.get("window", 20)
        if i < window:
            return 0
        ret = close.iloc[i] / close.iloc[max(0, i - window)] - 1
        signal = np.tanh(ret * 5)
        return 1 if signal > 0.2 else (-1 if signal < -0.2 else 0)

    elif strategy_name == "term_structure":
        lookback = params.get("lookback", 20)
        if i < lookback:
            return 0
        ma = close.iloc[max(0, i - lookback) : i + 1].mean()
        if ma <= 0:
            return 0
        spread_pct = (close.iloc[i] - ma) / ma * 100
        signal = np.tanh(-spread_pct / 3.0)
        return 1 if signal > 0.2 else (-1 if signal < -0.2 else 0)

    elif strategy_name == "mean_reversion":
        short_window = params.get("short_window", 7)
        if i < short_window:
            return 0
        delta_n = close.iloc[i] - close.iloc[max(0, i - short_window)]
        sign_val = -1 if delta_n > 0 else 1
        return sign_val if abs(delta_n / close.iloc[i]) > 0.01 else 0

    elif strategy_name == "vol_breakout":
        ma_window = params.get("ma_window", 7)
        if i < ma_window:
            return 0
        ma = close.iloc[max(0, i - ma_window) : i + 1].mean()
        deviation = ma - close.iloc[i]
        signal = np.tanh(deviation * 0.1)
        return 1 if signal > 0.2 else (-1 if signal < -0.2 else 0)

    elif strategy_name == "composite_resonance":
        window = params.get("window", 20)
        short_window = params.get("short_window", 7)
        if i < max(window, short_window):
            return 0
        ret = close.iloc[i] / close.iloc[max(0, i - window)] - 1
        trend_signal = np.tanh(ret * 5)
        delta_n = close.iloc[i] - close.iloc[max(0, i - short_window)]
        mr_signal = -1 if delta_n > 0 else 1
        composite = (trend_signal + mr_signal * 0.5) / 2.0
        return 1 if composite > 0.2 else (-1 if composite < -0.2 else 0)

    return 0


def compute_simple_metrics(
    equity: pd.Series, daily_returns: pd.Series
) -> Dict[str, float]:
    """计算简化绩效指标。"""
    if len(daily_returns) < 2 or len(equity) < 2:
        return {"error": "insufficient_data"}

    total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
    ann_factor = np.sqrt(252)
    sharpe = (daily_returns.mean() / max(daily_returns.std(), 1e-8)) * ann_factor
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    max_dd = dd.min()
    calmar = total_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0
    win_rate = (daily_returns > 0).mean()

    return {
        "total_return": round(total_return, 4),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(float(max_dd), 4),
        "max_drawdown_pct": round(float(max_dd) * 100, 2),
        "calmar": round(float(calmar), 3),
        "win_rate": round(float(win_rate), 4),
        "n_days": len(daily_returns),
        "final_equity": round(float(equity.iloc[-1]), 2),
    }
