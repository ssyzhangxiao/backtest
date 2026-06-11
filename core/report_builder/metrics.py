"""
报告生成模块 — 计算工具函数。

提供绩效指标计算、回撤/波动率/夏普等衍生指标。
"""

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from loguru import logger

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.02


def safe_float(val: Any) -> float:
    """安全转 float，委托 runner.common.utils.safe_float。"""
    from runner.common.utils import safe_float as _sf

    return _sf(val)


def annualized_return(total_return_pct: float, years: float) -> float:
    """计算年化收益率。"""
    if years <= 0:
        return 0.0
    total_factor = 1 + total_return_pct / 100
    if total_factor <= 0:
        return -100.0
    return (total_factor ** (1 / years) - 1) * 100


def calmar_ratio(ann_return: float, max_dd_pct: float) -> float:
    """计算卡玛比率。"""
    dd = abs(max_dd_pct) if max_dd_pct != 0 else 0.01
    return ann_return / dd


def compute_drawdown(equity: List[float]) -> List[float]:
    """计算回撤序列（百分比）。"""
    if not equity:
        return []
    peak = equity[0]
    dd = []
    for v in equity:
        if v > peak:
            peak = v
        dd.append((v - peak) / peak * 100 if peak != 0 else 0)
    return dd


def compute_daily_returns(equity: List[float]) -> List[float]:
    """计算日收益率序列。"""
    rets = []
    for i in range(1, len(equity)):
        if equity[i - 1] != 0:
            rets.append((equity[i] - equity[i - 1]) / equity[i - 1])
        else:
            rets.append(0.0)
    return rets


def compute_volatility(daily_rets: List[float]) -> float:
    """计算年化波动率。"""
    if len(daily_rets) < 2:
        return 0.0
    mean = sum(daily_rets) / len(daily_rets)
    var = sum((r - mean) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
    return math.sqrt(var) * math.sqrt(TRADING_DAYS_PER_YEAR) * 100


def compute_sharpe(daily_rets: List[float]) -> float:
    """计算年化夏普比率。"""
    if len(daily_rets) < 2:
        return 0.0
    mean = sum(daily_rets) / len(daily_rets)
    var = sum((r - mean) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
    std = math.sqrt(var) if var > 0 else 1e-10
    ann_mean = mean * TRADING_DAYS_PER_YEAR
    ann_std = std * math.sqrt(TRADING_DAYS_PER_YEAR)
    return (ann_mean - RISK_FREE_RATE) / ann_std if ann_std != 0 else 0


def rolling_sharpe(daily_rets: List[float], window: int = 36) -> List[float]:
    """计算滚动夏普比率（窗口=36个月交易日≈756天）。"""
    window_days = window * 21
    result = []
    for i in range(window_days, len(daily_rets) + 1):
        slice_rets = daily_rets[i - window_days : i]
        result.append(compute_sharpe(slice_rets))
    return result


def rolling_max_drawdown(equity: List[float], window_months: int = 12) -> List[float]:
    """计算滚动最大回撤（窗口=12个月交易日≈252天）。"""
    window_days = window_months * 21
    result = []
    for i in range(window_days, len(equity) + 1):
        slice_eq = equity[i - window_days : i]
        peak = slice_eq[0]
        max_dd = 0.0
        for v in slice_eq:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100 if peak != 0 else 0
            if dd < max_dd:
                max_dd = dd
        result.append(max_dd)
    return result


def extract_monthly_returns(dates: List[str], equity: List[float]) -> Dict[str, float]:
    """提取月度收益率。"""
    monthly: Dict[str, Dict[str, float]] = {}
    for i, d in enumerate(dates):
        month_key = d[:7]
        if month_key not in monthly:
            monthly[month_key] = {"start_eq": equity[i]}
        monthly[month_key]["end_eq"] = equity[i]

    result = {}
    for k, v in monthly.items():
        if v["start_eq"] != 0:
            result[k] = (v["end_eq"] - v["start_eq"]) / v["start_eq"] * 100
        else:
            result[k] = 0.0
    return result


def correlation_matrix(
    strategy_data: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[List[float]]]:
    """计算策略间日收益率相关性矩阵。"""
    all_daily_rets = {}
    for name, sd in strategy_data.items():
        eq = sd.get("equity", [])
        all_daily_rets[name] = compute_daily_returns(eq) if eq else []

    names = sorted(all_daily_rets.keys())
    n = len(names)
    if n == 0:
        return [], []

    corr = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            x = all_daily_rets[names[i]]
            y = all_daily_rets[names[j]]
            min_len = min(len(x), len(y))
            if min_len < 2:
                corr[i][j] = 1.0 if i == j else 0.0
                continue
            xs = x[:min_len]
            ys = y[:min_len]
            mean_x = sum(xs) / min_len
            mean_y = sum(ys) / min_len
            cov = sum((xs[k] - mean_x) * (ys[k] - mean_y) for k in range(min_len))
            std_x = math.sqrt(sum((xk - mean_x) ** 2 for xk in xs))
            std_y = math.sqrt(sum((yk - mean_y) ** 2 for yk in ys))
            if std_x == 0 or std_y == 0:
                corr[i][j] = 0.0
            else:
                corr[i][j] = cov / (std_x * std_y)
    return names, corr


def calc_pl_ratio(metrics: Dict[str, Any]) -> float:
    """计算盈亏比。"""
    ap = safe_float(metrics.get("avg_profit_pct", 0))
    al = safe_float(metrics.get("avg_loss_pct", 1))
    return ap / abs(al) if abs(al) > 0 else 0


# ── CSV 读取工具 ──


def read_csv(path: Path) -> List[Dict[str, str]]:
    """读取 CSV 文件返回字典列表，失败返回空列表。"""
    try:
        with open(path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        logger.warning(f"读取文件失败: {path}, 错误: {e}")
        return []


def read_equity_csv(path: Path) -> Tuple[List[str], List[float]]:
    """读取净值 CSV，返回 (日期列表, 净值列表)。失败返回空元组。"""
    rows = read_csv(path)
    if not rows:
        return [], []
    dates = [r["date"] for r in rows]
    equity = [float(r["equity"]) for r in rows]
    return dates, equity


# ── 策略标签工具 ──


def _short_label_from_description(description: str) -> str:
    """从 StrategyProfile.description 中提取简短标签。"""
    if not description:
        return ""
    sep = description.find("。")
    return description[:sep] if sep > 0 else description


def _build_strategy_label(name: str, library: Any) -> str:
    """动态构造策略标签。"""
    if not name:
        return ""
    sub = name.split("_", 1)[1] if "_" in name else name
    profile = library.get_profile(sub)
    if profile is not None:
        return _short_label_from_description(profile.description)
    return name


def get_strategy_label(name: str, library: Any = None) -> str:
    """对外的薄封装：缺省构造一个 StrategyLibrary。"""
    from core.config.strategy_profiles import StrategyLibrary

    lib = library or StrategyLibrary()
    return _build_strategy_label(name, lib)


def build_kpi_card(label: str, value: str, numeric_val: float) -> str:
    """构建 KPI 卡片 HTML。"""
    cls = ""
    if numeric_val > 0:
        cls = "positive"
    elif numeric_val < 0:
        cls = "negative"
    return f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {cls}">{value}</div>
        </div>"""
