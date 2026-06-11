"""报告图表数据构建模块。

从 generator.py 拆分，负责构建图表数据、样本内/外净值等。
"""

from typing import Any, Dict, List, Optional, Tuple

from core.report_builder.metrics import (
    TRADING_DAYS_PER_YEAR,
    compute_daily_returns,
    compute_drawdown,
    compute_sharpe,
    compute_volatility,
    correlation_matrix,
    extract_monthly_returns,
    rolling_max_drawdown,
    rolling_sharpe,
    safe_float,
)
from core.report_builder.diversification import _build_diversification_chart_data


# ---------------------------------------------------------------------------
# 图表数据构建
# ---------------------------------------------------------------------------


def build_chart_data(
    strategy_names: List[str],
    strategies_data: Dict[str, Any],
    best_dates: List[str],
    best_equity: List[float],
    in_sample_dates: List[str],
    in_sample_equity: List[float],
    out_sample_dates: List[str],
    out_sample_equity: List[float],
    diversification_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """构建图表数据、样本内/外净值、分散化图表数据。

    Returns:
        (chart_data, in_sample_js, out_sample_js)
    """
    # ── 净值曲线 ──
    equity_js = {}
    for name in strategy_names:
        sd = strategies_data[name]
        eq = sd.get("equity", [])
        dates = sd.get("dates", [])
        first_eq = eq[0] if eq else 1
        equity_js[name] = {
            "dates": dates,
            "equity": [e / first_eq for e in eq] if eq else [],
            "label": sd.get("label", name),
        }

    # ── 回撤 ──
    all_drawdowns = {}
    for name, sd in strategies_data.items():
        eq = sd.get("equity", [])
        dates = sd.get("dates", [])
        if eq:
            dd_seq = compute_drawdown(eq)
            peak = eq[0]
            max_dd_val = 0.0
            max_dd_idx = 0
            max_dd_start_idx = 0
            current_peak_idx = 0
            for i, v in enumerate(eq):
                if v > peak:
                    peak = v
                    current_peak_idx = i
                dd_val = (v - peak) / peak * 100 if peak != 0 else 0
                if dd_val < max_dd_val:
                    max_dd_val = dd_val
                    max_dd_idx = i
                    max_dd_start_idx = current_peak_idx
            duration_days = (
                max_dd_idx - max_dd_start_idx if max_dd_idx > max_dd_start_idx else 0
            )
            all_drawdowns[name] = {
                "dates": dates,
                "drawdown": dd_seq,
                "max_dd_pct": round(max_dd_val, 2),
                "max_dd_date": dates[max_dd_idx] if max_dd_idx < len(dates) else "",
                "max_dd_start_date": dates[max_dd_start_idx]
                if max_dd_start_idx < len(dates)
                else "",
                "duration_days": duration_days,
            }

    main_dd = compute_drawdown(best_equity)
    main_daily_rets = compute_daily_returns(best_equity)

    # ── 滚动指标 ──
    all_rolling_sharpe = {}
    all_rolling_dd = {}
    for name, sd in strategies_data.items():
        eq = sd.get("equity", [])
        dates = sd.get("dates", [])
        if eq:
            daily_rets = compute_daily_returns(eq)
            rs_vals = rolling_sharpe(daily_rets, window=36)
            rd_vals = rolling_max_drawdown(eq, window_months=12)
            all_rolling_sharpe[name] = {
                "dates": dates[36 * 21 :] if len(dates) >= 36 * 21 else [],
                "values": rs_vals,
            }
            all_rolling_dd[name] = {
                "dates": dates[12 * 21 :] if len(dates) >= 12 * 21 else [],
                "values": rd_vals,
            }

    rolling_sharpe_vals = rolling_sharpe(main_daily_rets, window=36)
    rolling_dd_vals = rolling_max_drawdown(best_equity, window_months=12)

    # ── 月度热力图 ──
    monthly_returns = extract_monthly_returns(best_dates, best_equity)
    months_sorted = sorted(monthly_returns.keys())
    years_set = sorted(set(k[:4] for k in months_sorted))
    months_labels = [f"{m:02d}" for m in range(1, 13)]
    heatmap_data = [[None] * 12 for _ in range(len(years_set))]
    for yi, year in enumerate(years_set):
        for mi, month in enumerate(months_labels):
            key = f"{year}-{month}"
            if key in monthly_returns:
                heatmap_data[yi][mi] = round(monthly_returns[key], 2)

    all_heatmaps = {}
    for name, sd in strategies_data.items():
        eq = sd.get("equity", [])
        dates = sd.get("dates", [])
        if eq and dates:
            mr = extract_monthly_returns(dates, eq)
            ms = sorted(mr.keys())
            ys = sorted(set(k[:4] for k in ms))
            hd = [[None] * 12 for _ in range(len(ys))]
            for yi2, year in enumerate(ys):
                for mi2, month in enumerate(months_labels):
                    key = f"{year}-{month}"
                    if key in mr:
                        hd[yi2][mi2] = round(mr[key], 2)
            all_heatmaps[name] = {"data": hd, "years_set": ys}

    # ── 风险收益散点 ──
    risk_return = []
    for name in strategy_names:
        sd = strategies_data[name]
        eq = sd.get("equity", [])
        rets = compute_daily_returns(eq)
        ann_ret = (sum(rets) / len(rets)) * TRADING_DAYS_PER_YEAR * 100 if rets else 0
        ann_vol = compute_volatility(rets)
        risk_return.append(
            {
                "name": sd.get("label", name),
                "key": name,
                "ann_return": round(ann_ret, 2),
                "ann_volatility": round(ann_vol, 2),
                "sharpe": safe_float(sd.get("metrics", {}).get("sharpe", 0)),
            }
        )

    # ── 相关性矩阵 ──
    corr_names, corr_matrix = correlation_matrix(strategies_data)

    # ── 样本内/外净值 ──
    in_sample_js = {}
    if in_sample_equity:
        first_is = in_sample_equity[0] if in_sample_equity[0] != 0 else 1
        in_sample_js = {
            "dates": in_sample_dates,
            "equity": [e / first_is for e in in_sample_equity],
        }
    out_sample_js = {}
    if out_sample_equity:
        first_os = out_sample_equity[0] if out_sample_equity[0] != 0 else 1
        out_sample_js = {
            "dates": out_sample_dates,
            "equity": [e / first_os for e in out_sample_equity],
        }

    all_is_equity = {}
    all_os_equity = {}
    is_end_date = in_sample_dates[-1] if in_sample_dates else ""
    os_start_date = out_sample_dates[0] if out_sample_dates else ""
    for name, sd in strategies_data.items():
        eq = sd.get("equity", [])
        dates = sd.get("dates", [])
        if eq and dates:
            is_eq, is_dt = [], []
            os_eq, os_dt = [], []
            for i, d in enumerate(dates):
                if i < len(eq):
                    if in_sample_dates and d <= is_end_date:
                        is_eq.append(eq[i])
                        is_dt.append(d)
                    if out_sample_dates and d >= os_start_date:
                        os_eq.append(eq[i])
                        os_dt.append(d)
            if is_eq:
                first_is_v = is_eq[0] if is_eq[0] != 0 else 1
                all_is_equity[name] = {
                    "dates": is_dt,
                    "equity": [e / first_is_v for e in is_eq],
                }
            if os_eq:
                first_os_v = os_eq[0] if os_eq[0] != 0 else 1
                all_os_equity[name] = {
                    "dates": os_dt,
                    "equity": [e / first_os_v for e in os_eq],
                }

    # ── 分散化图表数据 ──
    diversification_chart_data = _build_diversification_chart_data(diversification_data)

    chart_data = {
        "equity_curves": equity_js,
        "main_drawdown": {
            "dates": best_dates,
            "drawdown": main_dd,
        },
        "all_drawdowns": all_drawdowns,
        "risk_return": risk_return,
        "heatmap_data": heatmap_data,
        "all_heatmaps": all_heatmaps,
        "years_set": years_set,
        "rolling_sharpe": {
            "dates": best_dates[36 * 21 :] if len(best_dates) >= 36 * 21 else [],
            "values": rolling_sharpe_vals,
        },
        "all_rolling_sharpe": all_rolling_sharpe,
        "rolling_dd": {
            "dates": best_dates[12 * 21 :] if len(best_dates) >= 12 * 21 else [],
            "values": rolling_dd_vals,
        },
        "all_rolling_dd": all_rolling_dd,
        "all_is_equity": all_is_equity,
        "all_os_equity": all_os_equity,
        "correlation": {
            "names": corr_names,
            "matrix": corr_matrix,
        },
        "diversification": diversification_chart_data,
    }

    return chart_data, in_sample_js, out_sample_js
