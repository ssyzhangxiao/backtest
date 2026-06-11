"""多品种分散化分析、调仓决策分析、策略OOS对比表。

从 sections.py 拆分，按职责独立管理。
"""

from typing import Any, Dict, List, Optional

from core.report_builder.metrics import (
    TRADING_DAYS_PER_YEAR,
    RISK_FREE_RATE,
    annualized_return,
    build_kpi_card,
    calmar_ratio,
    compute_daily_returns,
    compute_drawdown,
    compute_sharpe,
    compute_volatility,
    get_strategy_label,
)


# ---------------------------------------------------------------------------
# 分散化图表数据
# ---------------------------------------------------------------------------


def _build_diversification_chart_data(
    diversification_data: Dict[str, Any],
) -> Dict[str, Any]:
    """构建分散化图表数据。"""
    if not diversification_data:
        return {}

    single_symbols = diversification_data.get("single_symbol_equities", {})
    multi_symbol = diversification_data.get("multi_symbol_equity", {})

    if not single_symbols and not multi_symbol:
        return {}

    normalized_equities = {}
    for name, data_item in single_symbols.items():
        dates = data_item.get("dates", [])
        equity = data_item.get("equity", [])
        if dates and equity and len(equity) > 0:
            first_val = equity[0] if equity[0] != 0 else 1
            normalized = [e / first_val for e in equity]
            normalized_equities[name] = {"dates": dates, "equity": normalized}

    if multi_symbol:
        dates = multi_symbol.get("dates", [])
        equity = multi_symbol.get("equity", [])
        if dates and equity and len(equity) > 0:
            first_val = equity[0] if equity[0] != 0 else 1
            normalized = [e / first_val for e in equity]
            normalized_equities["多品种组合"] = {
                "dates": dates,
                "equity": normalized,
            }

    performance_comparison = []
    for name, data_item in normalized_equities.items():
        equity = data_item.get("equity", [])
        if len(equity) < 2:
            continue
        total_return = (equity[-1] / equity[0] - 1) * 100
        daily_rets = compute_daily_returns(equity)
        ann_ret = (
            (sum(daily_rets) / len(daily_rets)) * TRADING_DAYS_PER_YEAR * 100
            if daily_rets
            else 0
        )
        vol = compute_volatility(daily_rets)
        sh = (ann_ret - RISK_FREE_RATE) / vol if vol != 0 else 0
        drawdown_seq = compute_drawdown(equity)
        max_drawdown = min(drawdown_seq) if drawdown_seq else 0

        performance_comparison.append(
            {
                "name": name,
                "total_return": round(total_return, 2),
                "annualized_return": round(ann_ret, 2),
                "volatility": round(vol, 2),
                "sharpe": round(sh, 4),
                "max_drawdown": round(max_drawdown, 2),
            }
        )

    corr_matrix_data = None
    if "correlation_matrix" in diversification_data:
        df_corr = diversification_data["correlation_matrix"]
        if hasattr(df_corr, "to_dict"):
            try:
                corr_matrix_data = df_corr.to_dict(orient="split")
            except Exception:
                pass

    return {
        "equity_curves": normalized_equities,
        "performance": performance_comparison,
        "correlation": corr_matrix_data,
    }


# ---------------------------------------------------------------------------
# 多品种分散化分析 HTML
# ---------------------------------------------------------------------------


def build_diversification_html(
    diversification_data: Dict[str, Any],
) -> str:
    """构建多品种分散化分析 HTML。"""
    if not diversification_data:
        return ""

    chart_data = _build_diversification_chart_data(diversification_data)
    if not chart_data:
        return ""

    performance_comparison = chart_data.get("performance", [])
    corr_matrix_data = chart_data.get("correlation")

    perf_rows = ""
    for item in performance_comparison:
        name = item["name"]
        tr = item["total_return"]
        ar = item["annualized_return"]
        vol = item["volatility"]
        sh = item["sharpe"]
        md = item["max_drawdown"]

        perf_rows += f'''
                <tr>
                    <td><strong>{name}</strong></td>
                    <td class="{"positive" if tr > 0 else "negative"}">{tr:+.2f}%</td>
                    <td class="{"positive" if ar > 0 else "negative"}">{ar:+.2f}%</td>
                    <td>{vol:.2f}%</td>
                    <td class="{"positive" if sh > 0.1 else ""}">{sh:.4f}</td>
                    <td class="{"negative" if md < -5 else ""}">{md:.2f}%</td>
                </tr>'''

    corr_html = ""
    if corr_matrix_data:
        headers = corr_matrix_data.get("columns", [])
        data_rows = corr_matrix_data.get("data", [])
        header_html = "<th></th>" + "".join([f"<th>{h}</th>" for h in headers])
        corr_rows = []
        for i, row in enumerate(data_rows):
            cells = [f"<td><strong>{headers[i]}</strong></td>"]
            for j, val in enumerate(row):
                cell = ""
                if isinstance(val, (int, float)):
                    color = ""
                    if val > 0.7:
                        color = "background-color: #ff6b6b; color: white;"
                    elif val > 0.5:
                        color = "background-color: #ffa726;"
                    elif val > 0.3:
                        color = "background-color: #ffee58;"
                    elif val < -0.5:
                        color = "background-color: #42a5f5; color: white;"
                    cell = f'<td style="{color}">{val:.2f}</td>'
                else:
                    cell = f"<td>{val}</td>"
                cells.append(cell)
            corr_rows.append("<tr>" + "".join(cells) + "</tr>")

        corr_html = f"""
                <div class="table-box" style="margin-top: 20px;">
                    <h4 style="margin-bottom: 10px;">品种相关性矩阵</h4>
                    <div class="table-wrapper">
                        <table class="data-table" style="font-size: 12px;">
                            <thead>
                                <tr>{header_html}</tr>
                            </thead>
                            <tbody>{"".join(corr_rows)}</tbody>
                        </table>
                    </div>
                </div>"""

    return f"""
            <div class="section-title">多品种分散化分析</div>
            <div class="section-desc">对比单品种与多品种组合的绩效表现，展示分散化效应</div>

            <div class="card">
                <div class="card-header">净值曲线对比（归一化）</div>
                <div id="diversificationEquityChart" style="height: 300px;"></div>
            </div>

            <div class="card" style="margin-top: 20px;">
                <div class="card-header">绩效对比</div>
                <div class="table-wrapper">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>名称</th>
                                <th>总收益</th>
                                <th>年化收益</th>
                                <th>波动率</th>
                                <th>夏普比率</th>
                                <th>最大回撤</th>
                            </tr>
                        </thead>
                        <tbody>{perf_rows}</tbody>
                    </table>
                </div>
            </div>

            {corr_html}
            """


# ---------------------------------------------------------------------------
# 调仓决策分析 HTML
# ---------------------------------------------------------------------------


def build_rebalance_html(
    rebalance_analysis: Optional[Dict[str, Any]],
) -> str:
    """构建调仓决策分析 HTML。"""
    if not rebalance_analysis:
        return ""

    rebalance_sections = []
    for key, analysis_data in rebalance_analysis.items():
        decisions = analysis_data.get("decisions", [])
        analysis = analysis_data.get("analysis", {})

        if not decisions:
            continue

        total_decisions = analysis.get("total_decisions", 0)
        winning_decisions = analysis.get("winning_decisions", 0)
        win_rate = analysis.get("win_rate", 0)
        avg_return = analysis.get("avg_return_5d", 0)

        decision_rows = []
        for d in decisions[:30]:
            direction_class = ""
            if d["direction"] == "多":
                direction_class = "positive"
            elif d["direction"] == "空":
                direction_class = "negative"

            winning_class = "positive" if d.get("winning") else "negative"

            decision_rows.append(f"""
                <tr>
                    <td>{d["date"]}</td>
                    <td class="{direction_class}">{d["direction"]}</td>
                    <td>{d["composite_score"]}</td>
                    <td>{d["position_pct"]}</td>
                    <td class="{winning_class}">{d["return_5d"]}%</td>
                </tr>""")

        rebalance_sections.append(f"""
            <div class="section-title">调仓决策分析 - {key}</div>
            <div class="two-col">
                {build_kpi_card("总决策次数", str(total_decisions), total_decisions)}
                {build_kpi_card("盈利决策", str(winning_decisions), winning_decisions)}
                {build_kpi_card("决策胜率", f"{win_rate}%", win_rate)}
                {build_kpi_card("平均5日收益", f"{avg_return}%", avg_return)}
            </div>
            <div class="table-box">
                <table class="data-table">
                    <thead>
                        <tr><th>日期</th><th>方向</th><th>综合得分</th><th>仓位比例</th><th>5日收益</th></tr>
                    </thead>
                    <tbody>{"".join(decision_rows)}</tbody>
                </table>
            </div>""")

    if rebalance_sections:
        return "\n".join(rebalance_sections)
    return ""


# ---------------------------------------------------------------------------
# 策略样本内/外绩效对比表
# ---------------------------------------------------------------------------


def build_strategy_oos_rows(
    strategy_names: List[str],
    strategies_data: Dict[str, Any],
    in_sample_dates: List[str],
    out_sample_dates: List[str],
) -> str:
    """构建所有策略样本内/外绩效对比表 HTML 行。"""
    from core.config.strategy_profiles import StrategyLibrary
    from core.report_builder.metrics import _short_label_from_description

    strategy_label_lib = StrategyLibrary()
    is_end_date = in_sample_dates[-1] if in_sample_dates else ""
    os_start_date = out_sample_dates[0] if out_sample_dates else ""

    rows = ""
    for name in strategy_names:
        sd = strategies_data[name]
        eq = sd.get("equity", [])
        dates = sd.get("dates", [])
        if "_" in name:
            _, sub = name.split("_", 1)
            profile = strategy_label_lib.get_profile(sub)
            label = _short_label_from_description(profile.description) if profile else name
        else:
            label = get_strategy_label(name, strategy_label_lib)
        if not eq or not dates or not in_sample_dates or not out_sample_dates:
            continue
        for split_label, split_eq, split_dates in [
            (
                "样本内",
                [
                    eq[i]
                    for i, d in enumerate(dates)
                    if i < len(eq) and d <= is_end_date
                ],
                [d for d in dates if d <= is_end_date],
            ),
            (
                "样本外",
                [
                    eq[i]
                    for i, d in enumerate(dates)
                    if i < len(eq) and d >= os_start_date
                ],
                [d for d in dates if d >= os_start_date],
            ),
        ]:
            if len(split_eq) < 10:
                continue
            split_rets = compute_daily_returns(split_eq)
            tr = (
                (split_eq[-1] - split_eq[0]) / split_eq[0] * 100
                if split_eq[0] != 0
                else 0
            )
            years_split = len(split_eq) / TRADING_DAYS_PER_YEAR
            ar = annualized_return(tr, years_split) if years_split > 0 else 0
            sh = compute_sharpe(split_rets)
            dd_seq = compute_drawdown(split_eq)
            dd = min(dd_seq) if dd_seq else 0
            ca = calmar_ratio(ar, dd)
            wr = (
                sum(1 for r in split_rets if r > 0) / len(split_rets) * 100
                if split_rets
                else 0
            )
            tc = len([1 for r in split_rets if abs(r) > 0.001])
            rows += f"""
            <tr>
                <td><strong>{label}</strong></td>
                <td>{split_label}</td>
                <td class="{"positive" if tr > 0 else "negative"}">{tr:+.2f}%</td>
                <td class="{"positive" if ar > 0 else "negative"}">{ar:+.2f}%</td>
                <td class="{"positive" if sh > 0.1 else ""}">{sh:.4f}</td>
                <td class="{"negative" if dd < -0.5 else ""}">{dd:.2f}%</td>
                <td class="{"positive" if ca > 0.1 else ""}">{ca:.4f}</td>
                <td>{wr:.1f}%</td>
                <td>{tc}</td>
            </tr>"""
    return rows
