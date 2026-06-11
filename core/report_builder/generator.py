"""报告生成模块 — 主逻辑。

包含 generate_report() 主入口，委托 sections/evaluation 子模块构建各区块。
"""

import json
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from core.config.strategy_profiles import StrategyLibrary
from core.report_builder.data_collector import load_strategies_data
from core.report_builder.evaluation import build_dynamic_evaluation_html
from core.report_builder.html_template import HTML_TEMPLATE
from core.report_builder.metrics import (
    TRADING_DAYS_PER_YEAR,
    annualized_return,
    build_kpi_card,
    calmar_ratio,
    get_strategy_label,
    safe_float,
)
from core.report_builder.sections import build_chart_data
from core.report_builder.diversification import (
    build_diversification_html,
    build_rebalance_html,
    build_strategy_oos_rows,
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc_pl_ratio(metrics: Dict[str, Any]) -> float:
    """计算盈亏比。"""
    ap = safe_float(metrics.get("avg_profit_pct", 0))
    al = safe_float(metrics.get("avg_loss_pct", 1))
    return ap / abs(al) if abs(al) > 0 else 0


def _build_html_report(report_data: Dict[str, Any]) -> str:
    """根据报告数据构建完整 HTML 字符串。"""
    ctx = {**report_data}
    now_str_val = ctx.get("now_str", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ctx["now_year"] = now_str_val[:4]
    tpl = string.Template(HTML_TEMPLATE)
    return tpl.substitute(**ctx)


# ---------------------------------------------------------------------------
# 衍生指标计算
# ---------------------------------------------------------------------------


def _enrich_strategy_metrics(
    strategies_data: Dict[str, Any],
    total_days: int,
    total_years: float,
) -> None:
    """为每个策略计算衍生指标（ann_return, calmar, label）。"""
    strategy_label_lib = StrategyLibrary()

    def _label_for(name: str) -> str:
        if "_" in name:
            prefix, sub = name.split("_", 1)
            from core.report_builder.metrics import _short_label_from_description

            profile = strategy_label_lib.get_profile(sub)
            if profile is not None:
                short = _short_label_from_description(profile.description)
                if short:
                    return f"{prefix} {short}"
        return get_strategy_label(name, strategy_label_lib)

    for name, sd in strategies_data.items():
        metrics = sd.get("metrics", {})
        dates = sd.get("dates", [])

        n_days = len(dates) if dates else total_days
        years_exp = n_days / TRADING_DAYS_PER_YEAR if n_days > 0 else total_years

        total_ret = safe_float(metrics.get("total_return_pct", 0))
        ann_ret = annualized_return(total_ret, years_exp)
        max_dd = safe_float(metrics.get("max_drawdown_pct", 0))
        calmar = calmar_ratio(ann_ret, max_dd)

        metrics["ann_return"] = ann_ret
        metrics["calmar"] = calmar
        metrics["total_years"] = years_exp
        sd["label"] = _label_for(name)


# ---------------------------------------------------------------------------
# HTML 表格构建
# ---------------------------------------------------------------------------


def _build_strategy_table(strategy_names: List[str], strategies_data: Dict[str, Any]) -> str:
    """构建策略对比表格 HTML 行。"""
    rows = ""
    for name in strategy_names:
        sd = strategies_data[name]
        m = sd.get("metrics", {})
        tr = safe_float(m.get("total_return_pct", 0))
        ar = m.get("ann_return", 0)
        sh = safe_float(m.get("sharpe", 0))
        dd = safe_float(m.get("max_drawdown_pct", 0))
        ca = m.get("calmar", 0)
        wr = safe_float(m.get("win_rate", 0))
        tc = int(safe_float(m.get("trade_count", 0)))
        pl = _calc_pl_ratio(m)

        def _cls(v, t=0):
            if v > t:
                return "positive"
            elif v < -t:
                return "negative"
            return ""

        rows += f"""
        <tr>
            <td><strong>{sd["label"]}</strong></td>
            <td class="{_cls(tr)}">{tr:+.2f}%</td>
            <td class="{_cls(ar)}">{ar:+.2f}%</td>
            <td class="{_cls(sh, 0.1)}">{sh:.4f}</td>
            <td class="{"negative" if dd < -0.5 else ""}">{dd:.2f}%</td>
            <td class="{_cls(ca, 0.1)}">{ca:.4f}</td>
            <td>{wr:.1f}%</td>
            <td>{pl:.2f}</td>
            <td>{tc}</td>
        </tr>"""
    return rows


def _build_oos_table(
    out_sample_metrics: Dict[str, Any],
    total_years: float,
) -> str:
    """构建样本内/外表格 HTML 行。"""
    rows = ""
    for split_name in ["in_sample", "out_sample"]:
        if split_name in out_sample_metrics:
            m = out_sample_metrics[split_name]
            tr = safe_float(m.get("total_return_pct", 0))
            years_this = 5 if split_name == "in_sample" else max(total_years - 5, 0.5)
            ar = annualized_return(tr, years_this)
            sh = safe_float(m.get("sharpe", 0))
            dd = safe_float(m.get("max_drawdown_pct", 0))
            ca = calmar_ratio(ar, dd)
            wr = safe_float(m.get("win_rate", 0))
            tc = int(safe_float(m.get("trade_count", 0)))
            split_label = (
                "样本内 (2016-2020)"
                if split_name == "in_sample"
                else "样本外 (2021-2025)"
            )
            rows += f"""
            <tr>
                <td><strong>{split_label}</strong></td>
                <td class="{"positive" if tr > 0 else "negative"}">{tr:+.2f}%</td>
                <td class="{"positive" if ar > 0 else "negative"}">{ar:+.2f}%</td>
                <td class="{"positive" if sh > 0.1 else ""}">{sh:.4f}</td>
                <td class="{"negative" if dd < -0.5 else ""}">{dd:.2f}%</td>
                <td class="{"positive" if ca > 0.1 else ""}">{ca:.4f}</td>
                <td>{wr:.1f}%</td>
                <td>{tc}</td>
            </tr>"""
    return rows


# ---------------------------------------------------------------------------
# 主入口：generate_report
# ---------------------------------------------------------------------------


def generate_report(
    output_dir: Optional[Union[Path, str]] = None,
    strategies_data: Optional[Dict[str, Any]] = None,
    out_sample_metrics: Optional[Dict[str, Any]] = None,
    in_sample_dates: Optional[List[str]] = None,
    in_sample_equity: Optional[List[float]] = None,
    out_sample_dates: Optional[List[str]] = None,
    out_sample_equity: Optional[List[float]] = None,
    rebalance_analysis: Optional[Dict[str, Any]] = None,
    title: str = "量化回测分析报告",
    subtitle: str = "多策略期货量化交易系统 · 综合绩效评估",
    report_name: str = "backtest_report_full.html",
    include_evaluation: bool = True,
    config: Optional[Dict[str, Any]] = None,
    evaluation_html: Optional[str] = None,
) -> Path:
    """
    生成完整的量化回测分析 HTML 报告。

    Args:
        output_dir: 输出目录路径。若 strategies_data 未提供，则从此目录自动扫描。
        strategies_data: {策略名: {"metrics": {...}, "dates": [...], "equity": [...]}}
        out_sample_metrics: 样本外指标
        in_sample_dates / in_sample_equity: 样本内净值数据
        out_sample_dates / out_sample_equity: 样本外净值数据
        rebalance_analysis: 调仓决策分析
        title / subtitle: 报告标题与副标题
        report_name: 输出文件名
        include_evaluation: 是否包含综合评价与改进建议模块
        config: 配置字典，用于动态生成评价
        evaluation_html: 自定义评价 HTML，若提供则不自动生成

    Returns:
        生成的报告文件 Path
    """
    out_path = Path(output_dir) if output_dir else Path("output_backtest_pybroker")
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 数据加载 ──
    loaded = load_strategies_data(
        out_path, strategies_data, out_sample_metrics,
        in_sample_dates, in_sample_equity,
        out_sample_dates, out_sample_equity,
        rebalance_analysis,
    )
    strategies_data = loaded["strategies_data"]
    out_sample_metrics = loaded["out_sample_metrics"]
    in_sample_dates = loaded["in_sample_dates"]
    in_sample_equity = loaded["in_sample_equity"]
    out_sample_dates = loaded["out_sample_dates"]
    out_sample_equity = loaded["out_sample_equity"]
    rebalance_analysis = loaded["rebalance_analysis"]
    diversification_data = loaded["diversification_data"]
    data_source_note = loaded["data_source_note"]
    missing_files_note = loaded["missing_files_note"]

    if not strategies_data:
        logger.error("报告生成: 无可用策略数据，中止")
        return out_path / report_name

    strategy_names = list(strategies_data.keys())
    logger.info(f"报告生成: 发现 {len(strategy_names)} 个策略: {strategy_names}")

    # ── 确定回测区间 ──
    first_strategy = strategies_data[strategy_names[0]]
    all_dates = first_strategy.get("dates", [])
    if all_dates:
        backtest_start = all_dates[0]
        backtest_end = all_dates[-1]
        total_days = len(all_dates)
    else:
        backtest_start = "N/A"
        backtest_end = "N/A"
        total_days = 0
    total_years = total_days / TRADING_DAYS_PER_YEAR if total_days > 0 else 0

    # ── 衍生指标 ──
    _enrich_strategy_metrics(strategies_data, total_days, total_years)

    # ── 选最佳策略 ──
    best_exp = strategy_names[0]
    for name in strategy_names:
        if (
            safe_float(
                strategies_data[name].get("metrics", {}).get("total_return_pct", 0)
            )
            > 0
        ):
            best_exp = name
            break

    best = strategies_data[best_exp]
    best_metrics = best.get("metrics", {})
    best_equity = best.get("equity", [])
    best_dates = best.get("dates", [])
    best_label = best.get("label", best_exp)

    # ── KPI 卡片 HTML ──
    kpi_items = [
        (
            "年化收益率",
            f"{best_metrics.get('ann_return', 0):+.2f}%",
            best_metrics.get("ann_return", 0),
        ),
        (
            "夏普比率",
            f"{safe_float(best_metrics.get('sharpe', 0)):.4f}",
            safe_float(best_metrics.get("sharpe", 0)),
        ),
        (
            "最大回撤",
            f"{safe_float(best_metrics.get('max_drawdown_pct', 0)):.2f}%",
            safe_float(best_metrics.get("max_drawdown_pct", 0)),
        ),
        (
            "卡玛比率",
            f"{best_metrics.get('calmar', 0):.4f}",
            best_metrics.get("calmar", 0),
        ),
        (
            "胜率",
            f"{safe_float(best_metrics.get('win_rate', 0)):.1f}%",
            safe_float(best_metrics.get("win_rate", 0)),
        ),
        ("盈亏比", f"{_calc_pl_ratio(best_metrics):.2f}", _calc_pl_ratio(best_metrics)),
        (
            "总交易次数",
            f"{int(safe_float(best_metrics.get('trade_count', 0)))}",
            safe_float(best_metrics.get("trade_count", 0)),
        ),
    ]
    kpi_cards_html = "\n".join(
        build_kpi_card(label, val, num) for label, val, num in kpi_items
    )

    # ── 表格 HTML ──
    strategy_table_rows = _build_strategy_table(strategy_names, strategies_data)
    oos_table_rows = _build_oos_table(out_sample_metrics, total_years)

    # ── 图表数据 ──
    chart_data, in_sample_js, out_sample_js = build_chart_data(
        strategy_names, strategies_data,
        best_dates, best_equity,
        in_sample_dates, in_sample_equity,
        out_sample_dates, out_sample_equity,
        diversification_data,
    )

    # ── 策略样本内/外绩效对比表 ──
    strategy_oos_rows = build_strategy_oos_rows(
        strategy_names, strategies_data,
        in_sample_dates, out_sample_dates,
    )

    # ── 多品种分散化分析 ──
    diversification_html = build_diversification_html(diversification_data)

    # ── 调仓决策分析 ──
    rebalance_html = build_rebalance_html(rebalance_analysis)

    # ── 综合评价 ──
    if not include_evaluation:
        evaluation_html = ""
    elif evaluation_html is not None:
        pass
    else:
        evaluation_html = build_dynamic_evaluation_html(
            config=config,
            strategies_data=strategies_data,
            out_sample_metrics=out_sample_metrics,
            output_dir=out_path,
        )

    # ── 构建报告上下文 ──
    report_ctx = {
        "backtest_start": backtest_start,
        "backtest_end": backtest_end,
        "total_days": total_days,
        "total_years": f"{total_years:.1f}",
        "init_capital": "1,000,000",
        "now_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "title": title,
        "subtitle": subtitle,
        "symbols_str": "SHFE.RB, DCE.M, CZCE.TA, SHFE.CU, CFFEX.IF",
        "best_strategy_label": best_label,
        "kpi_cards_html": kpi_cards_html,
        "strategy_table_html": strategy_table_rows,
        "oos_table_html": oos_table_rows,
        "strategy_oos_html": strategy_oos_rows,
        "diversification_html": diversification_html,
        "chart_data_json": json.dumps(chart_data, ensure_ascii=False),
        "rebalance_html": rebalance_html,
        "evaluation_html": evaluation_html,
        "in_sample_js": json.dumps(in_sample_js, ensure_ascii=False),
        "out_sample_js": json.dumps(out_sample_js, ensure_ascii=False),
        "data_source_note": data_source_note,
        "missing_files_note": missing_files_note,
    }

    report_ctx["data_source_warning_html"] = _build_data_source_warning(
        data_source_note, missing_files_note
    )

    html = _build_html_report(report_ctx)
    report_path = out_path / report_name
    report_path.write_text(html, encoding="utf-8")
    logger.info(
        f"报告已生成: {report_path} ({report_path.stat().st_size / 1024:.1f} KB)"
    )
    return report_path


def _build_data_source_warning(data_source_note: str, missing_files_note: str) -> str:
    """构建数据来源/缺失文件警告 HTML。"""
    if not data_source_note and not missing_files_note:
        return ""
    parts = []
    if data_source_note:
        parts.append(
            f'<div class="meta-item"><span class="label">&#9888; 数据来源</span>'
            f'<span class="value" style="color:#f59e0b;">{data_source_note}</span></div>'
        )
    if missing_files_note:
        parts.append(
            f'<div class="meta-item"><span class="label">&#10060; 缺失</span>'
            f'<span class="value" style="color:#ef4444;">{missing_files_note}</span></div>'
        )
    if not parts:
        return ""
    return (
        '<div class="meta-row" style="margin-top:12px; background:rgba(245,158,11,0.1);'
        ' padding:10px 14px; border-radius:8px; border:1px solid rgba(245,158,11,0.3);">'
        + "\n".join(parts) + "</div>"
    )
