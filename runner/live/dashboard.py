"""模拟交易看板 — 生成 HTML 报告（净值曲线 + 持仓热力图 + 绩效摘要）。

依赖 matplotlib 生成图表，base64 嵌入 HTML。
规则 17：绩效计算委托 utils/metrics.py::MetricsCalculator。

用法：
    from runner.live.dashboard import generate_dashboard
    generate_dashboard("output_backtest_pybroker/l3_daily_sim/daily_log.csv",
                       "output_backtest_pybroker/l3_daily_sim/dashboard.html")
"""
from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from utils.metrics import MetricsCalculator

__all__ = [
    "generate_dashboard",
    "plot_equity_curve",
    "plot_position_heatmap",
    "compute_performance_summary",
]

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 目标参数（来自 e12b 回测基准）
_TARGET_ANNUAL_RET = 1.8      # 目标年化收益 %
_TARGET_MAX_DD = -1.39        # 历史最大回撤 %
_INITIAL_CAPITAL = 1_000_000  # 初始资金


# ==================================================================
# 图表生成
# ==================================================================

def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_equity_curve(
    df: pd.DataFrame,
    equity_col: str = "equity",
    benchmark_series: Optional[pd.Series] = None,
    benchmark_label: str = "回测基准 e12b",
    title: str = "模拟交易净值曲线",
) -> str:
    """净值曲线 + 滚动回撤子图 + 可选基准叠加。

    Args:
        df: 含 date 和 equity 列的 DataFrame
        equity_col: 净值列名
        benchmark_series: 基准净值 Series（index=date, values=equity），可选
        benchmark_label: 基准图例名称
        title: 图表标题
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    dates = pd.to_datetime(df["date"])
    equity = df[equity_col].values

    # 净值曲线 — 深蓝 + 半透明填充
    ax1.plot(dates, equity, color="#1a5276", linewidth=1.8, label="模拟净值")
    ax1.fill_between(dates, equity, df[equity_col].iloc[0],
                     alpha=0.08, color="#1a5276")
    ax1.axhline(y=df[equity_col].iloc[0], color="gray", linestyle=":", alpha=0.4,
                label="初始净值")

    # 基准叠加
    if benchmark_series is not None and len(benchmark_series) > 0:
        bm = benchmark_series.reindex(dates, method="ffill") if len(benchmark_series) > 1 else benchmark_series
        ax1.plot(dates, bm.values, color="#e67e22", linewidth=1.2, linestyle="--",
                 alpha=0.8, label=benchmark_label)

    ax1.set_ylabel("净值")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

    # 滚动回撤子图 — 橙色警示色
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    max_dd = drawdown.min()
    ax2.fill_between(dates, drawdown, 0, color="#e67e22", alpha=0.25)
    ax2.plot(dates, drawdown, color="#e67e22", linewidth=0.8)
    ax2.axhline(y=max_dd, color="#c0392b", linestyle=":", alpha=0.5,
                label=f"最大回撤 {max_dd:.2f}%")
    ax2.set_ylabel("回撤 %")
    ax2.set_xlabel("日期")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_position_heatmap(
    pos_df: pd.DataFrame,
    title: str = "品种仓位热力图",
) -> str:
    """品种仓位热力图：横轴日期，纵轴品种，颜色=仓位方向。

    Args:
        pos_df: DataFrame with date, symbol, position columns
    """
    if pos_df.empty or "symbol" not in pos_df.columns:
        return ""

    pivot = pos_df.pivot_table(
        index="date", columns="symbol", values="position",
        aggfunc="sum", fill_value=0,
    )
    if pivot.empty:
        return ""

    n_dates = len(pivot)
    n_symbols = len(pivot.columns)
    fig, ax = plt.subplots(figsize=(max(10, n_dates * 0.3), max(3, n_symbols * 0.45)))

    # 红=空(-1), 白=空仓(0), 绿=多(+1)
    from matplotlib.colors import LinearSegmentedColormap
    pos_neg_cmap = LinearSegmentedColormap.from_list(
        "pos_neg", ["#d62728", "#ffffff", "#2ca02c"], N=256,
    )
    im = ax.imshow(pivot.T.values, aspect="auto", cmap=pos_neg_cmap, vmin=-1, vmax=1)

    ax.set_yticks(range(n_symbols))
    ax.set_yticklabels(pivot.columns, fontsize=9)
    ax.set_xlabel("日期", fontsize=10)

    # x 轴刻度
    step = max(1, n_dates // 12)
    ax.set_xticks(range(0, n_dates, step))
    ax.set_xticklabels(
        [str(d).split(" ")[0] for d in pivot.index[::step]],
        fontsize=7, rotation=45, ha="right",
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="仓位方向")
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["空", "空仓", "多"])

    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_performance_bars(metrics: Dict[str, Any]) -> str:
    """绩效摘要柱状图（3 子图）。"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.suptitle("绩效摘要", fontsize=12, fontweight="bold")

    # 子图 1：风险调整收益
    labels = ["Sharpe", "Calmar", "Sortino"]
    values = [
        metrics.get("sharpe", 0),
        metrics.get("calmar", 0),
        metrics.get("sortino", 0),
    ]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    axes[0].bar(labels, values, color=colors, alpha=0.85)
    axes[0].axhline(y=0, color="gray", linewidth=0.5)
    axes[0].set_title("风险调整收益")
    axes[0].grid(True, alpha=0.3)

    # 子图 2：收益与风险
    labels2 = ["总收益%", "年化收益%", "最大回撤%", "胜率%"]
    values2 = [
        metrics.get("total_return_pct", 0),
        metrics.get("annual_return_pct", 0),
        metrics.get("max_drawdown_pct", 0),
        metrics.get("win_rate", 0),
    ]
    colors2 = ["#1f77b4", "#2ca02c", "#e67e22", "#9467bd"]
    axes[1].bar(labels2, values2, color=colors2, alpha=0.85)
    axes[1].axhline(y=0, color="gray", linewidth=0.5)
    axes[1].set_title("收益与风险")
    axes[1].grid(True, alpha=0.3)
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(30)
        tick.set_fontsize(8)

    # 子图 3：日收益率分布
    if "equity" in metrics:
        equity = metrics["equity"]
        if isinstance(equity, pd.Series) and len(equity) > 1:
            daily_ret = equity.pct_change().dropna()
            axes[2].hist(daily_ret * 100, bins=min(30, len(daily_ret)),
                         color="#1a5276", alpha=0.7, edgecolor="white")
            axes[2].axvline(x=0, color="gray", linewidth=0.5)
            axes[2].set_title("日收益率分布")
            axes[2].set_xlabel("日收益率 %")
            axes[2].set_ylabel("频率")
            axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return _fig_to_base64(fig)


# ==================================================================
# 绩效计算
# ==================================================================

def _compute_drawdown_distance(
    current_dd_pct: float, target_max_dd_pct: float = _TARGET_MAX_DD,
) -> float:
    """当前回撤与历史最大回撤的距离（正数=有安全空间，负数=已突破）。"""
    return abs(target_max_dd_pct) - abs(current_dd_pct)


def compute_performance_summary(
    df: pd.DataFrame,
    equity_col: str = "equity",
    initial_capital: float = _INITIAL_CAPITAL,
) -> Dict[str, Any]:
    """从模拟日志 DataFrame 计算绩效摘要。

    规则 17：核心指标委托 MetricsCalculator.compute_from_equity_curve。
    """
    if df.empty:
        return {}

    df = df.sort_values("date").reset_index(drop=True)

    if equity_col not in df.columns:
        df["equity"] = df.get("pnl", 0).cumsum() + initial_capital
        equity_col = "equity"

    portfolio_df = df[["date", equity_col]].copy()
    portfolio_df = portfolio_df.rename(columns={equity_col: "equity"})
    portfolio_df = portfolio_df.set_index("date")

    metrics = MetricsCalculator.compute_from_equity_curve(portfolio_df, None)

    equity = portfolio_df["equity"]
    n_days = len(equity)

    # 基本摘要
    total_ret = float((equity.iloc[-1] / equity.iloc[0] - 1) * 100) if n_days > 1 else 0.0
    max_dd = float(metrics.get("max_drawdown_pct", 0))
    annual_ret = float(metrics.get("annual_return_pct", 0))

    metrics["equity"] = equity
    metrics["start_date"] = str(df["date"].iloc[0]).split(" ")[0]
    metrics["end_date"] = str(df["date"].iloc[-1]).split(" ")[0]
    metrics["trading_days"] = n_days
    metrics["total_pnl"] = float(equity.iloc[-1] - equity.iloc[0] if n_days > 0 else 0)
    metrics["total_return_pct"] = total_ret
    metrics["max_drawdown_pct"] = max_dd
    metrics["annual_return_pct"] = annual_ret

    # 新增：回撤距离
    metrics["drawdown_distance"] = _compute_drawdown_distance(max_dd)

    # 新增：超额收益（相对初始资金）
    metrics["excess_return"] = float(equity.iloc[-1]) - initial_capital

    # 新增：年化收益进度（相对目标 1.8%）
    metrics["annual_progress"] = min(100, max(0, (annual_ret / _TARGET_ANNUAL_RET * 100)
                                              if _TARGET_ANNUAL_RET != 0 else 0))

    # 持仓统计
    if "position" in df.columns:
        pos = df["position"]
        metrics["avg_position"] = float(pos.mean())
        metrics["long_pct"] = float((pos > 0).sum() / len(pos) * 100)
        metrics["short_pct"] = float((pos < 0).sum() / len(pos) * 100)
        metrics["flat_pct"] = float((pos == 0).sum() / len(pos) * 100)

    # 交易统计
    if "signal_strength" in df.columns:
        sig = df["signal_strength"]
        metrics["signal_count"] = int((sig != 0).sum())
        metrics["signal_change_count"] = int((sig.diff() != 0).sum())

    return metrics


# ==================================================================
# 数据新鲜度
# ==================================================================

def _build_data_freshness_html(
    data_check_report: Optional[pd.DataFrame] = None,
    today: Optional[pd.Timestamp] = None,
) -> str:
    """构建数据新鲜度状态条 HTML。"""
    today = today or pd.Timestamp.now().normalize()

    if data_check_report is None or data_check_report.empty:
        return f"""
        <div class="freshness freshness-warn">
          <strong>数据新鲜度</strong> | 数据截止: 未知 | 未执行完整性检查
        </div>"""

    ok_count = int((data_check_report["status"] == "ok").sum())
    warn_count = int((data_check_report["status"] == "warn").sum())
    error_count = int((data_check_report["status"] == "error").sum())

    if error_count > 0:
        css_class = "freshness freshness-error"
        status_text = "错误"
    elif warn_count > 0:
        css_class = "freshness freshness-warn"
        status_text = "警告"
    else:
        css_class = "freshness freshness-ok"
        status_text = "完整"

    # 最新数据日期
    last_dates = []
    for _, row in data_check_report.iterrows():
        ld = row.get("last_date")
        if ld is not None:
            last_dates.append(pd.Timestamp(ld))
    data_end = max(last_dates).strftime("%Y-%m-%d") if last_dates else "未知"

    detail_parts = []
    for _, row in data_check_report.iterrows():
        s = row["status"]
        sym = row["symbol"]
        if s == "error":
            detail_parts.append(f"<span class='f-error'>{sym}</span>")
        elif s == "warn":
            gap = row.get("days_gap", "?")
            detail_parts.append(f"<span class='f-warn'>{sym}({gap}d)</span>")

    detail_html = " ".join(detail_parts) if detail_parts else "全部正常"

    return f"""
    <div class="{css_class}">
      <strong>数据新鲜度</strong> |
      数据截止: {data_end} |
      状态: {ok_count} OK / {warn_count} 警告 / {error_count} 错误 =
      {status_text} |
      详情: {detail_html}
    </div>"""


# ==================================================================
# 基准加载
# ==================================================================

def _load_benchmark(benchmark_path: Optional[str]) -> Optional[pd.Series]:
    """加载回测基准净值曲线。"""
    if not benchmark_path:
        return None
    p = Path(benchmark_path)
    if not p.exists():
        return None
    try:
        bm = pd.read_csv(p)
        if "date" in bm.columns and "equity" in bm.columns:
            bm["date"] = pd.to_datetime(bm["date"])
            return bm.set_index("date")["equity"]
    except Exception:
        pass
    return None


# ==================================================================
# HTML 看板生成
# ==================================================================

def generate_dashboard(
    log_path: str,
    output_path: str,
    title: str = "三因子+9品种等权 模拟交易看板",
    benchmark_path: Optional[str] = None,
    positions_path: Optional[str] = None,
    data_check_report: Optional[pd.DataFrame] = None,
) -> str:
    """从模拟日志 CSV 生成完整 HTML 看板。

    Args:
        log_path: 模拟日志 CSV（含 date, equity, position, signal 等列）
        output_path: 输出 HTML 路径
        title: 页面标题
        benchmark_path: 基准净值 CSV（可选，用于叠加对比）
        positions_path: 逐品种持仓 CSV（可选，用于热力图）
        data_check_report: 数据完整性检查报告 DataFrame（可选）
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"模拟日志不存在: {log_path}")

    df = pd.read_csv(log_path)
    if df.empty:
        raise ValueError("模拟日志为空")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "equity" not in df.columns:
        if "pnl" in df.columns:
            df["equity"] = df["pnl"].cumsum() + _INITIAL_CAPITAL
        else:
            df["equity"] = _INITIAL_CAPITAL

    # 基准
    benchmark = _load_benchmark(benchmark_path)

    # 绩效
    metrics = compute_performance_summary(df)

    # 图表
    equity_b64 = plot_equity_curve(
        df, equity_col="equity", benchmark_series=benchmark,
        title="净值曲线 vs 回测基准",
    )
    perf_b64 = plot_performance_bars(metrics) if metrics else ""

    # 持仓热力图
    pos_b64 = ""
    if positions_path:
        pos_path = Path(positions_path)
        if pos_path.exists():
            pos_df = pd.read_csv(pos_path)
            if not pos_df.empty:
                pos_b64 = plot_position_heatmap(pos_df)

    # 交易记录表
    trade_cols = ["date", "signal_strength", "trigger_symbols", "trigger_reason",
                  "top_factor", "position", "pnl", "equity", "status"]
    display_cols = [c for c in trade_cols if c in df.columns]
    trade_html = df[display_cols].tail(20).to_html(
        index=False, classes="trade-table", border=0,
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) and abs(x) < 10 else f"{x:,.0f}" if isinstance(x, float) else str(x),
    )

    # 绩效数值
    sharpe = metrics.get("sharpe", 0)
    calmar = metrics.get("calmar", 0)
    total_ret = metrics.get("total_return_pct", 0)
    max_dd = metrics.get("max_drawdown_pct", 0)
    annual_ret = metrics.get("annual_return_pct", 0)
    dd_distance = metrics.get("drawdown_distance", 0)
    excess_return = metrics.get("excess_return", 0)
    annual_progress = metrics.get("annual_progress", 0)
    total_pnl = metrics.get("total_pnl", 0)
    equity_last = df["equity"].iloc[-1]

    # 数据新鲜度
    freshness_html = _build_data_freshness_html(data_check_report)

    # 生成时间
    now = datetime.now()
    start_d = metrics.get("start_date", "?")
    end_d = metrics.get("end_date", "?")
    n_days = metrics.get("trading_days", 0)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f6fa; color: #2d3436; padding: 20px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 24px; margin-bottom: 5px; }}
  .subtitle {{ color: #636e72; font-size: 14px; margin-bottom: 20px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
            gap: 12px; margin-bottom: 20px; }}
  .card {{ background: #fff; border-radius: 8px; padding: 14px 16px;
           box-shadow: 0 1px 3px rgba(0,0,0,.1); text-align: center; }}
  .card .label {{ font-size: 11px; color: #636e72; margin-bottom: 4px; }}
  .card .value {{ font-size: 20px; font-weight: 700; }}
  .card .sub {{ font-size: 11px; color: #b2bec3; margin-top: 2px; }}
  .positive {{ color: #00b894; }}
  .negative {{ color: #d63031; }}
  .drawdown {{ color: #e67e22; }}
  .neutral {{ color: #636e72; }}
  /* 进度条 */
  .progress-bar {{ background: #f0f0f0; border-radius: 4px; height: 6px; margin-top: 6px; }}
  .progress-fill {{ background: #1a5276; border-radius: 4px; height: 6px; }}
  .progress-fill.good {{ background: #00b894; }}
  .progress-fill.warn {{ background: #e67e22; }}
  .chart {{ background: #fff; border-radius: 8px; padding: 16px; margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  .chart img {{ width: 100%; height: auto; }}
  .section-title {{ font-size: 16px; font-weight: 600; margin-bottom: 10px; }}
  .trade-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  .trade-table th {{ background: #dfe6e9; padding: 8px 10px; text-align: left; }}
  .trade-table td {{ padding: 6px 10px; border-bottom: 1px solid #f0f0f0; }}
  .trade-table tr:hover {{ background: #f8f9fa; }}
  .footer {{ text-align: center; color: #b2bec3; font-size: 12px; margin-top: 20px; }}
  /* 数据新鲜度 */
  .freshness {{ padding: 10px 14px; border-radius: 4px; margin-bottom: 16px; font-size: 13px; }}
  .freshness-ok {{ background: #d5f5e3; border-left: 4px solid #00b894; }}
  .freshness-warn {{ background: #ffeaa7; border-left: 4px solid #e67e22; }}
  .freshness-error {{ background: #fadbd8; border-left: 4px solid #d63031; }}
  .f-error {{ color: #d63031; font-weight: 600; }}
  .f-warn {{ color: #e67e22; }}
  /* 说明框 */
  .note {{ background: #d6eaf8; border-left: 4px solid #1a5276; padding: 10px 14px;
          border-radius: 4px; margin-bottom: 16px; font-size: 13px; }}
</style>
</head>
<body>
<div class="container">

<h1>{title}</h1>
<p class="subtitle">更新: {now.strftime('%Y-%m-%d %H:%M')} |
   区间: {start_d} ~ {end_d} |
   交易日: {n_days}</p>

<div class="note">
  <strong>L3 模拟交易</strong> —
  逐日 OOS 回测，T+1 开盘价撮合。回测基准: e12 三因子+9品种等权。
</div>

{freshness_html}

<!-- KPI 卡片 -->
<div class="cards">
  <div class="card">
    <div class="label">当前净值</div>
    <div class="value neutral">{equity_last:,.0f}</div>
    <div class="sub">超额 {excess_return:+,.0f}</div>
  </div>
  <div class="card">
    <div class="label">累计收益</div>
    <div class="value {'positive' if total_ret >= 0 else 'negative'}">{total_ret:.2f}%</div>
    <div class="sub">总 PnL {total_pnl:+,.0f}</div>
  </div>
  <div class="card">
    <div class="label">年化收益</div>
    <div class="value {'positive' if annual_ret >= 0 else 'negative'}">{annual_ret:.2f}%</div>
    <div class="progress-bar"><div class="progress-fill {'good' if annual_progress >= 60 else 'warn'}" style="width:{annual_progress:.0f}%"></div></div>
    <div class="sub">目标 {_TARGET_ANNUAL_RET}%</div>
  </div>
  <div class="card">
    <div class="label">当前回撤</div>
    <div class="value drawdown">{max_dd:.2f}%</div>
    <div class="sub">历史最大 {_TARGET_MAX_DD}% | 距离 {dd_distance:+.2f}%</div>
  </div>
  <div class="card">
    <div class="label">Sharpe</div>
    <div class="value {'positive' if sharpe >= 0 else 'negative'}">{sharpe:.2f}</div>
  </div>
  <div class="card">
    <div class="label">Calmar</div>
    <div class="value {'positive' if calmar >= 0 else 'negative'}">{calmar:.2f}</div>
  </div>
</div>

<!-- 净值曲线 -->
<div class="chart">
  <div class="section-title">净值曲线 & 滚动回撤</div>
  <img src="data:image/png;base64,{equity_b64}" alt="净值曲线">
</div>

<!-- 持仓热力图 -->
<div class="chart">
  <div class="section-title">品种仓位热力图（红=空，绿=多，白=空仓）</div>
  {"<img src='data:image/png;base64," + pos_b64 + "' alt='持仓热力图'>" if pos_b64 else "<p style='color:#b2bec3;text-align:center;padding:30px'>暂无持仓数据</p>"}
</div>

<!-- 绩效摘要 -->
{"<div class='chart'><div class='section-title'>绩效摘要</div><img src='data:image/png;base64," + perf_b64 + "' alt='绩效摘要'></div>" if perf_b64 else ""}

<!-- 交易记录 -->
<div class="chart">
  <div class="section-title">近期交易/信号记录</div>
  {trade_html}
</div>

<div class="footer">
  L3 模拟交易看板 | 自动生成于 {now.strftime('%Y-%m-%d %H:%M:%S')} |
  策略: 三因子等权 (donchian_breakout + carry + basis_momentum) + 方向二
</div>

</div>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"看板已生成: {output_path}")
    return str(output_path)