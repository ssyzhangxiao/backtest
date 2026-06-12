"""
静态绘图模块。

使用 matplotlib 绘制回测结果可视化图表，保存为 PNG。
委托 utils/plots.py 的 PlotManager（Plotly 后端）用于 Streamlit 场景，
本模块提供 matplotlib 后端用于静态报告生成。

两种后端通过 backend 参数分发，统一接口。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns  # 5 段式热力图（2026-06-12）

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from loguru import logger

_DEFAULT_CHART_DPI = 150
_DEFAULT_FIGSIZE_WIDE = (14, 6)
_DEFAULT_FIGSIZE_FULL = (12, 8)


def plot_equity_curve(
    eq: pd.DataFrame,
    title: str,
    label: str,
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
) -> None:
    """
    绘制净值曲线与回撤双图。

    Args:
        eq: 净值DataFrame，需包含date和equity列
        title: 图表标题
        label: 曲线图例标签
        path: 保存路径
        dpi: 图片DPI
    """
    if eq is None or eq.empty:
        logger.debug(f"跳过空净值曲线绘图: {title}")
        return

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=_DEFAULT_FIGSIZE_FULL,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        dates = pd.to_datetime(eq["date"])
        equity = eq["equity"].values

        ax1.plot(dates, equity, linewidth=1, label=label)
        ax1.set_title(f"{title} — 净值曲线", fontsize=14)
        ax1.set_ylabel("净值")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / np.where(peak > 0, peak, 1.0) * 100
        ax2.fill_between(dates, 0, dd, color="red", alpha=0.3)
        ax2.plot(dates, dd, color="red", linewidth=0.8)
        ax2.set_ylabel("回撤 %")
        ax2.set_xlabel("日期")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"净值曲线已保存: {path}")
    except Exception as e:
        logger.error(f"净值曲线绘图失败 {title}: {e}")
        plt.close("all")


def plot_monte_carlo(
    median: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
) -> None:
    """
    绘制蒙特卡洛模拟净值分布图。

    Args:
        median: 中位数序列
        lower: 5%分位序列
        upper: 95%分位序列
        path: 保存路径
        dpi: 图片DPI
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE_WIDE)
        days = np.arange(len(median))
        ax.fill_between(days, lower, upper, alpha=0.3, color="blue", label="90% CI")
        ax.plot(days, median, color="blue", linewidth=1.5, label="Median")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="初始值")
        ax.set_title("蒙特卡洛模拟 — 净值曲线分布 (1000次)", fontsize=14)
        ax.set_xlabel("交易日")
        ax.set_ylabel("净值")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"蒙特卡洛图表已保存: {path}")
    except Exception as e:
        logger.error(f"蒙特卡洛绘图失败: {e}")
        plt.close("all")


def plot_monte_carlo_distribution(
    mc_results: Dict[str, Any],
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
) -> None:
    """
    绘制蒙特卡洛终值和回撤分布图。

    Args:
        mc_results: {策略名: 模拟结果字典}
        path: 保存路径
        dpi: 图片DPI
    """
    try:
        n = len(mc_results)
        if n == 0:
            return
        fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for i, (sname, mc) in enumerate(mc_results.items()):
            axes[i, 0].hist(mc["final_values"], bins=50, alpha=0.7, edgecolor="black")
            axes[i, 0].axvline(1.0, color="red", linestyle="--", label="盈亏平衡")
            axes[i, 0].set_title(f"{sname} 终值分布")
            axes[i, 0].legend()

            axes[i, 1].hist(
                mc["max_drawdowns"],
                bins=50,
                alpha=0.7,
                edgecolor="black",
                color="orange",
            )
            axes[i, 1].set_title(f"{sname} 最大回撤分布")

        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"  已保存: {path}")
    except Exception as e:
        logger.error(f"  绘图失败: {e}")


def plot_wf_comparison(
    df: pd.DataFrame,
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
) -> None:
    """
    绘制 WalkForward 新旧配置对比图。

    Args:
        df: WalkForward 对比数据
        path: 保存路径
        dpi: 图片DPI
    """
    try:
        if df.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        strategies = df["strategy"].tolist()
        x = np.arange(len(strategies))
        width = 0.35

        axes[0].bar(
            x - width / 2, df["new_avg_sharpe"], width, label="新配置", alpha=0.8
        )
        axes[0].bar(
            x + width / 2, df["old_avg_sharpe"], width, label="旧配置", alpha=0.8
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(strategies, rotation=45, ha="right")
        axes[0].set_title("平均Sharpe对比")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(
            x - width / 2, df["new_min_sharpe"], width, label="新配置", alpha=0.8
        )
        axes[1].bar(
            x + width / 2, df["old_min_sharpe"], width, label="旧配置", alpha=0.8
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(strategies, rotation=45, ha="right")
        axes[1].set_title("最低Sharpe对比")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"  已保存: {path}")
    except Exception as e:
        logger.error(f"  绘图失败: {e}")


def plot_wf_sensitivity(
    df: pd.DataFrame,
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
) -> None:
    """
    绘制 WalkForward 窗口敏感性分析图。

    Args:
        df: 窗口敏感性数据
        path: 保存路径
        dpi: 图片DPI
    """
    try:
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        for sname in df["strategy"].unique():
            sub = df[df["strategy"] == sname]
            ax.plot(sub["train_bars"], sub["avg_sharpe"], marker="o", label=sname)
        ax.set_xlabel("训练窗口长度（交易日）")
        ax.set_ylabel("平均Sharpe")
        ax.set_title("WalkForward 窗口敏感性分析")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"  已保存: {path}")
    except Exception as e:
        logger.error(f"  绘图失败: {e}")


def plot_yearly_comparison(
    df: pd.DataFrame,
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
) -> None:
    """
    绘制按年验证对比图。

    Args:
        df: 按年验证数据
        path: 保存路径
        dpi: 图片DPI
    """
    try:
        if df.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for sname in df["strategy"].unique():
            sub = df[df["strategy"] == sname]
            axes[0].plot(
                sub["year"], sub["fixed_sharpe"], marker="o", label=f"{sname}(固定)"
            )
            axes[0].plot(
                sub["year"],
                sub["regime_sharpe"],
                marker="s",
                linestyle="--",
                label=f"{sname}(环境)",
            )
        axes[0].set_title("按年Sharpe对比")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        for sname in df["strategy"].unique():
            sub = df[df["strategy"] == sname]
            axes[1].plot(
                sub["year"], sub["fixed_drawdown"], marker="o", label=f"{sname}(固定)"
            )
            axes[1].plot(
                sub["year"],
                sub["regime_drawdown"],
                marker="s",
                linestyle="--",
                label=f"{sname}(环境)",
            )
        axes[1].set_title("按年最大回撤对比")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"  已保存: {path}")
    except Exception as e:
        logger.error(f"  绘图失败: {e}")


def plot_env_distribution(
    env_stats: pd.DataFrame,
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
) -> None:
    """
    绘制环境分布图。

    Args:
        env_stats: 环境分布统计数据
        path: 保存路径
        dpi: 图片DPI
    """
    try:
        if env_stats.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        regimes = [
            "trend_up",
            "trend_down",
            "range_bound",
            "high_volatility",
            "low_volatility",
            "breakout",
            "exhaustion_bull",
            "exhaustion_bear",
        ]
        x = env_stats["symbol"]
        bottom = np.zeros(len(x))
        for regime in regimes:
            if regime in env_stats.columns:
                vals = env_stats[regime].values
                ax.bar(x, vals, bottom=bottom, label=regime, alpha=0.7)
                bottom += vals
        ax.set_title("各品种市场环境分布")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"  已保存: {path}")
    except Exception as e:
        logger.error(f"  绘图失败: {e}")


def plot_factor_ic_stability(
    df: pd.DataFrame,
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
) -> None:
    """
    绘制因子IC稳定性对比图。

    Args:
        df: 因子IC稳定性数据
        path: 保存路径
        dpi: 图片DPI
    """
    try:
        if df.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：各因子IC均值（按品种分组柱状）
        pivot_mean = df.pivot_table(
            values="mean_ic",
            index="symbol",
            columns="factor",
            aggfunc="mean",
        )
        if not pivot_mean.empty:
            pivot_mean.plot(kind="bar", ax=axes[0], alpha=0.8)
            axes[0].set_title("各因子IC均值（按品种）")
            axes[0].set_ylabel("Mean IC")
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis="x", rotation=45)

        # 右图：IC信息比率散点
        for sym in df["symbol"].unique():
            sub = df[df["symbol"] == sym]
            axes[1].scatter(
                sub["ir"], sub["current_weight"], label=sym, s=50, alpha=0.8
            )
        axes[1].set_xlabel("IC IR（信息比率）")
        axes[1].set_ylabel("当前权重")
        axes[1].set_title("因子IR vs 动态权重")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"  已保存: {path}")
    except Exception as e:
        logger.error(f"  绘图失败: {e}")


def plot_ic_analysis(
    ic_df: pd.DataFrame,
    title: str,
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
) -> None:
    """
    绘制滚动IC时间序列和动态权重图。

    Args:
        ic_df: IC分析数据
        title: 图表标题
        path: 保存路径
        dpi: 图片DPI
    """
    if ic_df is None or ic_df.empty:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        factor_cols = [c for c in ic_df.columns if c.startswith("ic_")]
        if not factor_cols:
            return

        fig, axes = plt.subplots(2, 1, figsize=_DEFAULT_FIGSIZE_FULL, sharex=True)
        dates = (
            pd.to_datetime(ic_df["date"])
            if "date" in ic_df.columns
            else range(len(ic_df))
        )

        for col in factor_cols:
            label = col.replace("ic_", "")
            axes[0].plot(dates, ic_df[col].values, linewidth=1, alpha=0.8, label=label)
        axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_title(f"{title} — 滚动IC", fontsize=14)
        axes[0].set_ylabel("IC")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        weight_cols = [c for c in ic_df.columns if c.startswith("w_")]
        if weight_cols:
            for col in weight_cols:
                label = col.replace("w_", "")
                axes[1].plot(
                    dates, ic_df[col].values, linewidth=1, alpha=0.8, label=label
                )
            axes[1].set_ylabel("权重")
            axes[1].legend(fontsize=9)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].set_visible(False)

        if "date" in ic_df.columns:
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[-1].set_xlabel("日期")

        plt.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"IC分析图表已保存: {path}")
    except Exception as e:
        logger.error(f"IC分析绘图失败: {e}")
        plt.close("all")


def plot_decay_analysis(
    decay_df: pd.DataFrame,
    title: str,
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
) -> None:
    """
    绘制因子衰减状态图。

    Args:
        decay_df: 衰减状态数据
        title: 图表标题
        path: 保存路径
        dpi: 图片DPI
    """
    if decay_df is None or decay_df.empty:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        status_cols = [c for c in decay_df.columns if c.startswith("status_")]
        if not status_cols:
            return

        status_map = {"healthy": 0, "warning": 1, "decaying": 2, "dead": 3}
        dates = (
            pd.to_datetime(decay_df["date"])
            if "date" in decay_df.columns
            else range(len(decay_df))
        )

        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE_WIDE)
        for col in status_cols:
            label = col.replace("status_", "")
            numeric = decay_df[col].map(status_map).fillna(0).astype(int)
            ax.plot(
                dates,
                numeric.values,
                linewidth=1.5,
                alpha=0.8,
                label=label,
                marker=".",
                markersize=2,
            )

        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["健康", "警告", "衰减", "失效"])
        ax.set_title(f"{title} — 因子衰减状态", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"衰减分析图表已保存: {path}")
    except Exception as e:
        logger.error(f"衰减分析绘图失败: {e}")
        plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
# 5 段式因子验证绘图（2026-06-12 集成）
# ══════════════════════════════════════════════════════════════════════════════


def plot_factor_prf(
    prf_df: pd.DataFrame,
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
) -> None:
    """
    绘制 PRF 离散信号柱状图（按因子）。

    输入：factor_prf.csv，含列 factor / precision / recall / lift / f1 / is_pass
    图：
      - 左：Precision / Recall / F1 三指标柱状对比
      - 右：Lift 柱状（正负着色，基线 0 红线）
    阈值参考线：Precision=0.55, Lift=0

    Args:
        prf_df: PRF DataFrame
        path: 保存路径
        dpi: DPI
    """
    if prf_df is None or prf_df.empty:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        # 兼容列名：可能为 precision/recall/lift 或 prf_precision 等
        col_map = {
            "precision": next(
                (c for c in prf_df.columns if "precision" in c.lower()), None
            ),
            "recall": next((c for c in prf_df.columns if "recall" in c.lower()), None),
            "f1": next(
                (
                    c
                    for c in prf_df.columns
                    if c.lower() == "f1" or "f1_score" in c.lower()
                ),
                None,
            ),
            "lift": next((c for c in prf_df.columns if "lift" in c.lower()), None),
        }
        factor_col = "factor" if "factor" in prf_df.columns else prf_df.columns[0]
        if not all(col_map.values()):
            logger.warning(f"  PRF plot 缺少必要列（{col_map}），跳过")
            return

        # 按因子聚合均值（多品种多策略时）
        agg = (
            prf_df.groupby(factor_col)
            .agg(
                {
                    col_map["precision"]: "mean",
                    col_map["recall"]: "mean",
                    col_map["f1"]: "mean",
                    col_map["lift"]: "mean",
                }
            )
            .reset_index()
        )

        fig, axes = plt.subplots(1, 2, figsize=_DEFAULT_FIGSIZE_FULL)

        x = np.arange(len(agg))
        width = 0.27

        # 左：P / R / F1
        axes[0].bar(
            x - width,
            agg[col_map["precision"]],
            width,
            label="Precision",
            color="#2b6cb0",
            alpha=0.85,
        )
        axes[0].bar(
            x,
            agg[col_map["recall"]],
            width,
            label="Recall",
            color="#38a169",
            alpha=0.85,
        )
        axes[0].bar(
            x + width,
            agg[col_map["f1"]],
            width,
            label="F1",
            color="#d69e2e",
            alpha=0.85,
        )
        axes[0].axhline(
            y=0.55,
            color="#c53030",
            linestyle="--",
            alpha=0.6,
            label="Precision阈值=0.55",
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(
            agg[factor_col].astype(str), rotation=30, ha="right", fontsize=9
        )
        axes[0].set_ylabel("分数")
        axes[0].set_title("PRF 离散信号表现（按因子）")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3, axis="y")

        # 右：Lift（正负着色）
        colors = ["#38a169" if v >= 0 else "#c53030" for v in agg[col_map["lift"]]]
        axes[1].bar(x, agg[col_map["lift"]], color=colors, alpha=0.85)
        axes[1].axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(
            agg[factor_col].astype(str), rotation=30, ha="right", fontsize=9
        )
        axes[1].set_ylabel("Lift")
        axes[1].set_title("Lift 增量（信号组正收益比例 − 基线）")
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"  PRF 图已保存: {path}")
    except Exception as e:
        logger.error(f"  PRF 绘图失败: {e}")
        plt.close("all")


def plot_event_study_returns(
    es_df: pd.DataFrame,
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
) -> None:
    """
    绘制事件研究 T+N 累计收益图。

    输入：event_study.csv，含列 factor / window(T+1,T+3,T+5,T+10) / mean_return / p_value
    图：
      - 左：T+1/3/5/10 平均累计收益（按因子分组柱状）
      - 右：显著性热力图（factor × window，p_value 取 -log10）

    Args:
        es_df: 事件研究 DataFrame
        path: 保存路径
        dpi: DPI
    """
    if es_df is None or es_df.empty:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        # 兼容列名
        window_col = next(
            (c for c in ["window", "horizon", "t_window"] if c in es_df.columns), None
        )
        ret_col = next(
            (
                c
                for c in ["mean_return", "cumulative_return", "avg_return"]
                if c in es_df.columns
            ),
            None,
        )
        pval_col = next(
            (c for c in ["p_value", "pvalue", "pval"] if c in es_df.columns), None
        )
        factor_col = "factor" if "factor" in es_df.columns else es_df.columns[0]

        if not (window_col and ret_col):
            logger.warning("  EventStudy plot 缺少 window/return 列，跳过")
            return

        # 透视：factor × window → mean_return
        pivot_ret = es_df.pivot_table(
            values=ret_col, index=factor_col, columns=window_col, aggfunc="mean"
        )
        # 窗口排序（T+1 → T+10）
        win_order = sorted(
            pivot_ret.columns.tolist(),
            key=lambda w: (
                int(str(w).replace("T+", "").replace("t+", "") or 0)
                if str(w).replace("T+", "").replace("t+", "").isdigit()
                else 99
            ),
        )
        pivot_ret = pivot_ret.reindex(columns=win_order)

        fig, axes = plt.subplots(1, 2, figsize=_DEFAULT_FIGSIZE_FULL)

        # 左：累计收益分组柱状
        pivot_ret.plot(kind="bar", ax=axes[0], alpha=0.85)
        axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_ylabel("平均累计收益")
        axes[0].set_title("事件研究：T+N 累计收益（按因子）")
        axes[0].legend(title="窗口", fontsize=8)
        axes[0].grid(True, alpha=0.3, axis="y")
        axes[0].tick_params(axis="x", rotation=30)

        # 右：显著性热力图（若 p_value 列存在）
        if pval_col:
            pivot_p = es_df.pivot_table(
                values=pval_col, index=factor_col, columns=window_col, aggfunc="mean"
            ).reindex(columns=win_order)
            # -log10(p) ：值越大越显著；clip 避免 log(0)
            with np.errstate(divide="ignore"):
                neg_log_p = -np.log10(pivot_p.clip(lower=1e-300))
            neg_log_p = neg_log_p.replace([np.inf, -np.inf], np.nan).fillna(0)

            sns.heatmap(
                neg_log_p,
                annot=True,
                fmt=".1f",
                cmap="YlOrRd",
                cbar_kws={"label": "-log10(p_value)"},
                ax=axes[1],
                linewidths=0.5,
            )
            axes[1].set_title("显著性热力图（-log10 p）")
            axes[1].set_xlabel("窗口")
            axes[1].set_ylabel("")
        else:
            axes[1].set_visible(False)
            axes[1].text(
                0.5,
                0.5,
                "（无 p_value 列）",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
                color="#888",
            )

        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"  EventStudy 图已保存: {path}")
    except Exception as e:
        logger.error(f"  EventStudy 绘图失败: {e}")
        plt.close("all")


def plot_factor_redundancy_heatmap(
    corr_matrix: "pd.DataFrame",
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
    threshold: float = 0.7,
) -> None:
    """
    绘制 Spearman 因子冗余热力图。

    输入：因子 × 因子的 Spearman ρ 矩阵（绝对值）
    标记：|ρ| ≥ threshold 的格子用红色边框（"应剔除"信号）

    Args:
        corr_matrix: 因子相关性矩阵（对称，值域 [-1, 1]）
        path: 保存路径
        dpi: DPI
        threshold: 冗余阈值（默认 0.7）
    """
    if corr_matrix is None or corr_matrix.empty:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        # 取绝对值展示（冗余只看强度不看方向）
        abs_corr = corr_matrix.abs()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            abs_corr,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "|Spearman ρ|"},
            ax=ax,
            linewidths=0.5,
            square=True,
        )
        ax.set_title(f"因子 Spearman 冗余热力图（|ρ| ≥ {threshold} 视为冗余）")

        # 标记超阈值格子（加红框）
        for i in range(abs_corr.shape[0]):
            for j in range(abs_corr.shape[1]):
                if i != j and abs_corr.iloc[i, j] >= threshold:
                    ax.add_patch(
                        plt.Rectangle(
                            (j, i),
                            1,
                            1,
                            fill=False,
                            edgecolor="red",
                            lw=2.5,
                        )
                    )

        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"  冗余热力图已保存: {path}")
    except Exception as e:
        logger.error(f"  冗余热力图绘图失败: {e}")
        plt.close("all")
