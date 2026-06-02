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
            2, 1, figsize=_DEFAULT_FIGSIZE_FULL,
            sharex=True, gridspec_kw={"height_ratios": [3, 1]},
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
                mc["max_drawdowns"], bins=50, alpha=0.7,
                edgecolor="black", color="orange",
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

        axes[0].bar(x - width / 2, df["new_avg_sharpe"], width, label="新配置", alpha=0.8)
        axes[0].bar(x + width / 2, df["old_avg_sharpe"], width, label="旧配置", alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(strategies, rotation=45, ha="right")
        axes[0].set_title("平均Sharpe对比")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(x - width / 2, df["new_min_sharpe"], width, label="新配置", alpha=0.8)
        axes[1].bar(x + width / 2, df["old_min_sharpe"], width, label="旧配置", alpha=0.8)
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
            axes[0].plot(sub["year"], sub["fixed_sharpe"], marker="o", label=f"{sname}(固定)")
            axes[0].plot(sub["year"], sub["regime_sharpe"], marker="s", linestyle="--", label=f"{sname}(环境)")
        axes[0].set_title("按年Sharpe对比")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        for sname in df["strategy"].unique():
            sub = df[df["strategy"] == sname]
            axes[1].plot(sub["year"], sub["fixed_drawdown"], marker="o", label=f"{sname}(固定)")
            axes[1].plot(sub["year"], sub["regime_drawdown"], marker="s", linestyle="--", label=f"{sname}(环境)")
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
            "trend_up", "trend_down", "range_bound",
            "high_volatility", "low_volatility",
            "breakout", "exhaustion_bull", "exhaustion_bear",
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
            values="mean_ic", index="symbol", columns="factor", aggfunc="mean",
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
            axes[1].scatter(sub["ir"], sub["current_weight"], label=sym, s=50, alpha=0.8)
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
        dates = pd.to_datetime(ic_df["date"]) if "date" in ic_df.columns else range(len(ic_df))

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
                axes[1].plot(dates, ic_df[col].values, linewidth=1, alpha=0.8, label=label)
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
                dates, numeric.values, linewidth=1.5, alpha=0.8,
                label=label, marker=".", markersize=2,
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
