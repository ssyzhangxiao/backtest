"""图表绘制模块 — 参数优化与敏感性分析。"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Optional

from utils.plots._base import COLORS, BasePlotMixin


class OptimizationMixin(BasePlotMixin):
    """参数优化与敏感性分析相关图表。"""

    @staticmethod
    def plot_param_heatmap(
        results_df: pd.DataFrame, param_x: str, param_y: str,
        metric: str = "sharpe", title: str = "参数热力图",
    ) -> go.Figure:
        """二维参数网格热力图。"""
        empty = OptimizationMixin._check_df(results_df, title)
        if empty:
            return empty
        missing = [col for col in [param_x, param_y, metric] if col not in results_df.columns]
        if missing:
            return OptimizationMixin._empty_fig(title, missing_cols=missing)
        pivot = results_df.pivot_table(values=metric, index=param_y, columns=param_x, aggfunc="mean")
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale="RdYlGn",
            text=np.round(pivot.values, 4) if pivot.values.size < 500 else None,
            texttemplate="%{text}" if pivot.values.size < 500 else None,
            colorbar=dict(title=metric),
            zmid=0 if metric in ["sharpe", "sortino", "total_return"] else None,
        ))
        fig.update_layout(title=f"{title} ({metric})", xaxis_title=param_x,
                          yaxis_title=param_y, height=500)
        return fig

    @staticmethod
    def plot_parallel_coordinate(
        results_df: pd.DataFrame, param_cols: List[str], metric_col: str,
        title: str = "平行坐标图",
    ) -> go.Figure:
        """平行坐标图，用于 ≥3 个参数的组合可视化。"""
        empty = OptimizationMixin._check_df(results_df, title)
        if empty:
            return empty
        dimensions = []
        for col in param_cols:
            dimensions.append(dict(label=col, values=results_df[col]))
        dimensions.append(dict(label=metric_col, values=results_df[metric_col]))
        fig = go.Figure(go.Parcoords(
            line=dict(color=results_df[metric_col], colorscale="RdYlGn",
                      showscale=True, colorbar=dict(title=metric_col)),
            dimensions=dimensions,
        ))
        fig.update_layout(title=title, template="plotly_white", height=500)
        return fig

    @staticmethod
    def plot_param_scan(
        results_df: pd.DataFrame, param_name: str, metric: str = "sharpe",
        extra_metrics: Optional[List[str]] = None, title: str = "参数扫描",
    ) -> go.Figure:
        """一维参数扫描线图。"""
        empty = OptimizationMixin._check_df(results_df, title)
        if empty:
            return empty
        if param_name not in results_df.columns:
            return OptimizationMixin._empty_fig(title, missing_cols=[param_name])
        fig = go.Figure()
        grouped = results_df.groupby(param_name)[metric].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=grouped[param_name], y=grouped[metric], mode="lines+markers",
            name=metric, line=dict(color=COLORS["blue"], width=2),
        ))
        if extra_metrics:
            for idx, em in enumerate(extra_metrics):
                if em in results_df.columns:
                    g = results_df.groupby(param_name)[em].mean().reset_index()
                    fig.add_trace(go.Scatter(
                        x=g[param_name], y=g[em], mode="lines+markers", name=em,
                        line=dict(color=OptimizationMixin._get_color(idx + 1), width=1.5, dash="dash"),
                    ))
        best_idx = grouped[metric].idxmax()
        best_val = grouped.loc[best_idx, param_name]
        fig.add_vline(x=best_val, line_dash="dash", line_color=COLORS["green"],
                      annotation_text=f"最优: {best_val}")
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title=param_name, yaxis_title="指标值")
        return fig

    @staticmethod
    def plot_param_stability(
        rolling_results_df: pd.DataFrame, param_name: str, title: str = "参数稳定性",
    ) -> go.Figure:
        """滚动优化结果中参数随窗口的变化折线图。"""
        empty = OptimizationMixin._check_df(rolling_results_df, title)
        if empty:
            return empty
        if param_name not in rolling_results_df.columns:
            return OptimizationMixin._empty_fig(title, missing_cols=[param_name])
        date_col = "window_date" if "window_date" in rolling_results_df.columns else rolling_results_df.columns[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_results_df[date_col], y=rolling_results_df[param_name],
            mode="lines+markers", name=param_name,
            line=dict(color=COLORS["blue"], width=2),
        ))
        mean_val = rolling_results_df[param_name].mean()
        fig.add_hline(y=mean_val, line_dash="dash", line_color=COLORS["orange"],
                      annotation_text=f"均值: {mean_val:.4f}")
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title="优化窗口", yaxis_title=param_name)
        return fig

    @staticmethod
    def plot_surface_3d(
        results_df: pd.DataFrame, x_param: str, y_param: str, z_metric: str,
        title: str = "3D 参数曲面",
    ) -> go.Figure:
        """三维表面图。"""
        empty = OptimizationMixin._check_df(results_df, title)
        if empty:
            return empty
        missing = [col for col in [x_param, y_param, z_metric] if col not in results_df.columns]
        if missing:
            return OptimizationMixin._empty_fig(title, missing_cols=missing)
        pivot = results_df.pivot_table(values=z_metric, index=y_param, columns=x_param, aggfunc="mean")
        fig = go.Figure(go.Surface(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale="RdYlGn", colorbar=dict(title=z_metric),
        ))
        fig.update_layout(title=title, height=600,
                          scene=dict(xaxis_title=x_param, yaxis_title=y_param, zaxis_title=z_metric))
        return fig

    @staticmethod
    def plot_param_importance(
        results_df: pd.DataFrame, param_cols: List[str], metric_col: str,
        title: str = "参数重要性",
    ) -> go.Figure:
        """各参数与目标指标的相关性绝对值条形图。"""
        empty = OptimizationMixin._check_df(results_df, title)
        if empty:
            return empty
        if metric_col not in results_df.columns:
            return OptimizationMixin._empty_fig(title, missing_cols=[metric_col])
        importances = {}
        for col in param_cols:
            if col in results_df.columns:
                importances[col] = abs(results_df[col].corr(results_df[metric_col]))
        if not importances:
            return OptimizationMixin._empty_fig(title, missing_cols=param_cols)
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_imp]
        values = [x[1] for x in sorted_imp]
        fig = go.Figure(go.Bar(
            x=names, y=values, marker_color=[COLORS["blue"]] * len(names),
            text=[f"{v:.3f}" for v in values], textposition="outside",
        ))
        fig.update_layout(title=title, template="plotly_white", height=400,
                          xaxis_title="参数", yaxis_title=f"|相关系数| vs {metric_col}")
        return fig
