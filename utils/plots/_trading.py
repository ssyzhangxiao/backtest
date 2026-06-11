"""图表绘制模块 — 交易执行分析。"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional

from utils.plots._base import COLORS, BasePlotMixin


class TradingMixin(BasePlotMixin):
    """交易执行分析相关图表。"""

    @staticmethod
    def plot_vwap_scatter(
        orders_df: pd.DataFrame, title: str = "成交价 vs VWAP",
        date_col: str = "date",
    ) -> go.Figure:
        """成交价 vs VWAP 散点图，参考线 y=x。"""
        empty = TradingMixin._check_df(orders_df, title)
        if empty:
            return empty
        orders_df = TradingMixin._ensure_date_column(orders_df, date_col=date_col)
        fill_col = "filled_price" if "filled_price" in orders_df.columns else "fill_price"
        if fill_col not in orders_df.columns:
            fill_col = "price"
        if "vwap" not in orders_df.columns:
            return TradingMixin._empty_fig(title, missing_cols=["vwap"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=orders_df["vwap"], y=orders_df[fill_col], mode="markers", name="成交",
            marker=dict(color=COLORS["blue"], size=6, opacity=0.6),
        ))
        all_vals = pd.concat([orders_df["vwap"], orders_df[fill_col]]).dropna()
        if not all_vals.empty:
            lo, hi = all_vals.min(), all_vals.max()
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines", name="y=x",
                line=dict(color="gray", dash="dash", width=1),
            ))
        fig.update_layout(title=title, template="plotly_white", height=500,
                          xaxis_title="VWAP", yaxis_title="成交价")
        return fig

    @staticmethod
    def plot_holding_histogram(
        trades_df: pd.DataFrame, title: str = "持仓时长分布",
    ) -> go.Figure:
        """持仓时长直方图。"""
        empty = TradingMixin._check_df(trades_df, title)
        if empty:
            return empty
        if "bars" in trades_df.columns:
            holding = trades_df["bars"].dropna()
            x_label = "持仓时长 (bars)"
        elif "entry_date" in trades_df.columns and "exit_date" in trades_df.columns:
            holding = (pd.to_datetime(trades_df["exit_date"]) - pd.to_datetime(trades_df["entry_date"])).dt.days.dropna()
            x_label = "持仓时长 (天)"
        else:
            return TradingMixin._empty_fig(title, missing_cols=["bars", "entry_date", "exit_date"])
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=holding, nbinsx=50, name="持仓时长",
            marker_color=COLORS["blue"], opacity=0.7,
        ))
        mean_holding = holding.mean()
        fig.add_vline(x=mean_holding, line_dash="dash", line_color=COLORS["red"],
                      annotation_text=f"均值: {mean_holding:.1f}")
        fig.update_layout(title=title, template="plotly_white", height=400,
                          xaxis_title=x_label, yaxis_title="频次")
        return fig

    @staticmethod
    def plot_daily_trades_count(
        trades_df: pd.DataFrame, title: str = "每日交易次数",
        date_col: str = "date",
    ) -> go.Figure:
        """每日交易次数条形图。"""
        empty = TradingMixin._check_df(trades_df, title)
        if empty:
            return empty
        trades_df = TradingMixin._ensure_date_column(trades_df, date_col=date_col)
        if "type" in trades_df.columns:
            daily = trades_df.groupby([trades_df[date_col].dt.date, "type"]).size().unstack(fill_value=0)
            daily.index = pd.to_datetime(daily.index)
            fig = go.Figure()
            for col in daily.columns:
                color = COLORS["green"] if "buy" in str(col).lower() else COLORS["red"]
                fig.add_trace(go.Bar(x=daily.index, y=daily[col], name=str(col), marker_color=color))
        else:
            daily = trades_df.groupby(trades_df[date_col].dt.date).size()
            daily.index = pd.to_datetime(daily.index)
            fig = go.Figure(go.Bar(x=daily.index, y=daily.values, name="交易次数",
                                    marker_color=COLORS["blue"]))
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title="日期", yaxis_title="交易次数", barmode="stack")
        return fig

    @staticmethod
    def plot_pnl_by_symbol(
        trades_df: pd.DataFrame, title: str = "各品种盈亏分布",
        pnl_col: str = "pnl_pct",
    ) -> go.Figure:
        """按品种分组的盈亏箱线图。"""
        empty = TradingMixin._check_df(trades_df, title)
        if empty:
            return empty
        if pnl_col not in trades_df.columns:
            pnl_col = "pnl"
        if pnl_col not in trades_df.columns or "symbol" not in trades_df.columns:
            return TradingMixin._empty_fig(title, missing_cols=["pnl_pct", "pnl", "symbol"])
        fig = go.Figure()
        for sym in trades_df["symbol"].unique():
            sub = trades_df[trades_df["symbol"] == sym][pnl_col].dropna()
            fig.add_trace(go.Box(y=sub, name=sym, boxmean="sd"))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(title=title, template="plotly_white", height=500,
                          yaxis_title="盈亏" + ("(%)" if pnl_col == "pnl_pct" else ""),
                          showlegend=False)
        return fig

    @staticmethod
    def plot_slippage_time(
        orders_df: pd.DataFrame, cumulative: bool = False, title: str = "滑点分析",
        date_col: str = "date",
    ) -> go.Figure:
        """滑点时间序列曲线。"""
        empty = TradingMixin._check_df(orders_df, title)
        if empty:
            return empty
        orders_df = TradingMixin._ensure_date_column(orders_df, date_col=date_col)
        fill_col = "filled_price" if "filled_price" in orders_df.columns else "fill_price"
        missing = []
        if fill_col not in orders_df.columns:
            missing.append("filled_price")
        if "price" not in orders_df.columns:
            missing.append("price")
        if missing:
            return TradingMixin._empty_fig(title, missing_cols=missing)
        slippage = orders_df[fill_col] - orders_df["price"]
        if cumulative:
            slippage = slippage.cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=orders_df[date_col], y=slippage, mode="lines+markers",
            name="累积滑点" if cumulative else "滑点",
            line=dict(color=COLORS["purple"], width=1.5), marker=dict(size=4),
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title="日期", yaxis_title="滑点")
        return fig

    @staticmethod
    def plot_rollover_cost_curve(
        rollover_costs_df: pd.DataFrame, title: str = "展期成本累积曲线",
        date_col: str = "date",
    ) -> go.Figure:
        """展期成本累积曲线。"""
        empty = TradingMixin._check_df(rollover_costs_df, title)
        if empty:
            return empty
        rollover_costs_df = TradingMixin._ensure_date_column(rollover_costs_df, date_col=date_col)
        if "cost" not in rollover_costs_df.columns:
            return TradingMixin._empty_fig(title, missing_cols=["cost"])
        df = rollover_costs_df.sort_values(date_col)
        df["cum_cost"] = df["cost"].cumsum()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.4], vertical_spacing=0.08)
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df["cum_cost"], mode="lines", name="累积成本",
            line=dict(color=COLORS["red"], width=2),
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=df[date_col], y=df["cost"], name="单次成本",
            marker_color=COLORS["orange"],
        ), row=2, col=1)
        fig.update_layout(title=title, template="plotly_white", height=500, hovermode="x unified")
        fig.update_yaxes(title_text="累积成本", row=1, col=1)
        fig.update_yaxes(title_text="单次成本", row=2, col=1)
        return fig
