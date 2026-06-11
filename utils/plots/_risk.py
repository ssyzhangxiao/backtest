"""图表绘制模块 — 风险与归因分析。"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

from utils.plots._base import COLORS, BasePlotMixin


class RiskMixin(BasePlotMixin):
    """风险与归因分析相关图表。"""

    @staticmethod
    def plot_risk_pie(risk_contrib_dict: Dict[str, float], title: str = "风险贡献") -> go.Figure:
        """饼图显示各品种/策略的风险贡献。"""
        if not risk_contrib_dict:
            return RiskMixin._empty_fig(title)
        labels = list(risk_contrib_dict.keys())
        values = [abs(v) for v in risk_contrib_dict.values()]
        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.4, textinfo="label+percent",
            marker=dict(colors=px.colors.qualitative.Set2[:len(labels)],
                        line=dict(color="white", width=2)),
        ))
        fig.update_layout(title=title, template="plotly_white", height=500)
        return fig

    @staticmethod
    def plot_concentration_curve(
        portfolio_df: pd.DataFrame, positions_df: Optional[pd.DataFrame] = None,
        title: str = "持仓集中度", equity_col: str = "equity", date_col: str = "date",
    ) -> go.Figure:
        """单合约市值占权益比例的时间序列。"""
        empty = RiskMixin._check_df(positions_df, title, missing_cols=["symbol", "market_value"])
        if empty:
            return empty
        empty = RiskMixin._check_df(portfolio_df, title, missing_cols=[equity_col, date_col])
        if empty:
            return empty
        positions_df = RiskMixin._ensure_date_column(positions_df, date_col=date_col)
        portfolio_df = RiskMixin._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = RiskMixin._get_equity_col(portfolio_df, equity_col=equity_col)
        fig = go.Figure()
        if eq_col and date_col in portfolio_df.columns:
            equity_map = portfolio_df.set_index(date_col)[eq_col]
            for idx, sym in enumerate(positions_df["symbol"].unique()):
                sub = positions_df[positions_df["symbol"] == sym].copy().sort_values(date_col)
                sub["equity"] = sub[date_col].map(equity_map)
                sub["concentration"] = sub["market_value"] / sub["equity"] * 100
                fig.add_trace(go.Scatter(
                    x=sub[date_col], y=sub["concentration"], mode="lines", name=sym,
                    line=dict(color=RiskMixin._get_color(idx), width=1.5),
                ))
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title="日期", yaxis_title="占比 (%)")
        return fig

    @staticmethod
    def plot_leverage_ratio(
        portfolio_df: pd.DataFrame, positions_df: Optional[pd.DataFrame] = None,
        threshold: float = 1.0, title: str = "杠杆率",
        equity_col: str = "equity", date_col: str = "date",
    ) -> go.Figure:
        """杠杆率曲线，标注安全阈值。"""
        empty = RiskMixin._check_df(portfolio_df, title)
        if empty:
            return empty
        df = RiskMixin._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = RiskMixin._get_equity_col(df, equity_col=equity_col)
        if not eq_col or date_col not in df.columns:
            return RiskMixin._empty_fig(title, missing_cols=[equity_col, date_col])
        fig = go.Figure()
        if positions_df is not None and not positions_df.empty:
            positions_df = RiskMixin._ensure_date_column(positions_df, date_col=date_col)
            total_exposure = positions_df.groupby(date_col)["market_value"].sum().abs()
            equity_series = df.set_index(date_col)[eq_col]
            leverage = total_exposure / equity_series
            fig.add_trace(go.Scatter(
                x=leverage.index, y=leverage, mode="lines", name="杠杆率",
                line=dict(color=COLORS["blue"], width=1.5),
            ))
        elif "margin" in df.columns:
            leverage = df["margin"] / df[eq_col]
            fig.add_trace(go.Scatter(
                x=df[date_col], y=leverage, mode="lines", name="杠杆率",
                line=dict(color=COLORS["blue"], width=1.5),
            ))
        fig.add_hline(y=threshold, line_dash="dash", line_color=COLORS["red"],
                      annotation_text=f"安全阈值: {threshold}")
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title="日期", yaxis_title="杠杆率")
        return fig

    @staticmethod
    def plot_stress_test(
        portfolio_df: pd.DataFrame, stress_events: List[Dict], title: str = "压力测试",
        equity_col: str = "equity", date_col: str = "date",
    ) -> go.Figure:
        """瀑布图：历史极端行情区间内策略累计收益。"""
        empty = RiskMixin._check_df(portfolio_df, title)
        if empty:
            return empty
        if not stress_events:
            return RiskMixin._empty_fig(title + "（无压力事件）")
        df = RiskMixin._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = RiskMixin._get_equity_col(df, equity_col=equity_col)
        if not eq_col or date_col not in df.columns:
            return RiskMixin._empty_fig(title, missing_cols=[equity_col, date_col])
        names, returns = [], []
        for event in stress_events:
            start, end = pd.to_datetime(event["start"]), pd.to_datetime(event["end"])
            mask = (df[date_col] >= start) & (df[date_col] <= end)
            sub = df.loc[mask]
            ret = (sub[eq_col].iloc[-1] / sub[eq_col].iloc[0] - 1) * 100 if not sub.empty else 0
            names.append(event["name"])
            returns.append(ret)
        colors = [COLORS["green"] if r >= 0 else COLORS["red"] for r in returns]
        fig = go.Figure(go.Bar(
            x=names, y=returns, marker_color=colors,
            text=[f"{r:.2f}%" for r in returns], textposition="outside",
        ))
        fig.add_hline(y=0, line_color="gray", line_width=0.5)
        fig.update_layout(title=title, template="plotly_white", height=400,
                          xaxis_title="压力事件", yaxis_title="收益率 (%)")
        return fig

    @staticmethod
    def plot_correlation_heatmap(
        returns_df: pd.DataFrame, method: str = "pearson", title: str = "相关性热力图",
    ) -> go.Figure:
        """多品种日收益率相关性热力图。"""
        empty = RiskMixin._check_df(returns_df, title)
        if empty:
            return empty
        corr = returns_df.corr(method=method)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu_r", zmid=0,
            text=np.round(corr.values, 3), texttemplate="%{text}",
            colorbar=dict(title="相关系数"),
        ))
        fig.update_layout(title=f"{title} ({method})", template="plotly_white", height=500)
        return fig

    @staticmethod
    def plot_up_down_capture(
        returns_df: Union[pd.Series, pd.DataFrame],
        benchmark_returns: Union[pd.Series, pd.DataFrame],
        title: str = "上下捕获比率",
    ) -> go.Figure:
        """上/下捕获比率柱状图。"""
        if returns_df is None or benchmark_returns is None:
            return RiskMixin._empty_fig(title)
        if isinstance(returns_df, pd.DataFrame):
            returns_df = returns_df.iloc[:, 0]
        if isinstance(benchmark_returns, pd.DataFrame):
            benchmark_returns = benchmark_returns.iloc[:, 0]
        up_mask = benchmark_returns > 0
        down_mask = benchmark_returns < 0
        up_capture = (returns_df[up_mask].mean() / benchmark_returns[up_mask].mean()) * 100 if np.any(up_mask.values) else 0
        down_capture = (returns_df[down_mask].mean() / benchmark_returns[down_mask].mean()) * 100 if np.any(down_mask.values) else 0
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["上捕获比率", "下捕获比率"], y=[up_capture, down_capture],
            marker_color=[COLORS["green"], COLORS["red"]],
            text=[f"{up_capture:.1f}%", f"{down_capture:.1f}%"], textposition="outside",
        ))
        fig.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="基准线")
        fig.update_layout(title=title, template="plotly_white", height=400, yaxis_title="捕获比率 (%)")
        return fig

    @staticmethod
    def plot_rolling_var(
        returns_df: Union[pd.Series, pd.DataFrame], window: int = 252,
        ci: float = 0.95, title: str = "滚动 VaR",
    ) -> go.Figure:
        """滚动 VaR（历史模拟法）曲线。"""
        if returns_df is None or (hasattr(returns_df, "empty") and returns_df.empty):
            return RiskMixin._empty_fig(title)
        if isinstance(returns_df, pd.DataFrame):
            returns_df = returns_df.iloc[:, 0]
        returns_df = returns_df.dropna()

        def _calc_var(window_data):
            if len(window_data) < 10:
                return np.nan
            return np.percentile(window_data, (1 - ci) * 100)

        rolling_var = returns_df.rolling(window).apply(_calc_var, raw=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_var.index, y=rolling_var, mode="lines",
            name=f"VaR({ci:.0%})", line=dict(color=COLORS["red"], width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=returns_df.index, y=returns_df, mode="lines", name="日收益率",
            line=dict(color=COLORS["blue"], width=0.5), opacity=0.3,
        ))
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title="日期", yaxis_title="收益率 / VaR")
        return fig
