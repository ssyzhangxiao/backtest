"""图表绘制模块 — 净值/回撤/收益/滚动指标。"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, Union

from utils.plots._base import COLORS, BasePlotMixin


class EquityMixin(BasePlotMixin):
    """净值/回撤/收益/滚动指标相关图表。"""

    @staticmethod
    def plot_equity_curve(
        portfolio_df: pd.DataFrame, benchmark_series: Optional[pd.Series] = None,
        title: str = "资金曲线", log_y: bool = False,
        line_color: Optional[str] = None, fill_color: Optional[str] = None,
        equity_col: str = "equity", date_col: str = "date",
    ) -> go.Figure:
        """净值曲线 + 回撤曲线，可选基准叠加和对数坐标。"""
        from plotly.subplots import make_subplots

        empty = EquityMixin._check_df(portfolio_df, title)
        if empty:
            return empty
        df = EquityMixin._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = EquityMixin._get_equity_col(df, equity_col=equity_col)
        _line_color = line_color or COLORS["blue"]
        _fill_color = fill_color or "rgba(214, 39, 40, 0.15)"
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3], vertical_spacing=0.08)
        if eq_col and date_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_col], y=df[eq_col], mode="lines", name="账户净值",
                line=dict(color=_line_color, width=2),
            ), row=1, col=1)
            initial = df[eq_col].iloc[0]
            fig.add_hline(y=initial, line_dash="dash", line_color="gray",
                          annotation_text="初始资金", row=1, col=1)
            peak = df[eq_col].cummax()
            drawdown = (df[eq_col] - peak) / peak * 100
            fig.add_trace(go.Scatter(
                x=df[date_col], y=drawdown, mode="lines", name="回撤%",
                fill="tozeroy", line=dict(color=COLORS["red"], width=1),
                fillcolor=_fill_color,
            ), row=2, col=1)
        if benchmark_series is not None and not benchmark_series.empty:
            bench = benchmark_series.reset_index()
            bench.columns = [date_col, "benchmark"]
            fig.add_trace(go.Scatter(
                x=bench[date_col], y=bench["benchmark"], mode="lines", name="基准",
                line=dict(color=COLORS["orange"], width=1.5, dash="dash"),
            ), row=1, col=1)
        fig.update_layout(title=title, template="plotly_white", height=600, hovermode="x unified")
        fig.update_yaxes(title_text="净值", type="log" if log_y else "linear", row=1, col=1)
        fig.update_yaxes(title_text="回撤 (%)", tickformat=".0f%", row=2, col=1)
        return fig

    @staticmethod
    def plot_log_returns(
        daily_returns: Union[pd.Series, pd.DataFrame], title: str = "对数收益率",
    ) -> go.Figure:
        """对数收益率曲线 + 波动率条带（±1σ, ±2σ）。"""
        if daily_returns is None or (hasattr(daily_returns, "empty") and daily_returns.empty):
            return EquityMixin._empty_fig(title)
        if isinstance(daily_returns, pd.DataFrame):
            daily_returns = daily_returns.iloc[:, 0]
        log_ret = np.log1p(daily_returns / 100) if daily_returns.abs().max() > 1 else np.log1p(daily_returns)
        std, mean = float(log_ret.std()), float(log_ret.mean())
        upper_2 = np.full(len(log_ret), mean + 2 * std)
        lower_2 = np.full(len(log_ret), mean - 2 * std)
        upper_1 = np.full(len(log_ret), mean + std)
        lower_1 = np.full(len(log_ret), mean - std)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=log_ret.index, y=upper_2, mode="lines",
                                  line=dict(color="rgba(214,39,40,0.3)", width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=log_ret.index, y=lower_2, mode="lines", fill="tonexty",
                                  line=dict(color="rgba(214,39,40,0.3)", width=0),
                                  fillcolor="rgba(214,39,40,0.08)", name="±2σ"))
        fig.add_trace(go.Scatter(x=log_ret.index, y=upper_1, mode="lines",
                                  line=dict(color="rgba(255,152,0,0.4)", width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=log_ret.index, y=lower_1, mode="lines", fill="tonexty",
                                  line=dict(color="rgba(255,152,0,0.4)", width=0),
                                  fillcolor="rgba(255,152,0,0.12)", name="±1σ"))
        fig.add_trace(go.Scatter(x=log_ret.index, y=log_ret, mode="lines", name="对数收益率",
                                  line=dict(color=COLORS["blue"], width=1)))
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title="日期", yaxis_title="对数收益率")
        return fig

    @staticmethod
    def plot_drawdown(
        portfolio_df: pd.DataFrame, title: str = "回撤曲线",
        line_color: Optional[str] = None, fill_color: Optional[str] = None,
        equity_col: str = "equity", date_col: str = "date",
    ) -> go.Figure:
        """回撤曲线，标注最大回撤点。"""
        empty = EquityMixin._check_df(portfolio_df, title)
        if empty:
            return empty
        df = EquityMixin._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = EquityMixin._get_equity_col(df, equity_col=equity_col)
        _line_color = line_color or COLORS["red"]
        _fill_color = fill_color or "rgba(214, 39, 40, 0.15)"
        fig = go.Figure()
        if eq_col and date_col in df.columns:
            equity = df[eq_col]
            peak = equity.cummax()
            drawdown = (equity - peak) / peak * 100
            fig.add_trace(go.Scatter(
                x=df[date_col], y=drawdown, mode="lines", name="回撤%", fill="tozeroy",
                line=dict(color=_line_color, width=1.5),
                fillcolor=_fill_color,
            ))
            min_dd_idx = drawdown.idxmin()
            if pd.notna(min_dd_idx):
                fig.add_trace(go.Scatter(
                    x=[df.loc[min_dd_idx, date_col]], y=[drawdown.loc[min_dd_idx]],
                    mode="markers", name="最大回撤",
                    marker=dict(color=COLORS["red"], size=10, symbol="x"),
                ))
                fig.add_annotation(
                    x=df.loc[min_dd_idx, date_col], y=drawdown.loc[min_dd_idx],
                    text=f"最大回撤: {drawdown.loc[min_dd_idx]:.2f}%",
                    showarrow=True, arrowhead=2, font=dict(color=COLORS["red"]),
                )
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title="日期", yaxis_title="回撤 (%)")
        fig.update_yaxes(tickformat=".0f%")
        return fig

    @staticmethod
    def plot_monthly_heatmap(
        portfolio_df: pd.DataFrame, title: str = "月度收益率热力图",
        equity_col: str = "equity", date_col: str = "date",
    ) -> go.Figure:
        """月度收益率热力图（行=年份，列=月份）。"""
        empty = EquityMixin._check_df(portfolio_df, title)
        if empty:
            return empty
        df = EquityMixin._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = EquityMixin._get_equity_col(df, equity_col=equity_col)
        if not eq_col or date_col not in df.columns:
            return EquityMixin._empty_fig(title, missing_cols=[equity_col, date_col])
        df[date_col] = pd.to_datetime(df[date_col])
        df["year"] = df[date_col].dt.year
        df["month"] = df[date_col].dt.month
        monthly = df.groupby(["year", "month"])[eq_col].agg(["first", "last"])
        monthly["return"] = (monthly["last"] / monthly["first"] - 1) * 100
        monthly = monthly.reset_index()
        pivot = monthly.pivot(index="year", columns="month", values="return")
        month_names = ["1月", "2月", "3月", "4月", "5月", "6月",
                       "7月", "8月", "9月", "10月", "11月", "12月"]
        existing_months = [month_names[m - 1] for m in pivot.columns]
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=existing_months, y=pivot.index,
            colorscale="RdYlGn", zmid=0,
            text=np.round(pivot.values, 2), texttemplate="%{text}%",
            colorbar=dict(title="收益率%"),
        ))
        fig.update_layout(title=title, template="plotly_white",
                          height=max(400, len(pivot) * 30),
                          xaxis_title="月份", yaxis_title="年份")
        return fig

    @staticmethod
    def plot_rolling_sharpe(
        daily_returns: Union[pd.Series, pd.DataFrame], window: int = 252,
        risk_free_rate: float = 0.0, title: str = "滚动夏普比率",
        line_color: Optional[str] = None,
    ) -> go.Figure:
        """滚动夏普比率曲线（年化），附均值线。"""
        if daily_returns is None or (hasattr(daily_returns, "empty") and daily_returns.empty):
            return EquityMixin._empty_fig(title)
        if isinstance(daily_returns, pd.DataFrame):
            daily_returns = daily_returns.iloc[:, 0]
        daily_rf = risk_free_rate / 252
        excess = daily_returns - daily_rf
        rolling_sharpe = (
            excess.rolling(window).mean() / excess.rolling(window).std()
        ) * np.sqrt(252)
        _line_color = line_color or COLORS["blue"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe, mode="lines",
            name=f"夏普({window}日)", line=dict(color=_line_color, width=1.5),
        ))
        mean_sharpe = rolling_sharpe.mean()
        fig.add_hline(y=mean_sharpe, line_dash="dash", line_color=COLORS["orange"],
                      annotation_text=f"均值: {mean_sharpe:.2f}")
        fig.add_hline(y=0, line_color="gray", line_width=0.5)
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title="日期", yaxis_title="年化夏普比率")
        fig.update_yaxes(tickformat=".2f")
        return fig

    @staticmethod
    def plot_rolling_max_drawdown(
        portfolio_df: pd.DataFrame, window: int = 252, title: str = "滚动最大回撤",
        line_color: Optional[str] = None, fill_color: Optional[str] = None,
        equity_col: str = "equity", date_col: str = "date",
    ) -> go.Figure:
        """滚动最大回撤曲线。"""
        empty = EquityMixin._check_df(portfolio_df, title)
        if empty:
            return empty
        df = EquityMixin._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = EquityMixin._get_equity_col(df, equity_col=equity_col)
        if not eq_col or date_col not in df.columns:
            return EquityMixin._empty_fig(title, missing_cols=[equity_col, date_col])
        equity = df.set_index(date_col)[eq_col]
        peak = equity.cummax()
        dd = (equity - peak) / peak
        rolling_max_dd = dd.rolling(window).min() * 100
        _line_color = line_color or COLORS["red"]
        _fill_color = fill_color or "rgba(214,39,40,0.1)"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_max_dd.index, y=rolling_max_dd, mode="lines",
            name=f"最大回撤({window}日)", fill="tozeroy",
            line=dict(color=_line_color, width=1.5),
            fillcolor=_fill_color,
        ))
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title="日期", yaxis_title="最大回撤 (%)")
        fig.update_yaxes(tickformat=".0f%")
        return fig

    @staticmethod
    def plot_pnl_distribution(
        trades_df: pd.DataFrame, title: str = "盈亏分布",
        pnl_col: str = "pnl_pct",
    ) -> go.Figure:
        """单笔盈亏直方图 + 核密度曲线。"""
        empty = EquityMixin._check_df(trades_df, title)
        if empty:
            return empty
        if pnl_col not in trades_df.columns:
            pnl_col = "pnl"
        if pnl_col not in trades_df.columns:
            return EquityMixin._empty_fig(title, missing_cols=["pnl_pct", "pnl"])
        pnl = trades_df[pnl_col].dropna()
        if pnl.empty:
            return EquityMixin._empty_fig(title)
        win_rate = (pnl > 0).mean() * 100
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=pnl, nbinsx=50, name="盈亏分布",
            marker_color=COLORS["blue"], opacity=0.7, histnorm="probability density",
        ))
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(pnl.values)
            x_range = np.linspace(pnl.min(), pnl.max(), 200)
            fig.add_trace(go.Scatter(
                x=x_range, y=kde(x_range), mode="lines", name="核密度",
                line=dict(color=COLORS["red"], width=2),
            ))
        except ImportError:
            pass
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.add_annotation(
            x=0.98, y=0.98, xref="paper", yref="paper", text=f"胜率: {win_rate:.1f}%",
            showarrow=False,
            font=dict(size=14, color=COLORS["green"] if win_rate > 50 else COLORS["red"]),
        )
        x_suffix = "%" if pnl_col == "pnl_pct" else ""
        fig.update_layout(title=title, template="plotly_white", height=400,
                          xaxis_title=f"盈亏({x_suffix})",
                          yaxis_title="概率密度")
        if x_suffix:
            fig.update_xaxes(ticksuffix="%")
        return fig

    @staticmethod
    def plot_qq_plot(
        daily_returns: Union[pd.Series, pd.DataFrame], title: str = "Q-Q 正态分位数图",
    ) -> go.Figure:
        """正态分位数图（Q-Q plot）。"""
        if daily_returns is None or (hasattr(daily_returns, "empty") and daily_returns.empty):
            return EquityMixin._empty_fig(title)
        if isinstance(daily_returns, pd.DataFrame):
            daily_returns = daily_returns.iloc[:, 0]
        daily_returns = daily_returns.dropna()
        try:
            from scipy.stats import probplot
            (theoretical_q, ordered_values), (slope, intercept, r) = probplot(
                daily_returns, dist="norm")
        except ImportError:
            return EquityMixin._empty_fig(title + "（需要 scipy）", missing_cols=["scipy"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=theoretical_q, y=ordered_values, mode="markers", name="样本分位数",
            marker=dict(color=COLORS["blue"], size=4),
        ))
        fit_line = slope * theoretical_q + intercept
        fig.add_trace(go.Scatter(
            x=theoretical_q, y=fit_line, mode="lines", name="正态拟合线",
            line=dict(color=COLORS["red"], width=2),
        ))
        fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                           text=f"R² = {r**2:.4f}", showarrow=False, font=dict(size=12))
        fig.update_layout(title=title, template="plotly_white", height=500,
                          xaxis_title="理论分位数", yaxis_title="样本分位数")
        return fig
