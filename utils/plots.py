"""
图表绘制模块。

使用 Plotly 绘制回测结果可视化图表，供 Streamlit 前端调用。
覆盖六大模块：数据概览、策略绩效、风险归因、交易执行、参数优化、市场状态。
所有图表返回 plotly.graph_objects.Figure 对象，可直接用于 st.plotly_chart()。
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

_COLORS = {
    "blue": "#1f77b4", "orange": "#ff7f0e", "green": "#2ca02c",
    "red": "#d62728", "purple": "#9467bd", "brown": "#8c564b",
    "pink": "#e377c2", "gray": "#7f7f7f", "olive": "#bcbd22",
    "cyan": "#17becf",
}

_REGIME_COLORS = {
    "trend": "rgba(76, 175, 80, 0.12)",
    "range": "rgba(255, 193, 7, 0.12)",
}


class PlotManager:

    @staticmethod
    def _ensure_date_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        df = df.copy()
        # 优先找 date_col
        if date_col in df.columns:
            return df
        # 找备选列名
        alt_date_cols = ["entry_date", "exit_date", "trade_date", "date"]
        found_col = None
        for col in alt_date_cols:
            if col in df.columns:
                found_col = col
                break
        if found_col and found_col != date_col:
            df[date_col] = df[found_col]
            return df
        # 检查索引
        if df.index.name == date_col:
            df = df.reset_index()
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if "index" in df.columns:
                df = df.rename(columns={"index": date_col})
        return df

    @staticmethod
    def _get_equity_col(df: pd.DataFrame, equity_col: str = "equity") -> str:
        if equity_col in df.columns:
            return equity_col
        if "market_value" in df.columns:
            return "market_value"
        return ""

    @classmethod
    def _get_color(cls, idx: int) -> str:
        colors = list(_COLORS.values())
        return colors[idx % len(colors)]

    @staticmethod
    def _empty_fig(title: str = "", height: int = 400, missing_cols: Optional[List[str]] = None) -> go.Figure:
        text = "暂无数据"
        if missing_cols:
            text = f"缺少列: {', '.join(missing_cols)}"
        fig = go.Figure()
        fig.update_layout(
            title=title, template="plotly_white", height=height,
            annotations=[dict(
                text=text, xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray"),
            )],
        )
        return fig

    @staticmethod
    def _check_df(df, title: str, missing_cols: Optional[List[str]] = None) -> Optional[go.Figure]:
        if df is None or (hasattr(df, "empty") and df.empty):
            return PlotManager._empty_fig(title, missing_cols=missing_cols)
        return None

    # ================================================================
    # 模块1：数据概览与预处理
    # ================================================================

    @staticmethod
    def plot_price_with_volume(
        df: pd.DataFrame, symbol: Optional[str] = None, title: Optional[str] = None,
        date_col: str = "date",
    ) -> go.Figure:
        """K线图（蜡烛图）+ 成交量柱状图。

        Args:
            df: 包含 date, open, high, low, close, volume 的 DataFrame
            symbol: 合约名称，用于标题
            title: 图表标题
            date_col: 日期列名

        Returns:
            Plotly Figure 对象
        """
        _title = title or (f"{symbol} 价格与成交量" if symbol else "价格与成交量")
        empty = PlotManager._check_df(df, _title, missing_cols=["open", "high", "low", "close"])
        if empty:
            return empty
        df = PlotManager._ensure_date_column(df, date_col=date_col)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.05)
        fig.add_trace(go.Candlestick(
            x=df[date_col], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="K线",
            increasing_line_color=_COLORS["red"],
            decreasing_line_color=_COLORS["green"],
        ), row=1, col=1)
        if "volume" in df.columns:
            colors = np.where(df["close"] >= df["open"], _COLORS["red"], _COLORS["green"])
            fig.add_trace(go.Bar(
                x=df[date_col], y=df["volume"], name="成交量",
                marker_color=colors, opacity=0.7,
            ), row=2, col=1)
        fig.update_layout(title=_title, template="plotly_white", height=600,
                          xaxis_rangeslider_visible=False, hovermode="x unified")
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text="成交量", row=2, col=1)
        return fig

    @staticmethod
    def plot_rollover_timeline(
        rollover_df: pd.DataFrame, price_df: Optional[pd.DataFrame] = None,
        title: str = "展期时间线", date_col: str = "date",
    ) -> go.Figure:
        """时间线标注主力切换事件，可选价格背景。

        Args:
            rollover_df: 展期记录，包含 date, prev_dominant, dominant
            price_df: 可选价格数据，包含 date, close
            title: 图表标题
            date_col: 日期列名

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(rollover_df, title)
        if empty:
            return empty
        rollover_df = PlotManager._ensure_date_column(rollover_df, date_col=date_col)
        fig = go.Figure()
        if price_df is not None and not price_df.empty:
            price_df = PlotManager._ensure_date_column(price_df, date_col=date_col)
            fig.add_trace(go.Scatter(
                x=price_df[date_col], y=price_df["close"],
                mode="lines", name="收盘价",
                line=dict(color=_COLORS["gray"], width=1),
            ))
        for _, row in rollover_df.iterrows():
            fig.add_vline(
                x=row[date_col], line_width=1.5, line_dash="dash",
                line_color=_COLORS["red"],
                annotation_text=f"{row.get('prev_dominant', '')}→{row.get('dominant', '')}",
                annotation_position="top left", annotation_font_size=9,
            )
        fig.update_layout(title=title, template="plotly_white", height=400, hovermode="x unified")
        return fig

    @staticmethod
    def plot_open_interest_volume(
        df: pd.DataFrame, symbol_list: Optional[List[str]] = None,
        title: str = "持仓量与成交量", date_col: str = "date",
    ) -> go.Figure:
        """多合约持仓量和成交量曲线。

        Args:
            df: 包含 date, symbol, open_interest, volume 的 DataFrame
            symbol_list: 合约列表
            title: 图表标题
            date_col: 日期列名

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(df, title)
        if empty:
            return empty
        df = PlotManager._ensure_date_column(df, date_col=date_col)
        if symbol_list:
            df = df[df["symbol"].isin(symbol_list)]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.5], vertical_spacing=0.08)
        for idx, sym in enumerate(df["symbol"].unique()):
            sub = df[df["symbol"] == sym]
            c = PlotManager._get_color(idx)
            if "open_interest" in sub.columns:
                fig.add_trace(go.Scatter(
                    x=sub[date_col], y=sub["open_interest"], mode="lines",
                    name=f"{sym} 持仓量", line=dict(color=c, width=1.5),
                ), row=1, col=1)
            if "volume" in sub.columns:
                fig.add_trace(go.Scatter(
                    x=sub[date_col], y=sub["volume"], mode="lines",
                    name=f"{sym} 成交量", line=dict(color=c, width=1, dash="dot"),
                ), row=2, col=1)
        fig.update_layout(title=title, template="plotly_white", height=600, hovermode="x unified")
        fig.update_yaxes(title_text="持仓量", row=1, col=1)
        fig.update_yaxes(title_text="成交量", row=2, col=1)
        return fig

    @staticmethod
    def plot_spread_heatmap(
        df: pd.DataFrame, product: Optional[str] = None,
        contract_list: Optional[List[str]] = None, title: str = "价差热力图",
        date_col: str = "date",
    ) -> go.Figure:
        """合约两两收盘价价差热力图。

        Args:
            df: 包含 date, symbol, close 的 DataFrame
            product: 品种名称
            contract_list: 合约列表
            title: 图表标题
            date_col: 日期列名

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(df, title)
        if empty:
            return empty
        df = PlotManager._ensure_date_column(df, date_col=date_col)
        if contract_list is None:
            last_date = df[date_col].max()
            contract_list = sorted(df[df[date_col] == last_date]["symbol"].unique().tolist())
        if len(contract_list) < 2:
            return PlotManager._empty_fig(title + "（合约数不足）")
        pivot = df[df["symbol"].isin(contract_list)].pivot_table(
            index=date_col, columns="symbol", values="close")
        pivot = pivot.dropna(how="all")
        n = len(contract_list)
        spread_matrix = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                if i != j and contract_list[i] in pivot.columns and contract_list[j] in pivot.columns:
                    spread_matrix[i, j] = (pivot[contract_list[i]] - pivot[contract_list[j]]).mean()
        fig = go.Figure(go.Heatmap(
            z=spread_matrix, x=contract_list, y=contract_list,
            colorscale="RdBu_r", text=np.round(spread_matrix, 2),
            texttemplate="%{text}", colorbar=dict(title="价差"),
        ))
        fig.update_layout(
            title=f"{title} ({product})" if product else title,
            template="plotly_white", height=500, xaxis_title="合约", yaxis_title="合约")
        return fig

    @staticmethod
    def plot_missing_data_heatmap(
        df: pd.DataFrame, date_col: str = "date", value_col: str = "close",
        title: str = "数据缺失热力图",
    ) -> go.Figure:
        """日历热力图显示数据缺失情况。

        Args:
            df: 数据 DataFrame
            date_col: 日期列名
            value_col: 数值列名
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(df, title)
        if empty:
            return empty
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        if "symbol" in df.columns:
            df["year"] = df[date_col].dt.year
            df["month"] = df[date_col].dt.month
            coverage = df.groupby(["symbol", "year", "month"]).size().reset_index(name="count")
            pivot = coverage.pivot_table(
                index="symbol", columns=["year", "month"], values="count", fill_value=0)
            col_labels = [f"{y}-{m:02d}" for y, m in pivot.columns]
            fig = go.Figure(go.Heatmap(
                z=pivot.values, x=col_labels, y=pivot.index,
                colorscale="YlGn", colorbar=dict(title="数据条数"),
            ))
            fig.update_layout(title=title, template="plotly_white",
                              height=max(400, len(pivot) * 20),
                              xaxis_title="年-月", yaxis_title="合约")
        else:
            df["year"] = df[date_col].dt.year
            df["month"] = df[date_col].dt.month
            df["day"] = df[date_col].dt.day
            coverage = df.groupby(["year", "month", "day"]).size().reset_index(name="count")
            coverage["date_str"] = coverage.apply(
                lambda r: f"{r['year']}-{r['month']:02d}-{r['day']:02d}", axis=1)
            fig = go.Figure(go.Scatter(
                x=coverage["date_str"], y=coverage["count"], mode="markers",
                marker=dict(color=coverage["count"], colorscale="YlGn", size=5),
                name="数据条数",
            ))
            fig.update_layout(title=title, template="plotly_white", height=400,
                              hovermode="x unified", xaxis_title="日期", yaxis_title="数据条数")
        return fig

    # ================================================================
    # 模块2：策略绩效核心图表
    # ================================================================

    @staticmethod
    def plot_equity_curve(
        portfolio_df: pd.DataFrame, benchmark_series: Optional[pd.Series] = None,
        title: str = "资金曲线", log_y: bool = False,
        line_color: Optional[str] = None, fill_color: Optional[str] = None,
        equity_col: str = "equity", date_col: str = "date",
    ) -> go.Figure:
        """净值曲线 + 回撤曲线，可选基准叠加和对数坐标。

        Args:
            portfolio_df: 组合明细 DataFrame
            benchmark_series: 可选基准序列
            title: 图表标题
            log_y: 是否使用对数坐标
            line_color: 净值线颜色
            fill_color: 回撤填充颜色
            equity_col: 净值列名
            date_col: 日期列名

        Returns:
            Plotly Figure 对象

        Note: When using in Streamlit, decorate the caller with @st.cache_data to avoid recomputation.
        """
        empty = PlotManager._check_df(portfolio_df, title)
        if empty:
            return empty
        df = PlotManager._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = PlotManager._get_equity_col(df, equity_col=equity_col)
        _line_color = line_color or _COLORS["blue"]
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
                fill="tozeroy", line=dict(color=_COLORS["red"], width=1),
                fillcolor=_fill_color,
            ), row=2, col=1)
        if benchmark_series is not None and not benchmark_series.empty:
            bench = benchmark_series.reset_index()
            bench.columns = [date_col, "benchmark"]
            fig.add_trace(go.Scatter(
                x=bench[date_col], y=bench["benchmark"], mode="lines", name="基准",
                line=dict(color=_COLORS["orange"], width=1.5, dash="dash"),
            ), row=1, col=1)
        fig.update_layout(title=title, template="plotly_white", height=600, hovermode="x unified")
        fig.update_yaxes(title_text="净值", type="log" if log_y else "linear", row=1, col=1)
        fig.update_yaxes(title_text="回撤 (%)", tickformat=".0f%", row=2, col=1)
        return fig

    @staticmethod
    def plot_log_returns(
        daily_returns: Union[pd.Series, pd.DataFrame], title: str = "对数收益率",
    ) -> go.Figure:
        """对数收益率曲线 + 波动率条带（±1σ, ±2σ）。

        Args:
            daily_returns: 每日收益率 Series 或 DataFrame
            title: 图表标题

        Returns:
            Plotly Figure 对象

        Note: When using in Streamlit, decorate the caller with @st.cache_data to avoid recomputation.
        """
        if daily_returns is None or (hasattr(daily_returns, "empty") and daily_returns.empty):
            return PlotManager._empty_fig(title)
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
                                  line=dict(color=_COLORS["blue"], width=1)))
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title="日期", yaxis_title="对数收益率")
        return fig

    @staticmethod
    def plot_drawdown(
        portfolio_df: pd.DataFrame, title: str = "回撤曲线",
        line_color: Optional[str] = None, fill_color: Optional[str] = None,
        equity_col: str = "equity", date_col: str = "date",
    ) -> go.Figure:
        """回撤曲线，标注最大回撤点。

        Args:
            portfolio_df: 组合明细 DataFrame
            title: 图表标题
            line_color: 回撤线颜色
            fill_color: 回撤填充颜色
            equity_col: 净值列名
            date_col: 日期列名

        Returns:
            Plotly Figure 对象

        Note: When using in Streamlit, decorate the caller with @st.cache_data to avoid recomputation.
        """
        empty = PlotManager._check_df(portfolio_df, title)
        if empty:
            return empty
        df = PlotManager._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = PlotManager._get_equity_col(df, equity_col=equity_col)
        _line_color = line_color or _COLORS["red"]
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
                    marker=dict(color=_COLORS["red"], size=10, symbol="x"),
                ))
                fig.add_annotation(
                    x=df.loc[min_dd_idx, date_col], y=drawdown.loc[min_dd_idx],
                    text=f"最大回撤: {drawdown.loc[min_dd_idx]:.2f}%",
                    showarrow=True, arrowhead=2, font=dict(color=_COLORS["red"]),
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
        """月度收益率热力图（行=年份，列=月份）。

        Args:
            portfolio_df: 组合明细 DataFrame
            title: 图表标题
            equity_col: 净值列名
            date_col: 日期列名

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(portfolio_df, title)
        if empty:
            return empty
        df = PlotManager._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = PlotManager._get_equity_col(df, equity_col=equity_col)
        if not eq_col or date_col not in df.columns:
            return PlotManager._empty_fig(title, missing_cols=[equity_col, date_col])
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
        """滚动夏普比率曲线（年化），附均值线。

        Args:
            daily_returns: 每日收益率 Series 或 DataFrame
            window: 滚动窗口大小
            risk_free_rate: 年化无风险利率，默认 0.0
            title: 图表标题
            line_color: 曲线颜色

        Returns:
            Plotly Figure 对象

        Note: When using in Streamlit, decorate the caller with @st.cache_data to avoid recomputation.
        """
        if daily_returns is None or (hasattr(daily_returns, "empty") and daily_returns.empty):
            return PlotManager._empty_fig(title)
        if isinstance(daily_returns, pd.DataFrame):
            daily_returns = daily_returns.iloc[:, 0]
        daily_rf = risk_free_rate / 252
        excess = daily_returns - daily_rf
        rolling_sharpe = (
            excess.rolling(window).mean() / excess.rolling(window).std()
        ) * np.sqrt(252)
        _line_color = line_color or _COLORS["blue"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe, mode="lines",
            name=f"夏普({window}日)", line=dict(color=_line_color, width=1.5),
        ))
        mean_sharpe = rolling_sharpe.mean()
        fig.add_hline(y=mean_sharpe, line_dash="dash", line_color=_COLORS["orange"],
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
        """滚动最大回撤曲线。

        使用向量化计算代替 apply，性能提升 10x+。

        Args:
            portfolio_df: 组合明细 DataFrame
            window: 滚动窗口大小
            title: 图表标题
            line_color: 曲线颜色
            fill_color: 填充颜色
            equity_col: 净值列名
            date_col: 日期列名

        Returns:
            Plotly Figure 对象

        Note: When using in Streamlit, decorate the caller with @st.cache_data to avoid recomputation.
        """
        empty = PlotManager._check_df(portfolio_df, title)
        if empty:
            return empty
        df = PlotManager._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = PlotManager._get_equity_col(df, equity_col=equity_col)
        if not eq_col or date_col not in df.columns:
            return PlotManager._empty_fig(title, missing_cols=[equity_col, date_col])
        equity = df.set_index(date_col)[eq_col]
        peak = equity.cummax()
        dd = (equity - peak) / peak
        rolling_max_dd = dd.rolling(window).min() * 100
        _line_color = line_color or _COLORS["red"]
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
        """单笔盈亏直方图 + 核密度曲线。

        Args:
            trades_df: 交易记录 DataFrame
            title: 图表标题
            pnl_col: 盈亏列名，默认 pnl_pct

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(trades_df, title)
        if empty:
            return empty
        if pnl_col not in trades_df.columns:
            pnl_col = "pnl"
        if pnl_col not in trades_df.columns:
            return PlotManager._empty_fig(title, missing_cols=["pnl_pct", "pnl"])
        pnl = trades_df[pnl_col].dropna()
        if pnl.empty:
            return PlotManager._empty_fig(title)
        win_rate = (pnl > 0).mean() * 100
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=pnl, nbinsx=50, name="盈亏分布",
            marker_color=_COLORS["blue"], opacity=0.7, histnorm="probability density",
        ))
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(pnl.values)
            x_range = np.linspace(pnl.min(), pnl.max(), 200)
            fig.add_trace(go.Scatter(
                x=x_range, y=kde(x_range), mode="lines", name="核密度",
                line=dict(color=_COLORS["red"], width=2),
            ))
        except ImportError:
            pass
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.add_annotation(
            x=0.98, y=0.98, xref="paper", yref="paper", text=f"胜率: {win_rate:.1f}%",
            showarrow=False,
            font=dict(size=14, color=_COLORS["green"] if win_rate > 50 else _COLORS["red"]),
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
        """正态分位数图（Q-Q plot）。

        Args:
            daily_returns: 每日收益率 Series 或 DataFrame
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        if daily_returns is None or (hasattr(daily_returns, "empty") and daily_returns.empty):
            return PlotManager._empty_fig(title)
        if isinstance(daily_returns, pd.DataFrame):
            daily_returns = daily_returns.iloc[:, 0]
        daily_returns = daily_returns.dropna()
        try:
            from scipy.stats import probplot
            (theoretical_q, ordered_values), (slope, intercept, r) = probplot(
                daily_returns, dist="norm")
        except ImportError:
            return PlotManager._empty_fig(title + "（需要 scipy）", missing_cols=["scipy"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=theoretical_q, y=ordered_values, mode="markers", name="样本分位数",
            marker=dict(color=_COLORS["blue"], size=4),
        ))
        fit_line = slope * theoretical_q + intercept
        fig.add_trace(go.Scatter(
            x=theoretical_q, y=fit_line, mode="lines", name="正态拟合线",
            line=dict(color=_COLORS["red"], width=2),
        ))
        fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                           text=f"R² = {r**2:.4f}", showarrow=False, font=dict(size=12))
        fig.update_layout(title=title, template="plotly_white", height=500,
                          xaxis_title="理论分位数", yaxis_title="样本分位数")
        return fig

    # ================================================================
    # 模块3：风险与归因分析
    # ================================================================

    @staticmethod
    def plot_risk_pie(risk_contrib_dict: Dict[str, float], title: str = "风险贡献") -> go.Figure:
        """饼图显示各品种/策略的风险贡献。

        Args:
            risk_contrib_dict: 风险贡献字典 {品种/策略: 贡献值}
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        if not risk_contrib_dict:
            return PlotManager._empty_fig(title)
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
        """单合约市值占权益比例的时间序列。

        Args:
            portfolio_df: 组合明细 DataFrame
            positions_df: 持仓 DataFrame，包含 date, symbol, market_value
            title: 图表标题
            equity_col: 净值列名
            date_col: 日期列名

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(positions_df, title, missing_cols=["symbol", "market_value"])
        if empty:
            return empty
        empty = PlotManager._check_df(portfolio_df, title, missing_cols=[equity_col, date_col])
        if empty:
            return empty
        positions_df = PlotManager._ensure_date_column(positions_df, date_col=date_col)
        portfolio_df = PlotManager._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = PlotManager._get_equity_col(portfolio_df, equity_col=equity_col)
        fig = go.Figure()
        if eq_col and date_col in portfolio_df.columns:
            equity_map = portfolio_df.set_index(date_col)[eq_col]
            for idx, sym in enumerate(positions_df["symbol"].unique()):
                sub = positions_df[positions_df["symbol"] == sym].copy().sort_values(date_col)
                sub["equity"] = sub[date_col].map(equity_map)
                sub["concentration"] = sub["market_value"] / sub["equity"] * 100
                fig.add_trace(go.Scatter(
                    x=sub[date_col], y=sub["concentration"], mode="lines", name=sym,
                    line=dict(color=PlotManager._get_color(idx), width=1.5),
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
        """杠杆率曲线，标注安全阈值。

        Args:
            portfolio_df: 组合明细 DataFrame
            positions_df: 持仓 DataFrame
            threshold: 安全阈值
            title: 图表标题
            equity_col: 净值列名
            date_col: 日期列名

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(portfolio_df, title)
        if empty:
            return empty
        df = PlotManager._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = PlotManager._get_equity_col(df, equity_col=equity_col)
        if not eq_col or date_col not in df.columns:
            return PlotManager._empty_fig(title, missing_cols=[equity_col, date_col])
        fig = go.Figure()
        if positions_df is not None and not positions_df.empty:
            positions_df = PlotManager._ensure_date_column(positions_df, date_col=date_col)
            total_exposure = positions_df.groupby(date_col)["market_value"].sum().abs()
            equity_series = df.set_index(date_col)[eq_col]
            leverage = total_exposure / equity_series
            fig.add_trace(go.Scatter(
                x=leverage.index, y=leverage, mode="lines", name="杠杆率",
                line=dict(color=_COLORS["blue"], width=1.5),
            ))
        elif "margin" in df.columns:
            leverage = df["margin"] / df[eq_col]
            fig.add_trace(go.Scatter(
                x=df[date_col], y=leverage, mode="lines", name="杠杆率",
                line=dict(color=_COLORS["blue"], width=1.5),
            ))
        fig.add_hline(y=threshold, line_dash="dash", line_color=_COLORS["red"],
                      annotation_text=f"安全阈值: {threshold}")
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title="日期", yaxis_title="杠杆率")
        return fig

    @staticmethod
    def plot_stress_test(
        portfolio_df: pd.DataFrame, stress_events: List[Dict], title: str = "压力测试",
        equity_col: str = "equity", date_col: str = "date",
    ) -> go.Figure:
        """瀑布图：历史极端行情区间内策略累计收益。

        Args:
            portfolio_df: 组合明细 DataFrame
            stress_events: 压力事件列表 [{"name", "start", "end"}, ...]
            title: 图表标题
            equity_col: 净值列名
            date_col: 日期列名

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(portfolio_df, title)
        if empty:
            return empty
        if not stress_events:
            return PlotManager._empty_fig(title + "（无压力事件）")
        df = PlotManager._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = PlotManager._get_equity_col(df, equity_col=equity_col)
        if not eq_col or date_col not in df.columns:
            return PlotManager._empty_fig(title, missing_cols=[equity_col, date_col])
        names, returns = [], []
        for event in stress_events:
            start, end = pd.to_datetime(event["start"]), pd.to_datetime(event["end"])
            mask = (df[date_col] >= start) & (df[date_col] <= end)
            sub = df.loc[mask]
            ret = (sub[eq_col].iloc[-1] / sub[eq_col].iloc[0] - 1) * 100 if not sub.empty else 0
            names.append(event["name"])
            returns.append(ret)
        colors = [_COLORS["green"] if r >= 0 else _COLORS["red"] for r in returns]
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
        """多品种日收益率相关性热力图。

        Args:
            returns_df: 日收益率 DataFrame（列为品种）
            method: 相关系数方法（pearson, spearman, kendall）
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(returns_df, title)
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
        """上/下捕获比率柱状图。

        Args:
            returns_df: 策略日收益率
            benchmark_returns: 基准日收益率
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        if returns_df is None or benchmark_returns is None:
            return PlotManager._empty_fig(title)
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
            marker_color=[_COLORS["green"], _COLORS["red"]],
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
        """滚动 VaR（历史模拟法）曲线。

        Args:
            returns_df: 日收益率 Series 或 DataFrame
            window: 滚动窗口大小
            ci: 置信水平
            title: 图表标题

        Returns:
            Plotly Figure 对象

        Warning:
            大数据集下滚动 apply 可能较慢，建议 window <= 504 且数据量 < 5000 条。

        Note: When using in Streamlit, decorate the caller with @st.cache_data to avoid recomputation.
        """
        if returns_df is None or (hasattr(returns_df, "empty") and returns_df.empty):
            return PlotManager._empty_fig(title)
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
            name=f"VaR({ci:.0%})", line=dict(color=_COLORS["red"], width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=returns_df.index, y=returns_df, mode="lines", name="日收益率",
            line=dict(color=_COLORS["blue"], width=0.5), opacity=0.3,
        ))
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title="日期", yaxis_title="收益率 / VaR")
        return fig

    # ================================================================
    # 模块4：交易执行分析
    # ================================================================

    @staticmethod
    def plot_vwap_scatter(
        orders_df: pd.DataFrame, title: str = "成交价 vs VWAP",
        date_col: str = "date",
    ) -> go.Figure:
        """成交价 vs VWAP 散点图，参考线 y=x。

        Args:
            orders_df: 订单记录 DataFrame，需包含 vwap 和 filled_price/fill_price/price
            title: 图表标题
            date_col: 日期列名

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(orders_df, title)
        if empty:
            return empty
        orders_df = PlotManager._ensure_date_column(orders_df, date_col=date_col)
        fill_col = "filled_price" if "filled_price" in orders_df.columns else "fill_price"
        if fill_col not in orders_df.columns:
            fill_col = "price"
        if "vwap" not in orders_df.columns:
            return PlotManager._empty_fig(title, missing_cols=["vwap"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=orders_df["vwap"], y=orders_df[fill_col], mode="markers", name="成交",
            marker=dict(color=_COLORS["blue"], size=6, opacity=0.6),
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
        """持仓时长直方图。

        期望 trades_df 包含 bars 或 entry_date/exit_date 列。

        Args:
            trades_df: 交易记录 DataFrame
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(trades_df, title)
        if empty:
            return empty
        if "bars" in trades_df.columns:
            holding = trades_df["bars"].dropna()
            x_label = "持仓时长 (bars)"
        elif "entry_date" in trades_df.columns and "exit_date" in trades_df.columns:
            holding = (pd.to_datetime(trades_df["exit_date"]) - pd.to_datetime(trades_df["entry_date"])).dt.days.dropna()
            x_label = "持仓时长 (天)"
        else:
            return PlotManager._empty_fig(title, missing_cols=["bars", "entry_date", "exit_date"])
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=holding, nbinsx=50, name="持仓时长",
            marker_color=_COLORS["blue"], opacity=0.7,
        ))
        mean_holding = holding.mean()
        fig.add_vline(x=mean_holding, line_dash="dash", line_color=_COLORS["red"],
                      annotation_text=f"均值: {mean_holding:.1f}")
        fig.update_layout(title=title, template="plotly_white", height=400,
                          xaxis_title=x_label, yaxis_title="频次")
        return fig

    @staticmethod
    def plot_daily_trades_count(
        trades_df: pd.DataFrame, title: str = "每日交易次数",
        date_col: str = "date",
    ) -> go.Figure:
        """每日交易次数条形图。

        期望 trades_df 包含 date 和可选的 type 列。

        Args:
            trades_df: 交易记录 DataFrame
            title: 图表标题
            date_col: 日期列名

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(trades_df, title)
        if empty:
            return empty
        trades_df = PlotManager._ensure_date_column(trades_df, date_col=date_col)
        if "type" in trades_df.columns:
            daily = trades_df.groupby([trades_df[date_col].dt.date, "type"]).size().unstack(fill_value=0)
            daily.index = pd.to_datetime(daily.index)
            fig = go.Figure()
            for col in daily.columns:
                color = _COLORS["green"] if "buy" in str(col).lower() else _COLORS["red"]
                fig.add_trace(go.Bar(x=daily.index, y=daily[col], name=str(col), marker_color=color))
        else:
            daily = trades_df.groupby(trades_df[date_col].dt.date).size()
            daily.index = pd.to_datetime(daily.index)
            fig = go.Figure(go.Bar(x=daily.index, y=daily.values, name="交易次数",
                                    marker_color=_COLORS["blue"]))
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title="日期", yaxis_title="交易次数", barmode="stack")
        return fig

    @staticmethod
    def plot_pnl_by_symbol(
        trades_df: pd.DataFrame, title: str = "各品种盈亏分布",
        pnl_col: str = "pnl_pct",
    ) -> go.Figure:
        """按品种分组的盈亏箱线图。

        期望 trades_df 包含 pnl_pct/pnl 和 symbol 列。

        Args:
            trades_df: 交易记录 DataFrame
            title: 图表标题
            pnl_col: 盈亏列名

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(trades_df, title)
        if empty:
            return empty
        if pnl_col not in trades_df.columns:
            pnl_col = "pnl"
        if pnl_col not in trades_df.columns or "symbol" not in trades_df.columns:
            return PlotManager._empty_fig(title, missing_cols=["pnl_pct", "pnl", "symbol"])
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
        """滑点时间序列曲线。

        Args:
            orders_df: 订单记录 DataFrame，需包含 filled_price/fill_price 和 price
            cumulative: 是否显示累积滑点
            title: 图表标题
            date_col: 日期列名

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(orders_df, title)
        if empty:
            return empty
        orders_df = PlotManager._ensure_date_column(orders_df, date_col=date_col)
        fill_col = "filled_price" if "filled_price" in orders_df.columns else "fill_price"
        missing = []
        if fill_col not in orders_df.columns:
            missing.append("filled_price")
        if "price" not in orders_df.columns:
            missing.append("price")
        if missing:
            return PlotManager._empty_fig(title, missing_cols=missing)
        slippage = orders_df[fill_col] - orders_df["price"]
        if cumulative:
            slippage = slippage.cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=orders_df[date_col], y=slippage, mode="lines+markers",
            name="累积滑点" if cumulative else "滑点",
            line=dict(color=_COLORS["purple"], width=1.5), marker=dict(size=4),
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
        """展期成本累积曲线。

        Args:
            rollover_costs_df: 展期成本 DataFrame，需包含 date 和 cost 列
            title: 图表标题
            date_col: 日期列名

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(rollover_costs_df, title)
        if empty:
            return empty
        rollover_costs_df = PlotManager._ensure_date_column(rollover_costs_df, date_col=date_col)
        if "cost" not in rollover_costs_df.columns:
            return PlotManager._empty_fig(title, missing_cols=["cost"])
        df = rollover_costs_df.sort_values(date_col)
        df["cum_cost"] = df["cost"].cumsum()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.4], vertical_spacing=0.08)
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df["cum_cost"], mode="lines", name="累积成本",
            line=dict(color=_COLORS["red"], width=2),
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=df[date_col], y=df["cost"], name="单次成本",
            marker_color=_COLORS["orange"],
        ), row=2, col=1)
        fig.update_layout(title=title, template="plotly_white", height=500, hovermode="x unified")
        fig.update_yaxes(title_text="累积成本", row=1, col=1)
        fig.update_yaxes(title_text="单次成本", row=2, col=1)
        return fig

    # ================================================================
    # 模块5：参数优化与敏感性分析
    # ================================================================

    @staticmethod
    def plot_param_heatmap(
        results_df: pd.DataFrame, param_x: str, param_y: str,
        metric: str = "sharpe", title: str = "参数热力图",
    ) -> go.Figure:
        """二维参数网格热力图。

        Args:
            results_df: 优化结果 DataFrame
            param_x: X轴参数名
            param_y: Y轴参数名
            metric: 颜色映射的指标
            title: 图表标题

        Returns:
            Plotly Figure 对象

        Note: When using in Streamlit, decorate the caller with @st.cache_data to avoid recomputation.
        """
        empty = PlotManager._check_df(results_df, title)
        if empty:
            return empty
        missing = [col for col in [param_x, param_y, metric] if col not in results_df.columns]
        if missing:
            return PlotManager._empty_fig(title, missing_cols=missing)
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
        """平行坐标图，用于 ≥3 个参数的组合可视化。

        Args:
            results_df: 优化结果 DataFrame
            param_cols: 参数列名列表
            metric_col: 指标列名
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(results_df, title)
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
        """一维参数扫描线图。

        Args:
            results_df: 优化结果 DataFrame
            param_name: 参数名
            metric: 主指标名
            extra_metrics: 额外指标列表
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(results_df, title)
        if empty:
            return empty
        if param_name not in results_df.columns:
            return PlotManager._empty_fig(title, missing_cols=[param_name])
        fig = go.Figure()
        grouped = results_df.groupby(param_name)[metric].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=grouped[param_name], y=grouped[metric], mode="lines+markers",
            name=metric, line=dict(color=_COLORS["blue"], width=2),
        ))
        if extra_metrics:
            for idx, em in enumerate(extra_metrics):
                if em in results_df.columns:
                    g = results_df.groupby(param_name)[em].mean().reset_index()
                    fig.add_trace(go.Scatter(
                        x=g[param_name], y=g[em], mode="lines+markers", name=em,
                        line=dict(color=PlotManager._get_color(idx + 1), width=1.5, dash="dash"),
                    ))
        best_idx = grouped[metric].idxmax()
        best_val = grouped.loc[best_idx, param_name]
        fig.add_vline(x=best_val, line_dash="dash", line_color=_COLORS["green"],
                      annotation_text=f"最优: {best_val}")
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title=param_name, yaxis_title="指标值")
        return fig

    @staticmethod
    def plot_param_stability(
        rolling_results_df: pd.DataFrame, param_name: str, title: str = "参数稳定性",
    ) -> go.Figure:
        """滚动优化结果中参数随窗口的变化折线图。

        Args:
            rolling_results_df: 滚动优化结果 DataFrame
            param_name: 参数名
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(rolling_results_df, title)
        if empty:
            return empty
        if param_name not in rolling_results_df.columns:
            return PlotManager._empty_fig(title, missing_cols=[param_name])
        date_col = "window_date" if "window_date" in rolling_results_df.columns else rolling_results_df.columns[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_results_df[date_col], y=rolling_results_df[param_name],
            mode="lines+markers", name=param_name,
            line=dict(color=_COLORS["blue"], width=2),
        ))
        mean_val = rolling_results_df[param_name].mean()
        fig.add_hline(y=mean_val, line_dash="dash", line_color=_COLORS["orange"],
                      annotation_text=f"均值: {mean_val:.4f}")
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title="优化窗口", yaxis_title=param_name)
        return fig

    @staticmethod
    def plot_surface_3d(
        results_df: pd.DataFrame, x_param: str, y_param: str, z_metric: str,
        title: str = "3D 参数曲面",
    ) -> go.Figure:
        """三维表面图。

        Args:
            results_df: 优化结果 DataFrame
            x_param: X轴参数名
            y_param: Y轴参数名
            z_metric: Z轴指标名
            title: 图表标题

        Returns:
            Plotly Figure 对象

        Note: When using in Streamlit, decorate the caller with @st.cache_data to avoid recomputation.
        """
        empty = PlotManager._check_df(results_df, title)
        if empty:
            return empty
        missing = [col for col in [x_param, y_param, z_metric] if col not in results_df.columns]
        if missing:
            return PlotManager._empty_fig(title, missing_cols=missing)
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
        """各参数与目标指标的相关性绝对值条形图。

        Args:
            results_df: 优化结果 DataFrame
            param_cols: 参数列名列表
            metric_col: 指标列名
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(results_df, title)
        if empty:
            return empty
        if metric_col not in results_df.columns:
            return PlotManager._empty_fig(title, missing_cols=[metric_col])
        importances = {}
        for col in param_cols:
            if col in results_df.columns:
                importances[col] = abs(results_df[col].corr(results_df[metric_col]))
        if not importances:
            return PlotManager._empty_fig(title, missing_cols=param_cols)
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_imp]
        values = [x[1] for x in sorted_imp]
        fig = go.Figure(go.Bar(
            x=names, y=values, marker_color=[_COLORS["blue"]] * len(names),
            text=[f"{v:.3f}" for v in values], textposition="outside",
        ))
        fig.update_layout(title=title, template="plotly_white", height=400,
                          xaxis_title="参数", yaxis_title=f"|相关系数| vs {metric_col}")
        return fig

    # ================================================================
    # 模块6：市场状态与环境适配
    # ================================================================

    @staticmethod
    def plot_regime_overlay(
        df: pd.DataFrame, regime_series: Optional[pd.Series] = None,
        price_col: str = "close", title: str = "市场状态与价格",
        date_col: str = "date",
    ) -> go.Figure:
        """价格走势上叠加趋势/震荡市场状态背景色。

        Args:
            df: 包含 date 和价格列的 DataFrame
            regime_series: 市场状态序列（trend/range）
            price_col: 价格列名
            title: 图表标题
            date_col: 日期列名

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(df, title)
        if empty:
            return empty
        df = PlotManager._ensure_date_column(df, date_col=date_col)
        if price_col not in df.columns:
            return PlotManager._empty_fig(title, missing_cols=[price_col])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df[price_col], mode="lines", name=price_col,
            line=dict(color="#333333", width=1.5),
        ))
        regime = regime_series if regime_series is not None else (
            df["market_regime"] if "market_regime" in df.columns else None)
        if regime is not None:
            if isinstance(regime, pd.DataFrame):
                regime = regime.iloc[:, 0]
            if isinstance(regime, pd.Series):
                regime = regime.reindex(df.index) if len(regime) == len(df) else regime
            else:
                regime = df["market_regime"]
            regime = regime.fillna("unknown")
            for regime_type in ["trend", "range"]:
                mask = (regime == regime_type)
                if np.any(mask.values):
                    color = _REGIME_COLORS.get(regime_type, "rgba(128,128,128,0.1)")
                    for start_idx, end_idx in PlotManager._find_contiguous_blocks(mask):
                        fig.add_vrect(
                            x0=df.iloc[start_idx][date_col],
                            x1=df.iloc[end_idx][date_col],
                            fillcolor=color, layer="below", line_width=0,
                            annotation_text=regime_type if start_idx == 0 or
                            regime.iloc[start_idx] != regime.iloc[start_idx - 1] else "",
                            annotation_position="top left", annotation_font_size=8,
                        )
        fig.update_layout(title=title, template="plotly_white", height=500,
                          hovermode="x unified", xaxis_title="日期", yaxis_title=price_col)
        return fig

    @staticmethod
    def plot_price_with_signals(
        df: pd.DataFrame, title: str = "价格与信号",
        date_col: str = "date", price_col: str = "close",
        buy_col: str = "buy", sell_col: str = "sell",
    ) -> go.Figure:
        """价格曲线叠加买卖信号标记。

        Args:
            df: 包含 date, close, 可选买卖信号列的 DataFrame
            title: 图表标题
            date_col: 日期列名
            price_col: 价格列名
            buy_col: 买入信号列名
            sell_col: 卖出信号列名

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(df, title)
        if empty:
            return empty
        df = PlotManager._ensure_date_column(df, date_col=date_col)
        if price_col not in df.columns:
            return PlotManager._empty_fig(title, missing_cols=[price_col])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df[price_col], mode="lines", name=price_col,
            line=dict(color=_COLORS["blue"], width=1.5),
        ))
        if buy_col in df.columns:
            buy_mask = df[buy_col].astype(bool) if df[buy_col].dtype != bool else df[buy_col]
            buy_df = df[buy_mask]
            if not buy_df.empty:
                fig.add_trace(go.Scatter(
                    x=buy_df[date_col], y=buy_df[price_col], mode="markers",
                    name="买入", marker=dict(color=_COLORS["green"], size=10, symbol="triangle-up"),
                ))
        if sell_col in df.columns:
            sell_mask = df[sell_col].astype(bool) if df[sell_col].dtype != bool else df[sell_col]
            sell_df = df[sell_mask]
            if not sell_df.empty:
                fig.add_trace(go.Scatter(
                    x=sell_df[date_col], y=sell_df[price_col], mode="markers",
                    name="卖出", marker=dict(color=_COLORS["red"], size=10, symbol="triangle-down"),
                ))
        fig.update_layout(title=title, template="plotly_white", height=500,
                          hovermode="x unified", xaxis_title="日期", yaxis_title=price_col)
        return fig

    @staticmethod
    def plot_monthly_returns(
        portfolio_df: pd.DataFrame, title: str = "月度收益率",
        equity_col: str = "equity", date_col: str = "date",
    ) -> go.Figure:
        """月度收益率柱状图。

        Args:
            portfolio_df: 组合明细 DataFrame
            title: 图表标题
            equity_col: 净值列名
            date_col: 日期列名

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(portfolio_df, title)
        if empty:
            return empty
        df = PlotManager._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = PlotManager._get_equity_col(df, equity_col=equity_col)
        if not eq_col or date_col not in df.columns:
            return PlotManager._empty_fig(title, missing_cols=[equity_col, date_col])
        df = df.sort_values(date_col).copy()
        df["month"] = pd.to_datetime(df[date_col]).dt.to_period("M")
        monthly_eq = df.groupby("month")[eq_col].last()
        monthly_ret = monthly_eq.pct_change().dropna() * 100
        monthly_ret.index = monthly_ret.index.astype(str)
        colors = np.where(monthly_ret >= 0, _COLORS["green"], _COLORS["red"])
        fig = go.Figure(go.Bar(
            x=monthly_ret.index, y=monthly_ret.values,
            marker_color=colors,
            text=[f"{v:.2f}%" for v in monthly_ret.values], textposition="outside",
        ))
        fig.add_hline(y=0, line_color="gray", line_width=0.5)
        fig.update_layout(title=title, template="plotly_white", height=400,
                          hovermode="x unified", xaxis_title="月份", yaxis_title="收益率 (%)")
        return fig

    @staticmethod
    def animate_rolling_correlation(
        returns_df: pd.DataFrame, window: int = 60, step: int = 5,
        max_frames: int = 30, method: str = "pearson",
        title: str = "滚动相关性动画",
    ) -> go.Figure:
        """滚动相关性热力图动画。

        Args:
            returns_df: 日收益率 DataFrame（列为品种）
            window: 滚动窗口大小
            step: 步长
            max_frames: 最大帧数，超过则采样
            method: 相关系数方法
            title: 图表标题

        Returns:
            Plotly Figure 对象

        Note: When using in Streamlit, decorate the caller with @st.cache_data to avoid recomputation.
        """
        empty = PlotManager._check_df(returns_df, title)
        if empty:
            return empty
        step = max(step, len(returns_df) // max_frames)
        cols = returns_df.columns.tolist()
        n = len(cols)
        frames = []
        for i in range(window, len(returns_df), step):
            sub = returns_df.iloc[i - window:i]
            corr = sub.corr(method=method)
            frames.append((returns_df.index[i], corr))
        if not frames:
            return PlotManager._empty_fig(title + "（数据不足）")
        fig = go.Figure()
        z0 = frames[0][1].values
        fig.add_trace(go.Heatmap(
            z=z0, x=cols, y=cols, colorscale="RdBu_r", zmid=0,
            text=np.round(z0, 2), texttemplate="%{text}",
            colorbar=dict(title="相关系数"),
        ))
        anim_frames = []
        for date_label, corr in frames[1:]:
            anim_frames.append(dict(
                name=str(date_label),
                data=[go.Heatmap(
                    z=corr.values, x=cols, y=cols,
                    text=np.round(corr.values, 2), texttemplate="%{text}",
                )],
            ))
        fig.update_layout(title=title, template="plotly_white", height=600,
                          updatemenus=[dict(
                              type="buttons", showactive=False,
                              buttons=[dict(label="播放", method="animate",
                                            args=[None, dict(frame=dict(duration=500, redraw=True),
                                                             fromcurrent=True)]),
                                       dict(label="暂停", method="animate",
                                            args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                               mode="immediate")])],
                          )],
                          sliders=[dict(active=0, steps=[
                              dict(label=str(frames[i][0]), method="animate",
                                   args=[[str(frames[i][0])],
                                         dict(frame=dict(duration=500, redraw=True),
                                              mode="immediate")])
                              for i in range(len(frames))
                          ])])
        fig.frames = anim_frames
        return fig

    @staticmethod
    def plot_regime_transition_matrix(
        regime_series: pd.Series, title: str = "市场状态转移矩阵",
    ) -> go.Figure:
        """市场状态转移概率矩阵热力图。

        Args:
            regime_series: 市场状态序列
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        if regime_series is None or (hasattr(regime_series, "empty") and regime_series.empty):
            return PlotManager._empty_fig(title)
        if isinstance(regime_series, pd.DataFrame):
            regime_series = regime_series.iloc[:, 0]
        states = regime_series.unique().tolist()
        n = len(states)
        trans = np.zeros((n, n))
        for i in range(len(regime_series) - 1):
            s_from = regime_series.iloc[i]
            s_to = regime_series.iloc[i + 1]
            if s_from in states and s_to in states:
                trans[states.index(s_from)][states.index(s_to)] += 1
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans = trans / row_sums
        fig = go.Figure(go.Heatmap(
            z=trans, x=states, y=states,
            colorscale="Blues", text=np.round(trans, 3), texttemplate="%{text}",
            colorbar=dict(title="转移概率"),
        ))
        fig.update_layout(title=title, template="plotly_white", height=500,
                          xaxis_title="下一状态", yaxis_title="当前状态")
        return fig

    @staticmethod
    def plot_regime_performance(
        portfolio_df: pd.DataFrame, regime_series: pd.Series,
        title: str = "各市场状态绩效", equity_col: str = "equity", date_col: str = "date",
    ) -> go.Figure:
        """各市场状态下策略收益柱状图。

        Args:
            portfolio_df: 组合明细 DataFrame
            regime_series: 市场状态序列
            title: 图表标题
            equity_col: 净值列名
            date_col: 日期列名

        Returns:
            Plotly Figure 对象
        """
        empty = PlotManager._check_df(portfolio_df, title)
        if empty:
            return empty
        df = PlotManager._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = PlotManager._get_equity_col(df, equity_col=equity_col)
        if not eq_col or date_col not in df.columns:
            return PlotManager._empty_fig(title, missing_cols=[equity_col, date_col])
        if regime_series is None or (hasattr(regime_series, "empty") and regime_series.empty):
            return PlotManager._empty_fig(title, missing_cols=["regime_series"])
        if isinstance(regime_series, pd.DataFrame):
            regime_series = regime_series.iloc[:, 0]
        if len(regime_series) != len(df):
            if isinstance(regime_series.index, pd.DatetimeIndex) and date_col in df.columns:
                regime_map = pd.Series(regime_series.values, index=regime_series.index)
                regime_map = regime_map[~regime_map.index.duplicated(keep="last")].sort_index()
                df["regime"] = df[date_col].map(
                    lambda d: regime_map.iloc[regime_map.index.get_indexer([pd.Timestamp(d)], method="ffill")[0]]
                    if regime_map.index.get_indexer([pd.Timestamp(d)], method="ffill")[0] >= 0 else "unknown"
                )
            else:
                regime_dedup = regime_series[~regime_series.index.duplicated(keep="last")].sort_index()
                regime_aligned = regime_dedup.reindex(df.index, method="ffill").fillna("unknown")
                df["regime"] = regime_aligned.values
        else:
            df["regime"] = regime_series.values
        df["daily_return"] = df[eq_col].pct_change()
        regime_returns = df.groupby("regime")["daily_return"].mean() * 252 * 100
        colors = [_REGIME_COLORS.get(r, "rgba(128,128,128,0.5)") for r in regime_returns.index]
        fig = go.Figure(go.Bar(
            x=regime_returns.index, y=regime_returns.values,
            marker_color=colors,
            text=[f"{v:.2f}%" for v in regime_returns.values], textposition="outside",
        ))
        fig.add_hline(y=0, line_color="gray", line_width=0.5)
        fig.update_layout(title=title, template="plotly_white", height=400,
                          xaxis_title="市场状态", yaxis_title="年化收益率 (%)")
        return fig

    @staticmethod
    def _find_contiguous_blocks(mask: pd.Series) -> List[Tuple[int, int]]:
        """找到布尔序列中连续为 True 的区间起止索引。"""
        blocks = []
        in_block = False
        start = 0
        for i, val in enumerate(mask):
            if val and not in_block:
                start = i
                in_block = True
            elif not val and in_block:
                blocks.append((start, i - 1))
                in_block = False
        if in_block:
            blocks.append((start, len(mask) - 1))
        return blocks