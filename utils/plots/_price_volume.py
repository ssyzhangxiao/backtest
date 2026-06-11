"""图表绘制模块 — 价格/量/OI/展期/缺失数据。"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional

from utils.plots._base import COLORS, BasePlotMixin

class PriceVolumeMixin(BasePlotMixin):
    """价格/量/OI/展期/缺失数据相关图表。"""

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
        empty = self._check_df(df, _title, missing_cols=["open", "high", "low", "close"])
        if empty:
            return empty
        df = self._ensure_date_column(df, date_col=date_col)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.05)
        fig.add_trace(go.Candlestick(
            x=df[date_col], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="K线",
            increasing_line_color=COLORS["red"],
            decreasing_line_color=COLORS["green"],
        ), row=1, col=1)
        if "volume" in df.columns:
            colors = np.where(df["close"] >= df["open"], COLORS["red"], COLORS["green"])
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
        empty = self._check_df(rollover_df, title)
        if empty:
            return empty
        rollover_df = self._ensure_date_column(rollover_df, date_col=date_col)
        fig = go.Figure()
        if price_df is not None and not price_df.empty:
            price_df = self._ensure_date_column(price_df, date_col=date_col)
            fig.add_trace(go.Scatter(
                x=price_df[date_col], y=price_df["close"],
                mode="lines", name="收盘价",
                line=dict(color=COLORS["gray"], width=1),
            ))
        for _, row in rollover_df.iterrows():
            fig.add_vline(
                x=row[date_col], line_width=1.5, line_dash="dash",
                line_color=COLORS["red"],
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
        empty = self._check_df(df, title)
        if empty:
            return empty
        df = self._ensure_date_column(df, date_col=date_col)
        if symbol_list:
            df = df[df["symbol"].isin(symbol_list)]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.5], vertical_spacing=0.08)
        for idx, sym in enumerate(df["symbol"].unique()):
            sub = df[df["symbol"] == sym]
            c = self._get_color(idx)
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
        empty = self._check_df(df, title)
        if empty:
            return empty
        df = self._ensure_date_column(df, date_col=date_col)
        if contract_list is None:
            last_date = df[date_col].max()
            contract_list = sorted(df[df[date_col] == last_date]["symbol"].unique().tolist())
        if len(contract_list) < 2:
            return self._empty_fig(title + "（合约数不足）")
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
        empty = self._check_df(df, title)
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
