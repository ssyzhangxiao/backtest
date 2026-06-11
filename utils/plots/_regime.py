"""图表绘制模块 — 市场状态与环境适配。"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from utils.plots._base import COLORS, REGIME_COLORS, BasePlotMixin


class RegimeMixin(BasePlotMixin):
    """市场状态与环境适配相关图表。"""

    @staticmethod
    def plot_regime_overlay(
        df: pd.DataFrame, regime_series: Optional[pd.Series] = None,
        price_col: str = "close", title: str = "市场状态与价格",
        date_col: str = "date",
    ) -> go.Figure:
        """价格走势上叠加趋势/震荡市场状态背景色。"""
        empty = RegimeMixin._check_df(df, title)
        if empty:
            return empty
        df = RegimeMixin._ensure_date_column(df, date_col=date_col)
        if price_col not in df.columns:
            return RegimeMixin._empty_fig(title, missing_cols=[price_col])
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
                    color = REGIME_COLORS.get(regime_type, "rgba(128,128,128,0.1)")
                    for start_idx, end_idx in RegimeMixin._find_contiguous_blocks(mask):
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
        """价格曲线叠加买卖信号标记。"""
        empty = RegimeMixin._check_df(df, title)
        if empty:
            return empty
        df = RegimeMixin._ensure_date_column(df, date_col=date_col)
        if price_col not in df.columns:
            return RegimeMixin._empty_fig(title, missing_cols=[price_col])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df[price_col], mode="lines", name=price_col,
            line=dict(color=COLORS["blue"], width=1.5),
        ))
        if buy_col in df.columns:
            buy_mask = df[buy_col].astype(bool) if df[buy_col].dtype != bool else df[buy_col]
            buy_df = df[buy_mask]
            if not buy_df.empty:
                fig.add_trace(go.Scatter(
                    x=buy_df[date_col], y=buy_df[price_col], mode="markers",
                    name="买入", marker=dict(color=COLORS["green"], size=10, symbol="triangle-up"),
                ))
        if sell_col in df.columns:
            sell_mask = df[sell_col].astype(bool) if df[sell_col].dtype != bool else df[sell_col]
            sell_df = df[sell_mask]
            if not sell_df.empty:
                fig.add_trace(go.Scatter(
                    x=sell_df[date_col], y=sell_df[price_col], mode="markers",
                    name="卖出", marker=dict(color=COLORS["red"], size=10, symbol="triangle-down"),
                ))
        fig.update_layout(title=title, template="plotly_white", height=500,
                          hovermode="x unified", xaxis_title="日期", yaxis_title=price_col)
        return fig

    @staticmethod
    def plot_monthly_returns(
        portfolio_df: pd.DataFrame, title: str = "月度收益率",
        equity_col: str = "equity", date_col: str = "date",
    ) -> go.Figure:
        """月度收益率柱状图。"""
        empty = RegimeMixin._check_df(portfolio_df, title)
        if empty:
            return empty
        df = RegimeMixin._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = RegimeMixin._get_equity_col(df, equity_col=equity_col)
        if not eq_col or date_col not in df.columns:
            return RegimeMixin._empty_fig(title, missing_cols=[equity_col, date_col])
        df = df.sort_values(date_col).copy()
        df["month"] = pd.to_datetime(df[date_col]).dt.to_period("M")
        monthly_eq = df.groupby("month")[eq_col].last()
        monthly_ret = monthly_eq.pct_change().dropna() * 100
        monthly_ret.index = monthly_ret.index.astype(str)
        colors = np.where(monthly_ret >= 0, COLORS["green"], COLORS["red"])
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
        """滚动相关性热力图动画。"""
        empty = RegimeMixin._check_df(returns_df, title)
        if empty:
            return empty
        step = max(step, len(returns_df) // max_frames)
        cols = returns_df.columns.tolist()
        frames = []
        for i in range(window, len(returns_df), step):
            sub = returns_df.iloc[i - window:i]
            corr = sub.corr(method=method)
            frames.append((returns_df.index[i], corr))
        if not frames:
            return RegimeMixin._empty_fig(title + "（数据不足）")
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
        """市场状态转移概率矩阵热力图。"""
        if regime_series is None or (hasattr(regime_series, "empty") and regime_series.empty):
            return RegimeMixin._empty_fig(title)
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
        """各市场状态下策略收益柱状图。"""
        empty = RegimeMixin._check_df(portfolio_df, title)
        if empty:
            return empty
        df = RegimeMixin._ensure_date_column(portfolio_df, date_col=date_col)
        eq_col = RegimeMixin._get_equity_col(df, equity_col=equity_col)
        if not eq_col or date_col not in df.columns:
            return RegimeMixin._empty_fig(title, missing_cols=[equity_col, date_col])
        if regime_series is None or (hasattr(regime_series, "empty") and regime_series.empty):
            return RegimeMixin._empty_fig(title, missing_cols=["regime_series"])
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
        colors = [REGIME_COLORS.get(r, "rgba(128,128,128,0.5)") for r in regime_returns.index]
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
