"""
图表绘制模块。

使用 Plotly 绘制回测结果可视化图表，供 Streamlit 前端调用。
包含：资金曲线、展期统计、持仓明细、参数热力图等。
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class PlotManager:
    """
    图表绘制管理器。

    提供各种可视化方法，返回 Plotly Figure 对象，
    可直接在 Streamlit 中通过 st.plotly_chart() 显示。
    """

    @staticmethod
    def plot_equity_curve(portfolio_df: pd.DataFrame,
                          title: str = "资金曲线") -> go.Figure:
        """
        绘制资金曲线图。

        Args:
            portfolio_df: PyBroker 回测返回的 portfolio DataFrame，
                         需包含 date, market_value 列
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        fig = go.Figure()

        if 'date' in portfolio_df.columns and 'market_value' in portfolio_df.columns:
            fig.add_trace(go.Scatter(
                x=portfolio_df['date'],
                y=portfolio_df['market_value'],
                mode='lines',
                name='账户净值',
                line=dict(color='#2196F3', width=2)
            ))

        initial = portfolio_df['market_value'].iloc[0] if len(portfolio_df) > 0 else 0
        fig.add_hline(y=initial, line_dash="dash", line_color="gray",
                      annotation_text="初始资金")

        fig.update_layout(
            title=title,
            xaxis_title="日期",
            yaxis_title="账户净值",
            template="plotly_white",
            hovermode='x unified',
            height=500
        )
        return fig

    @staticmethod
    def plot_drawdown(portfolio_df: pd.DataFrame,
                      title: str = "回撤曲线") -> go.Figure:
        """
        绘制回撤曲线。

        Args:
            portfolio_df: portfolio DataFrame
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        fig = go.Figure()

        if 'date' in portfolio_df.columns and 'market_value' in portfolio_df.columns:
            equity = portfolio_df['market_value']
            peak = equity.cummax()
            drawdown = (equity - peak) / peak * 100

            fig.add_trace(go.Scatter(
                x=portfolio_df['date'],
                y=drawdown,
                mode='lines',
                name='回撤%',
                fill='tozeroy',
                line=dict(color='#F44336', width=1.5),
                fillcolor='rgba(244, 67, 54, 0.2)'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="日期",
            yaxis_title="回撤 (%)",
            template="plotly_white",
            hovermode='x unified',
            height=400
        )
        return fig

    @staticmethod
    def plot_price_with_signals(df: pd.DataFrame,
                                trades_df: Optional[pd.DataFrame] = None,
                                title: str = "价格与交易信号") -> go.Figure:
        """
        绘制价格走势与交易信号。

        Args:
            df: 包含 date, open, high, low, close 的 DataFrame
            trades_df: 交易记录 DataFrame，需包含 date, type, fill_price
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25],
                            vertical_spacing=0.05)

        fig.add_trace(go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线'
        ), row=1, col=1)

        if trades_df is not None and not trades_df.empty:
            buys = trades_df[trades_df['type'] == 'buy']
            sells = trades_df[trades_df['type'] == 'sell']

            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys['date'],
                    y=buys['fill_price'],
                    mode='markers',
                    name='买入',
                    marker=dict(symbol='triangle-up', size=12, color='#4CAF50')
                ), row=1, col=1)

            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells['date'],
                    y=sells['fill_price'],
                    mode='markers',
                    name='卖出',
                    marker=dict(symbol='triangle-down', size=12, color='#F44336')
                ), row=1, col=1)

        if 'volume' in df.columns:
            fig.add_trace(go.Bar(
                x=df['date'],
                y=df['volume'],
                name='成交量',
                marker_color='#90CAF9'
            ), row=2, col=1)

        fig.update_layout(
            title=title,
            template="plotly_white",
            height=600,
            xaxis_rangeslider_visible=False
        )
        return fig

    @staticmethod
    def plot_rollover_timeline(rollover_dates: pd.DataFrame,
                               title: str = "展期时间线") -> go.Figure:
        """
        绘制展期时间线。

        Args:
            rollover_dates: 展期日期 DataFrame，需包含 date, prev_dominant_symbol, dominant_symbol
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        fig = go.Figure()

        if rollover_dates.empty:
            fig.update_layout(title=title + "（无展期事件）", height=300)
            return fig

        for i, row in rollover_dates.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['date'], row['date']],
                y=[0, 1],
                mode='lines+text',
                text=[row.get('prev_dominant_symbol', ''), row.get('dominant_symbol', '')],
                textposition=['bottom center', 'top center'],
                name=f"展期 {i+1}",
                line=dict(width=2, dash='dot'),
            ))

        fig.update_layout(
            title=title,
            template="plotly_white",
            height=300,
            showlegend=False,
            xaxis_title="日期",
            yaxis_visible=False
        )
        return fig

    @staticmethod
    def plot_param_heatmap(results_df: pd.DataFrame,
                           param_x: str, param_y: str,
                           metric: str = 'sharpe',
                           title: str = "参数热力图") -> go.Figure:
        """
        绘制参数优化热力图。

        Args:
            results_df: 优化结果 DataFrame
            param_x: X轴参数名
            param_y: Y轴参数名
            metric: 颜色映射的指标
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        if results_df.empty or param_x not in results_df.columns or param_y not in results_df.columns:
            fig = go.Figure()
            fig.update_layout(title=title + "（无数据）", height=400)
            return fig

        pivot = results_df.pivot_table(
            values=metric, index=param_y, columns=param_x, aggfunc='mean'
        )

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            text=np.round(pivot.values, 4),
            texttemplate='%{text}',
            colorbar=dict(title=metric)
        ))

        fig.update_layout(
            title=f"{title} ({metric})",
            xaxis_title=param_x,
            yaxis_title=param_y,
            height=500
        )
        return fig

    @staticmethod
    def plot_monthly_returns(portfolio_df: pd.DataFrame,
                             title: str = "月度收益") -> go.Figure:
        """
        绘制月度收益柱状图。

        Args:
            portfolio_df: portfolio DataFrame
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        fig = go.Figure()

        if 'date' in portfolio_df.columns and 'market_value' in portfolio_df.columns:
            df = portfolio_df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.to_period('M')

            monthly = df.groupby('month')['market_value'].agg(['first', 'last'])
            monthly['return'] = (monthly['last'] / monthly['first'] - 1) * 100
            monthly = monthly.reset_index()
            monthly['month_str'] = monthly['month'].astype(str)

            colors = ['#4CAF50' if r >= 0 else '#F44336' for r in monthly['return']]

            fig.add_trace(go.Bar(
                x=monthly['month_str'],
                y=monthly['return'],
                marker_color=colors,
                name='月度收益%'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="月份",
            yaxis_title="收益率 (%)",
            template="plotly_white",
            height=400
        )
        return fig

    @staticmethod
    def plot_regime_overlay(df: pd.DataFrame,
                            title: str = "市场状态与价格") -> go.Figure:
        """
        绘制市场状态叠加图。

        在价格走势上叠加趋势/震荡市场状态背景色。

        Args:
            df: 包含 date, close, market_regime 的 DataFrame
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        fig = go.Figure()

        if 'market_regime' in df.columns:
            trend_mask = df['market_regime'] == 'trend'

            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['close'],
                mode='lines',
                name='收盘价',
                line=dict(color='#333333', width=1.5)
            ))

            for i in range(len(df) - 1):
                if trend_mask.iloc[i]:
                    fig.add_vrect(
                        x0=df['date'].iloc[i],
                        x1=df['date'].iloc[i + 1] if i + 1 < len(df) else df['date'].iloc[i],
                        fillcolor='rgba(76, 175, 80, 0.1)',
                        line_width=0
                    )

        fig.update_layout(
            title=title,
            xaxis_title="日期",
            yaxis_title="价格",
            template="plotly_white",
            height=500
        )
        return fig
