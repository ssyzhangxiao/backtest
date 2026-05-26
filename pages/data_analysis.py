"""数据分析页面模块。"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.plots import PlotManager
from utils import safe_col


def render_data_analysis():
    st.header("📉 数据分析")

    if not st.session_state.data_loaded or st.session_state.env_df is None:
        st.warning("请先加载数据！")
        return

    df = st.session_state.env_df
    loader = st.session_state.data_loader

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 K线与信号", "📈 持仓量与成交量", "🔄 展期与价差",
        "🌡️ 数据缺失热力图", "🌐 市场状态",
    ])

    with tab1:
        symbols = df["symbol"].unique().tolist()
        selected_symbol = st.selectbox("选择合约", symbols, key="analysis_symbol")
        symbol_data = df[df["symbol"] == selected_symbol].copy()
        if not symbol_data.empty:
            fig = PlotManager.plot_price_with_volume(symbol_data, symbol=selected_symbol)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("买卖信号叠加"):
                fig_sig = PlotManager.plot_price_with_signals(symbol_data)
                st.plotly_chart(fig_sig, use_container_width=True)

    with tab2:
        with st.expander("多合约持仓量与成交量", expanded=True):
            fig_oi = PlotManager.plot_open_interest_volume(df)
            st.plotly_chart(fig_oi, use_container_width=True)

    with tab3:
        if loader:
            rollover_dates = loader.get_rollover_dates()
            if not rollover_dates.empty:
                with st.expander("展期时间线", expanded=True):
                    price_df = df[safe_col(df, "is_dominant")].copy() if "is_dominant" in df.columns else None
                    fig_roll = PlotManager.plot_rollover_timeline(
                        rollover_dates,
                        price_df=price_df[["date", "close"]] if price_df is not None and not price_df.empty else None,
                    )
                    st.plotly_chart(fig_roll, use_container_width=True)

            products = loader.get_product_symbols()
            for product, syms in products.items():
                if len(syms) >= 2:
                    with st.expander(f"{product} 价差热力图"):
                        product_data = df[df["product"] == product]
                        fig_spread = PlotManager.plot_spread_heatmap(product_data, product=product)
                        st.plotly_chart(fig_spread, use_container_width=True)

                    with st.expander(f"{product} 跨期价差走势"):
                        pivot_close = product_data.pivot_table(
                            values="close", index="date", columns="symbol"
                        )
                        if pivot_close.shape[1] >= 2:
                            cols_list = pivot_close.columns.tolist()
                            near, far = cols_list[0], cols_list[-1]
                            spread = pivot_close[near] - pivot_close[far]
                            fig_sp = go.Figure()
                            fig_sp.add_trace(go.Scatter(
                                x=spread.index, y=spread.values, mode="lines",
                                name=f"{near}-{far} 价差",
                            ))
                            fig_sp.update_layout(
                                title=f"{product} 跨期价差",
                                xaxis_title="日期", yaxis_title="价差",
                                template="plotly_white", height=400, hovermode="x unified",
                            )
                            st.plotly_chart(fig_sp, use_container_width=True)

    with tab4:
        with st.expander("数据缺失热力图", expanded=True):
            fig_missing = PlotManager.plot_missing_data_heatmap(df)
            st.plotly_chart(fig_missing, use_container_width=True)

    with tab5:
        if "env_market_regime" in df.columns:
            dominant_data = df[safe_col(df, "is_dominant")].copy() if "is_dominant" in df.columns else df.copy()
            if not dominant_data.empty:
                with st.expander("市场状态背景叠加", expanded=True):
                    fig_regime = PlotManager.plot_regime_overlay(
                        dominant_data.rename(columns={"env_market_regime": "market_regime"})
                    )
                    st.plotly_chart(fig_regime, use_container_width=True)

                regime_counts = safe_col(dominant_data, "env_market_regime").value_counts()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("趋势市天数", regime_counts.get("trend", 0))
                with col2:
                    st.metric("震荡市天数", regime_counts.get("range", 0))

                with st.expander("市场状态转移矩阵"):
                    fig_trans = PlotManager.plot_regime_transition_matrix(
                        safe_col(dominant_data, "env_market_regime")
                    )
                    st.plotly_chart(fig_trans, use_container_width=True)

                if "env_trend_score" in dominant_data.columns:
                    st.subheader("趋势强度分数")
                    ts = safe_col(dominant_data, "env_trend_score")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("平均趋势分数", f"{ts.mean():.3f}")
                    with col_b:
                        st.metric("最大趋势分数", f"{ts.max():.3f}")
                    with col_c:
                        st.metric("趋势分数 > 0.5 占比", f"{(ts > 0.5).mean():.1%}")

                if "env_bearish_exhaustion" in dominant_data.columns:
                    bearish_count = safe_col(dominant_data, "env_bearish_exhaustion").sum()
                    bullish_count = safe_col(dominant_data, "env_bullish_exhaustion").sum()
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("看跌衰竭信号", int(bearish_count))
                    with col_b:
                        st.metric("看涨衰竭信号", int(bullish_count))

                if "env_weight_trend" in dominant_data.columns:
                    st.subheader("动态策略权重")
                    wt = safe_col(dominant_data, "env_weight_trend").mean()
                    wr = safe_col(dominant_data, "env_weight_reversal").mean()
                    ws = safe_col(dominant_data, "env_weight_spread").mean()
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("趋势策略平均权重", f"{wt:.2%}")
                    with col_b:
                        st.metric("反转策略平均权重", f"{wr:.2%}")
                    with col_c:
                        st.metric("套利策略平均权重", f"{ws:.2%}")

                with st.expander("滚动相关性动画"):
                    if "close" in dominant_data.columns:
                        pivot_ret = dominant_data.pivot_table(
                            values="close", index="date", columns="symbol"
                        )
                        if pivot_ret.shape[1] >= 2:
                            ret_df = pivot_ret.pct_change().dropna()
                            fig_anim = PlotManager.animate_rolling_correlation(ret_df)
                            st.plotly_chart(fig_anim, use_container_width=True)
                        else:
                            st.info("合约数不足，无法计算滚动相关性")