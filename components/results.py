"""回测结果渲染模块。"""

from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.metrics import MetricsCalculator
from utils.plots import PlotManager
from utils import safe_col
from core.config import get_default_stress_events


def _build_positions_df(
    trades_df: pd.DataFrame, portfolio_df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    if trades_df is None or trades_df.empty:
        return None
    required = {"symbol", "date"}
    if not required.issubset(trades_df.columns):
        return None
    pnl_col = "pnl" if "pnl" in trades_df.columns else None
    shares_col = (
        "shares"
        if "shares" in trades_df.columns
        else "amount"
        if "amount" in trades_df.columns
        else None
    )
    price_col = (
        "close"
        if "close" in trades_df.columns
        else "fill_price"
        if "fill_price" in trades_df.columns
        else None
    )
    rows = []
    for _, row in trades_df.iterrows():
        r = {"date": row["date"], "symbol": row["symbol"]}
        if pnl_col:
            r["market_value"] = abs(row.get(pnl_col, 0))
        elif shares_col and price_col:
            r["market_value"] = abs(row.get(shares_col, 0) * row.get(price_col, 0))
        else:
            r["market_value"] = 0
        rows.append(r)
    if not rows:
        return None
    return pd.DataFrame(rows)


def render_backtest_results(result, config: Optional[dict] = None):
    if config is None:
        config = {}
    st.header("📊 回测结果")

    metrics_calc = MetricsCalculator()
    pybroker_metrics = metrics_calc.extract_from_pybroker_result(result)

    portfolio_df = result.portfolio
    trades_df = result.trades
    orders_df = result.orders

    additional_metrics = metrics_calc.calculate_additional_metrics(
        portfolio_df=portfolio_df, trades_df=trades_df
    )

    all_metrics = {**pybroker_metrics, **additional_metrics}

    st.subheader("绩效概览")
    cards = metrics_calc.format_metrics_card(all_metrics)

    cols = st.columns(5)
    for i, card in enumerate(cards):
        with cols[i % 5]:
            value = metrics_calc.format_value(card["value"], card["format"])
            st.metric(card["label"], value)

    st.divider()

    data_mode = "contract"
    if st.session_state.data_loader:
        data_mode = st.session_state.data_loader.data_mode or "contract"

    tab_names = [
        "📈 资金曲线",
        "📉 回撤分析",
        "📊 策略绩效",
        "🛡️ 风险归因",
        "🔄 展期统计",
        "📋 交易记录",
        "💹 交易执行",
        "📊 月度收益",
        "📅 逐年收益",
        "🌐 市场状态",
    ]
    tabs = st.tabs(tab_names)
    (
        tab_equity,
        tab_drawdown,
        tab_performance,
        tab_risk,
        tab_rollover,
        tab_trades,
        tab_execution,
        tab_monthly,
        tab_yearly,
        tab_regime,
    ) = tabs

    _render_equity_tab(tab_equity, portfolio_df)
    _render_drawdown_tab(tab_drawdown, portfolio_df)
    _render_performance_tab(tab_performance, portfolio_df, trades_df)
    _render_risk_tab(tab_risk, portfolio_df, trades_df)
    _render_rollover_tab(tab_rollover, trades_df, metrics_calc, data_mode)
    _render_trades_tab(tab_trades, trades_df, orders_df, config)
    _render_execution_tab(tab_execution, orders_df, trades_df)
    _render_monthly_tab(tab_monthly, portfolio_df)
    _render_yearly_tab(tab_yearly, portfolio_df)
    _render_regime_tab(tab_regime, portfolio_df)

    st.divider()
    st.subheader("完整指标表")
    if all_metrics:
        metrics_display = pd.DataFrame(
            [{"指标": k, "值": v} for k, v in all_metrics.items()]
        )
        st.dataframe(metrics_display, use_container_width=True, hide_index=True)


def _render_equity_tab(tab, portfolio_df):
    with tab:
        if portfolio_df is not None and not portfolio_df.empty:
            fig = PlotManager.plot_equity_curve(portfolio_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("无资金曲线数据")


def _render_drawdown_tab(tab, portfolio_df):
    with tab:
        if portfolio_df is not None and not portfolio_df.empty:
            fig = PlotManager.plot_drawdown(portfolio_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("无回撤数据")


def _render_performance_tab(tab, portfolio_df, trades_df):
    with tab:
        if portfolio_df is not None and not portfolio_df.empty:
            with st.expander("对数收益率 + 波动率条带", expanded=True):
                fig_log = PlotManager.plot_log_returns(portfolio_df)
                st.plotly_chart(fig_log, use_container_width=True)

            with st.expander("滚动夏普比率"):
                fig_rsharpe = PlotManager.plot_rolling_sharpe(portfolio_df)
                st.plotly_chart(fig_rsharpe, use_container_width=True)

            with st.expander("滚动最大回撤"):
                fig_rmdd = PlotManager.plot_rolling_max_drawdown(portfolio_df)
                st.plotly_chart(fig_rmdd, use_container_width=True)

            with st.expander("Q-Q 正态分位数图"):
                fig_qq = PlotManager.plot_qq_plot(portfolio_df)
                st.plotly_chart(fig_qq, use_container_width=True)
        else:
            st.info("无绩效数据")

        if trades_df is not None and not trades_df.empty:
            with st.expander("单笔盈亏分布"):
                fig_pnl = PlotManager.plot_pnl_distribution(trades_df)
                st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            st.info("无交易记录")


def _render_risk_tab(tab, portfolio_df, trades_df):
    with tab:
        if portfolio_df is not None and not portfolio_df.empty:
            with st.expander("滚动 VaR（历史模拟法）"):
                var_window = st.slider("VaR 滚动窗口", 60, 504, 252, key="var_window")
                var_ci = st.slider("置信水平", 0.90, 0.99, 0.95, 0.01, key="var_ci")
                eq_col = (
                    "equity"
                    if "equity" in portfolio_df.columns
                    else portfolio_df.columns[0]
                )
                if "date" in portfolio_df.columns:
                    daily_ret = (
                        portfolio_df.set_index("date")[eq_col].pct_change().dropna()
                    )
                else:
                    daily_ret = portfolio_df[eq_col].pct_change().dropna()
                fig_var = PlotManager.plot_rolling_var(
                    daily_ret, window=var_window, ci=var_ci
                )
                st.plotly_chart(fig_var, use_container_width=True)

            with st.expander("多品种相关性热力图"):
                if (
                    trades_df is not None
                    and not trades_df.empty
                    and "symbol" in trades_df.columns
                ):
                    symbols_in_trades = trades_df["symbol"].unique().tolist()
                    env_df = st.session_state.get("env_df")
                    if env_df is not None and not env_df.empty:
                        pivot_ret = env_df[
                            env_df["symbol"].isin(symbols_in_trades)
                        ].pivot_table(values="close", index="date", columns="symbol")
                        if pivot_ret.shape[1] >= 2:
                            ret_df = pivot_ret.pct_change().dropna()
                            fig_corr = PlotManager.plot_correlation_heatmap(ret_df)
                            st.plotly_chart(fig_corr, use_container_width=True)
                        else:
                            st.info("品种数不足，无法计算相关性")
                    else:
                        st.info("无环境数据")
                else:
                    st.info("无交易品种信息")

            with st.expander("上下捕获比率"):
                if trades_df is not None and not trades_df.empty:
                    eq_col = (
                        "equity"
                        if "equity" in portfolio_df.columns
                        else portfolio_df.columns[0]
                    )
                    if "date" in portfolio_df.columns:
                        strat_ret = (
                            portfolio_df.set_index("date")[eq_col].pct_change().dropna()
                        )
                    else:
                        strat_ret = portfolio_df[eq_col].pct_change().dropna()
                    env_df = st.session_state.get("env_df")
                    if env_df is not None and "close" in env_df.columns:
                        dom = env_df[
                            safe_col(env_df, "is_dominant")
                            if "is_dominant" in env_df.columns
                            else pd.Series([True] * len(env_df))
                        ].copy()
                        if not dom.empty:
                            bench_eq = dom.groupby("date")["close"].mean()
                            bench_ret = bench_eq.pct_change().dropna()
                            common_idx = strat_ret.index.intersection(bench_ret.index)
                            if len(common_idx) > 10:
                                fig_capture = PlotManager.plot_up_down_capture(
                                    strat_ret.loc[common_idx], bench_ret.loc[common_idx]
                                )
                                st.plotly_chart(fig_capture, use_container_width=True)
                            else:
                                st.info("策略与基准收益率重叠不足")
                        else:
                            st.info("无主力合约数据作为基准")
                    else:
                        st.info("无环境数据作为基准")
                else:
                    st.info("无交易记录")

            with st.expander("风险贡献饼图"):
                if (
                    trades_df is not None
                    and not trades_df.empty
                    and "symbol" in trades_df.columns
                ):
                    pnl_col = "pnl_pct" if "pnl_pct" in trades_df.columns else "pnl"
                    if pnl_col in trades_df.columns:
                        risk_contrib = (
                            trades_df.groupby("symbol")[pnl_col].std().to_dict()
                        )
                        fig_risk = PlotManager.plot_risk_pie(risk_contrib)
                        st.plotly_chart(fig_risk, use_container_width=True)
                    else:
                        st.info("交易记录缺少盈亏列")
                else:
                    st.info("无交易品种信息")

            with st.expander("持仓集中度曲线"):
                if (
                    trades_df is not None
                    and not trades_df.empty
                    and "symbol" in trades_df.columns
                ):
                    positions_df = _build_positions_df(trades_df, portfolio_df)
                    if positions_df is not None and not positions_df.empty:
                        fig_conc = PlotManager.plot_concentration_curve(
                            portfolio_df, positions_df
                        )
                        st.plotly_chart(fig_conc, use_container_width=True)
                    else:
                        st.info("无法构建持仓数据")
                else:
                    st.info("无持仓数据")

            with st.expander("杠杆率曲线"):
                if (
                    trades_df is not None
                    and not trades_df.empty
                    and "symbol" in trades_df.columns
                ):
                    positions_df = _build_positions_df(trades_df, portfolio_df)
                    fig_lev = PlotManager.plot_leverage_ratio(
                        portfolio_df, positions_df
                    )
                    st.plotly_chart(fig_lev, use_container_width=True)
                elif "margin" in portfolio_df.columns:
                    fig_lev = PlotManager.plot_leverage_ratio(portfolio_df)
                    st.plotly_chart(fig_lev, use_container_width=True)
                else:
                    st.info("无持仓或保证金数据")

            with st.expander("压力测试瀑布图"):
                stress_events = get_default_stress_events()
                fig_stress = PlotManager.plot_stress_test(portfolio_df, stress_events)
                st.plotly_chart(fig_stress, use_container_width=True)
                st.caption("压力事件为预设示例，可根据实际需要修改")
        else:
            st.info("无回测数据")


def _render_rollover_tab(tab, trades_df, metrics_calc, data_mode):
    if data_mode != "contract":
        with tab:
            st.info("当前数据模式非合约展期模式")
        return

    with tab:
        if st.session_state.data_loader:
            loader = st.session_state.data_loader
            rollover_dates = loader.get_rollover_dates()

            if not rollover_dates.empty:
                with st.expander("展期时间线", expanded=True):
                    price_df = st.session_state.env_df
                    if price_df is not None and "is_dominant" in price_df.columns:
                        dom_price = price_df[safe_col(price_df, "is_dominant")][
                            ["date", "close"]
                        ]
                    else:
                        dom_price = None
                    fig_roll = PlotManager.plot_rollover_timeline(
                        rollover_dates,
                        price_df=dom_price
                        if dom_price is not None and not dom_price.empty
                        else None,
                    )
                    st.plotly_chart(fig_roll, use_container_width=True)

                rollover_stats = metrics_calc.compute_rollover_stats(
                    trades_df, rollover_dates
                )
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("展期次数", rollover_stats["rollover_count"])
                with col2:
                    st.metric(
                        "总展期成本", f"{rollover_stats['total_rollover_cost']:.2f}"
                    )
                with col3:
                    st.metric(
                        "平均展期成本", f"{rollover_stats['avg_rollover_cost']:.2f}"
                    )

                with st.expander("展期成本累积曲线"):
                    if "cost" in rollover_dates.columns:
                        fig_rcost = PlotManager.plot_rollover_cost_curve(rollover_dates)
                        st.plotly_chart(fig_rcost, use_container_width=True)
                    else:
                        st.info("展期记录中无成本数据")

                st.dataframe(rollover_dates, use_container_width=True)
            else:
                st.info("未检测到展期事件")


def _render_trades_tab(tab, trades_df, orders_df, config):
    with tab:
        if trades_df is not None and not trades_df.empty:
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
            if config.get("save_trades", True):
                csv_data = trades_df.to_csv(index=False)
                st.download_button(
                    "📥 下载交易记录 CSV",
                    csv_data,
                    "trades.csv",
                    "text/csv",
                )
        elif orders_df is not None and not orders_df.empty:
            st.subheader("订单记录")
            st.dataframe(orders_df, use_container_width=True, hide_index=True)
            if config.get("save_trades", True):
                csv_data = orders_df.to_csv(index=False)
                st.download_button(
                    "📥 下载订单记录 CSV",
                    csv_data,
                    "orders.csv",
                    "text/csv",
                )
        else:
            st.info("无交易记录")


def _render_execution_tab(tab, orders_df, trades_df):
    with tab:
        if orders_df is not None and not orders_df.empty:
            with st.expander("成交价 vs VWAP 散点图"):
                fig_vwap = PlotManager.plot_vwap_scatter(orders_df)
                st.plotly_chart(fig_vwap, use_container_width=True)

            with st.expander("滑点时间序列"):
                cumulative_slip = st.checkbox(
                    "显示累积滑点", value=False, key="cum_slip"
                )
                fig_slip = PlotManager.plot_slippage_time(
                    orders_df, cumulative=cumulative_slip
                )
                st.plotly_chart(fig_slip, use_container_width=True)
        else:
            st.info("无订单数据")

        if trades_df is not None and not trades_df.empty:
            with st.expander("持仓时长分布", expanded=True):
                fig_hold = PlotManager.plot_holding_histogram(trades_df)
                st.plotly_chart(fig_hold, use_container_width=True)

            with st.expander("每日交易次数"):
                fig_daily = PlotManager.plot_daily_trades_count(trades_df)
                st.plotly_chart(fig_daily, use_container_width=True)

            with st.expander("各品种盈亏箱线图"):
                fig_sym_pnl = PlotManager.plot_pnl_by_symbol(trades_df)
                st.plotly_chart(fig_sym_pnl, use_container_width=True)
        else:
            st.info("无交易记录")


def _render_monthly_tab(tab, portfolio_df):
    with tab:
        if portfolio_df is not None and not portfolio_df.empty:
            with st.expander("月度收益率柱状图", expanded=True):
                fig_monthly = PlotManager.plot_monthly_returns(portfolio_df)
                st.plotly_chart(fig_monthly, use_container_width=True)

            with st.expander("月度收益热力图"):
                fig_heatmap = PlotManager.plot_monthly_heatmap(portfolio_df)
                st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("无月度收益数据")


def _render_yearly_tab(tab, portfolio_df):
    with tab:
        if portfolio_df is not None and not portfolio_df.empty:
            st.markdown("""
            **逐年收益率分析**帮助您了解策略在不同年份的表现差异：
            - 🟢 **绿色柱**表示盈利年份（收益率 > 0）
            - 🔴 **红色柱**表示亏损年份（收益率 < 0）
            - 浅红色背景标记**连续亏损区间**（策略可能失效的时段）
            - 虚线表示**年均收益率**，用于判断各年是否达到平均水平
            """)

            eq_col = (
                "equity"
                if "equity" in portfolio_df.columns
                else portfolio_df.columns[0]
            )
            date_col = (
                "date" if "date" in portfolio_df.columns else portfolio_df.columns[0]
            )

            df_yr = portfolio_df.copy()
            if date_col in df_yr.columns:
                df_yr[date_col] = pd.to_datetime(df_yr[date_col])
                df_yr["year"] = df_yr[date_col].dt.year

                yearly_eq = df_yr.groupby("year")[eq_col].agg(["first", "last"])
                yearly_eq["return_pct"] = (
                    yearly_eq["last"] / yearly_eq["first"] - 1
                ) * 100
                yearly_eq = yearly_eq.reset_index()

                years = yearly_eq["year"].tolist()
                returns = yearly_eq["return_pct"].tolist()

                with st.expander("逐年收益率柱状图", expanded=True):
                    bar_colors = ["#2ca02c" if r > 0 else "#d62728" for r in returns]
                    mean_return = yearly_eq["return_pct"].mean()

                    fig_yr = go.Figure()
                    fig_yr.add_trace(
                        go.Bar(
                            x=years,
                            y=returns,
                            marker_color=bar_colors,
                            text=[f"{r:.2f}%" for r in returns],
                            textposition="outside",
                            textfont=dict(size=10),
                            hovertemplate="%{x}年: %{y:.2f}%<extra></extra>",
                        )
                    )

                    fig_yr.add_hline(y=0, line_color="gray", line_width=0.8)
                    fig_yr.add_hline(
                        y=mean_return,
                        line_dash="dash",
                        line_color="#1f77b4",
                        line_width=1.5,
                        annotation_text=f"均值 {mean_return:.2f}%",
                        annotation_position="top right",
                    )

                    fig_yr.add_trace(
                        go.Scatter(
                            x=years,
                            y=returns,
                            mode="lines",
                            line=dict(color="#1f77b4", width=2, dash="dot"),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

                    consecutive_loss = []
                    loss_count = 0
                    for r in returns:
                        if r < 0:
                            loss_count += 1
                            consecutive_loss.append(loss_count)
                        else:
                            loss_count = 0
                            consecutive_loss.append(0)

                    for i, count in enumerate(consecutive_loss):
                        if count >= 2:
                            fig_yr.add_vrect(
                                x0=years[i] - 0.4,
                                x1=years[i] + 0.4,
                                fillcolor="rgba(214, 39, 40, 0.1)",
                                line_width=0,
                            )

                    fig_yr.update_layout(
                        title="逐年收益率（绿色=盈利，红色=亏损，浅红底色=连续亏损区间）",
                        xaxis_title="年份",
                        yaxis_title="收益率(%)",
                        template="plotly_white",
                        height=500,
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_yr, use_container_width=True)

                with st.expander("逐年收益率数据表", expanded=True):
                    display_df = yearly_eq[["year", "return_pct"]].copy()
                    display_df.columns = ["年份", "收益率(%)"]
                    display_df["收益率(%)"] = display_df["收益率(%)"].round(2)

                    profit_years = (yearly_eq["return_pct"] > 0).sum()
                    total_years = len(yearly_eq)
                    loss_years = total_years - profit_years

                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("总年数", f"{total_years}")
                    with col2:
                        st.metric(
                            "盈利年数",
                            f"{profit_years}",
                            delta=f"{profit_years / total_years * 100:.0f}%",
                        )
                    with col3:
                        st.metric(
                            "亏损年数",
                            f"{loss_years}",
                            delta=f"-{loss_years / total_years * 100:.0f}%",
                        )
                    with col4:
                        best_idx = yearly_eq["return_pct"].idxmax()
                        st.metric(
                            "最佳年份",
                            f"{int(yearly_eq.loc[best_idx, 'year'])}年",
                            delta=f"{yearly_eq.loc[best_idx, 'return_pct']:.2f}%",
                        )
                    with col5:
                        worst_idx = yearly_eq["return_pct"].idxmin()
                        st.metric(
                            "最差年份",
                            f"{int(yearly_eq.loc[worst_idx, 'year'])}年",
                            delta=f"{yearly_eq.loc[worst_idx, 'return_pct']:.2f}%",
                        )

                    st.dataframe(
                        display_df.style.applymap(
                            lambda v: (
                                "background-color: #d4edda; color: #155724"
                                if isinstance(v, (int, float)) and v > 0
                                else "background-color: #f8d7da; color: #721c24"
                                if isinstance(v, (int, float)) and v < 0
                                else ""
                            ),
                            subset=["收益率(%)"],
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    csv_data = display_df.to_csv(index=False)
                    st.download_button(
                        "📥 下载逐年收益率 CSV",
                        csv_data,
                        "yearly_returns.csv",
                        "text/csv",
                    )

                with st.expander("策略失效区间分析"):
                    _render_failure_analysis(years, returns)
            else:
                st.info("数据中无日期列，无法计算逐年收益率")
        else:
            st.info("无回测数据")


def _render_failure_analysis(years, returns):
    max_consecutive = 0
    current = 0
    failure_periods = []
    start_year = None

    for i, r in enumerate(returns):
        if r < 0:
            current += 1
            if current == 1:
                start_year = years[i]
            if current > max_consecutive:
                max_consecutive = current
        else:
            if current >= 2:
                failure_periods.append(
                    {
                        "起始年份": start_year,
                        "结束年份": years[i - 1],
                        "持续年数": current,
                        "累计亏损": sum(returns[j] for j in range(i - current, i)),
                    }
                )
            current = 0
            start_year = None

    if current >= 2:
        failure_periods.append(
            {
                "起始年份": start_year,
                "结束年份": years[-1],
                "持续年数": current,
                "累计亏损": sum(
                    returns[j] for j in range(len(returns) - current, len(returns))
                ),
            }
        )

    st.markdown(f"**最长连续亏损年数**: {max_consecutive}年")

    if failure_periods:
        failure_df = pd.DataFrame(failure_periods)
        st.markdown("**连续亏损≥2年的区间**（策略可能失效的时段）：")
        st.dataframe(failure_df, use_container_width=True, hide_index=True)

        st.warning(
            "⚠️ 连续亏损区间提示：策略在这些时段可能不适应市场环境，"
            "建议结合市场状态分析，考虑在这些时段暂停策略或切换到其他策略。"
        )
    else:
        st.success("✅ 未出现连续2年以上的亏损，策略表现相对稳定。")


def _render_regime_tab(tab, portfolio_df):
    with tab:
        env_df = st.session_state.get("env_df")
        if env_df is not None and "env_market_regime" in env_df.columns:
            dominant_data = env_df[
                safe_col(env_df, "is_dominant")
                if "is_dominant" in env_df.columns
                else pd.Series([True] * len(env_df))
            ].copy()
            if not dominant_data.empty:
                with st.expander("市场状态背景叠加", expanded=True):
                    fig_regime = PlotManager.plot_regime_overlay(
                        dominant_data.rename(
                            columns={"env_market_regime": "market_regime"}
                        )
                    )
                    st.plotly_chart(fig_regime, use_container_width=True)

                with st.expander("市场状态转移矩阵"):
                    fig_trans = PlotManager.plot_regime_transition_matrix(
                        safe_col(dominant_data, "env_market_regime")
                    )
                    st.plotly_chart(fig_trans, use_container_width=True)

                with st.expander("各市场状态绩效"):
                    regime_s = safe_col(dominant_data, "env_market_regime")
                    fig_reg_perf = PlotManager.plot_regime_performance(
                        portfolio_df,
                        dominant_data.set_index("date")[regime_s.name]
                        if "date" in dominant_data.columns
                        else regime_s,
                    )
                    st.plotly_chart(fig_reg_perf, use_container_width=True)

                with st.expander("价格 + 买卖信号标记"):
                    if (
                        "buy" in dominant_data.columns
                        or "sell" in dominant_data.columns
                    ):
                        fig_sig = PlotManager.plot_price_with_signals(dominant_data)
                        st.plotly_chart(fig_sig, use_container_width=True)
                    else:
                        st.info("数据中无买卖信号列")

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
                            st.info("品种数不足，无法计算滚动相关性")
            else:
                st.info("无主力合约数据")
        else:
            st.info("无市场状态数据，请先加载数据并计算环境指标")
