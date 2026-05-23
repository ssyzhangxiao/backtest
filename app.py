"""商品期货多策略量化回测系统 - 应用入口。"""
import streamlit as st

from utils.session_state import init_session_state
from components.sidebar import render_strategy_config
from components.results import render_backtest_results
from pages.data_import import render_data_import
from pages.data_analysis import render_data_analysis
from pages.backtest import run_backtest
from pages.optimization import render_optimization


def main():
    init_session_state()

    st.title("📊 商品期货多策略量化回测系统")
    st.caption("基于 PyBroker 回测引擎 | 展期法 | 多策略组合 | 参数优化")

    config = render_strategy_config()

    st.sidebar.divider()
    st.sidebar.subheader("💰 回测设置")
    st.session_state.initial_cash = st.sidebar.number_input(
        "初始资金", min_value=100000, max_value=100000000, value=1000000, step=100000
    )

    page = st.sidebar.radio(
        "导航",
        ["📁 数据导入", "📉 数据分析", "🚀 运行回测", "🔍 参数优化"],
        label_visibility="collapsed",
    )

    if page == "📁 数据导入":
        render_data_import()
    elif page == "📉 数据分析":
        render_data_analysis()
    elif page == "🚀 运行回测":
        st.header("🚀 运行回测")

        if not config["selected_strategies"]:
            st.warning("请在左侧选择至少一个策略！")
        elif not st.session_state.data_loaded:
            st.warning("请先在\"数据导入\"页面加载数据！")
        else:
            st.subheader("当前配置")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write("**策略**: ", ", ".join(config["selected_strategies"]))
            with col2:
                st.write("**展期模式**: ", config["rollover_mode"])
            with col3:
                st.write("**初始资金**: ", f"{st.session_state.initial_cash:,.0f}")
            with col4:
                st.write(f"**手续费率**: {config.get('commission', 0.0001):.4f} | **滑点**: {config.get('slippage', 0.0002):.4f}")

            if st.button("▶️ 开始回测", type="primary", use_container_width=True):
                with st.spinner("回测运行中..."):
                    result = run_backtest(config)
                    if result is not None:
                        st.session_state.backtest_result = result
                        st.session_state.backtest_run = True
                        st.success("回测完成！")

            if st.session_state.backtest_run and st.session_state.backtest_result:
                render_backtest_results(st.session_state.backtest_result, config)

    elif page == "🔍 参数优化":
        if config.get("enable_optimization"):
            render_optimization(config)
        else:
            st.info("请在左侧勾选\"启用参数优化\"")


if __name__ == "__main__":
    main()