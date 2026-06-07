"""
P0 整改（2026-06-07）：core/strategies/ 已整体删除，
create_strategy 接口已废弃。本页面改为调用 Pipeline 编排器
（规则17、规则18 公共入口），具体回测由 run_backtest.py + core.engine.backtest_runner
公共系统执行。
"""
import logging

import streamlit as st

logger = logging.getLogger("backtest_app")


def run_backtest(config: dict):
    if not st.session_state.data_loaded or st.session_state.pybroker_df is None:
        st.error("请先加载数据！")
        return None

    # P0 整改：所有回测执行均委托 Pipeline.run_backtest() 公共入口
    # 不再在 UI 层直接构造 PyBroker Strategy / create_strategy
    try:
        from runner.pipeline import Pipeline

        pipe = Pipeline(config.get("config_path", "config.yaml"))

        # 应用 UI 侧参数到 Pipeline（统一 with_config 热更新接口）
        pipe.with_config(
            initial_cash=config.get("initial_cash", 1_000_000),
            commission=config.get("commission", 0.0001),
            slippage=config.get("slippage", 0.0002),
            rollover_mode=config.get("rollover_mode", "auto"),
            selected_strategies=config.get("selected_strategies", []),
            strategy_params=config.get("strategy_params", {}),
            risk_params=config.get("risk_params", {}),
            dynamic_weighting=config.get("dynamic_weighting", False),
            total_allocation=config.get("total_allocation", 0.8),
        )

        # 应用已优化参数（如有）
        applied_params = st.session_state.get("applied_optimized_params")
        if applied_params:
            pipe.with_config(optimized_params=applied_params)

        with st.spinner("回测执行中（Pipeline.run_backtest）..."):
            result = pipe.run_backtest()

        st.success("回测完成！")
        return result

    except Exception as e:
        st.error(f"回测执行失败: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None