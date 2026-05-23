"""回测运行页面模块。"""
import logging

import streamlit as st
import pandas as pd
from pybroker import Strategy, StrategyConfig
from pybroker.common import FeeMode

from core.rollover import RolloverManager, RolloverMode
from core.risk_manager import RiskManager
from core.portfolio import PortfolioManager
from core.strategies import create_strategy

from utils.session_state import register_pybroker_columns

logger = logging.getLogger("backtest_app")


def run_backtest(config: dict):
    if not st.session_state.data_loaded or st.session_state.pybroker_df is None:
        st.error("请先加载数据！")
        return None

    register_pybroker_columns()

    df = st.session_state.pybroker_df.copy()

    if df.empty:
        st.error("数据为空！请检查数据加载和筛选设置。")
        return None

    symbols = df["symbol"].unique().tolist()
    date_min = str(df["date"].min().date()) if "date" in df.columns else "N/A"
    date_max = str(df["date"].max().date()) if "date" in df.columns else "N/A"
    filter_status = "✅ 已筛选" if st.session_state.get("data_filtered") else "⚠️ 全量数据"
    logger.info(
        "回测数据确认 | %s | 品种数=%d | 日期=%s~%s | 行数=%d",
        filter_status, len(symbols), date_min, date_max, len(df),
    )
    st.info(f"当前回测范围：{len(symbols)} 个品种 | {date_min} ~ {date_max} | {filter_status} | {len(df):,} 行")

    initial_cash = st.session_state.get("initial_cash", 1_000_000)
    commission = config.get("commission", 0.0001)
    slippage = config.get("slippage", 0.0002)
    fee_amount = commission + slippage
    pybroker_config = StrategyConfig(
        initial_cash=initial_cash,
        fee_mode=FeeMode.ORDER_PERCENT,
        fee_amount=fee_amount,
    )

    start_date = str(df["date"].min().date())
    end_date = str(df["date"].max().date())
    strategy = Strategy(df, start_date, end_date, pybroker_config)

    risk_manager = RiskManager(**config["risk_params"])

    try:
        rollover_mode = RolloverMode(config["rollover_mode"])
    except ValueError:
        st.error(f"无效展期模式: {config['rollover_mode']}")
        return None

    rollover_manager = RolloverManager(mode=rollover_mode, **config["rollover_params"])

    if config.get("dynamic_weighting") and len(config["selected_strategies"]) > 1:
        portfolio_mgr = PortfolioManager(total_allocation=config.get("total_allocation", 0.8))
        for strat_name in config["selected_strategies"]:
            params = config["strategy_params"].get(strat_name, {})
            strat_instance = create_strategy(strat_name, **params)
            portfolio_mgr.add_strategy(strat_name, strat_instance)
        portfolio_mgr.register_all_to_pybroker(
            pybroker_strategy=strategy,
            symbols=symbols,
            rollover_wrapper=rollover_manager.create_rollover_exec_fn,
        )
    else:
        for strat_name in config["selected_strategies"]:
            params = config["strategy_params"].get(strat_name, {})
            strat_instance = create_strategy(strat_name, **params)

            indicators = []
            if hasattr(strat_instance, "register_indicators"):
                indicators = strat_instance.register_indicators()

            exec_fn = strat_instance.execute
            exec_fn = rollover_manager.create_rollover_exec_fn(exec_fn)
            exec_fn = risk_manager.wrap_with_risk_control(exec_fn)

            strategy.add_execution(fn=exec_fn, symbols=symbols, indicators=indicators)

    try:
        result = strategy.backtest()
        return result
    except Exception as e:
        st.error(f"回测执行失败: {e}")
        import traceback

        st.code(traceback.format_exc())
        return None