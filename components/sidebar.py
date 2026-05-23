"""侧边栏策略配置模块。"""
import streamlit as st
from core.strategies import STRATEGY_REGISTRY


def render_strategy_config() -> dict:
    """渲染策略配置面板（侧边栏），返回配置字典。"""
    st.sidebar.header("⚙️ 策略配置")

    selected_strategies = st.sidebar.multiselect(
        "选择策略",
        list(STRATEGY_REGISTRY.keys()),
        default=["dual_ma"],
        format_func=lambda x: {
            "dual_ma": "📈 双均线趋势",
            "rsi": "📉 RSI反转",
            "spread": "🔄 跨期套利",
        }.get(x, x),
    )

    strategy_params = {}

    if "dual_ma" in selected_strategies:
        st.sidebar.subheader("双均线参数")
        _applied = st.session_state.get("applied_optimized_params", {})
        _is_applied = st.session_state.get("optimized_strategy") == "dual_ma"
        strategy_params["dual_ma"] = {
            "short_ma": st.sidebar.slider(
                "短期均线周期", 3, 15,
                int(_applied.get("short_ma", 5)) if _is_applied else 5,
                key="dual_short"
            ),
            "long_ma": st.sidebar.slider(
                "长期均线周期", 10, 60,
                int(_applied.get("long_ma", 20)) if _is_applied else 20,
                key="dual_long"
            ),
            "adx_threshold": st.sidebar.slider(
                "ADX趋势阈值", 15.0, 40.0, 25.0, 0.5, key="dual_adx"
            ),
            "position_size": st.sidebar.slider(
                "仓位比例", 0.05, 0.5, 0.3, 0.05, key="dual_pos"
            ),
        }

    if "rsi" in selected_strategies:
        st.sidebar.subheader("RSI参数")
        _applied = st.session_state.get("applied_optimized_params", {})
        _is_applied = st.session_state.get("optimized_strategy") == "rsi"
        strategy_params["rsi"] = {
            "rsi_period": st.sidebar.slider(
                "RSI周期", 5, 30,
                int(_applied.get("rsi_period", 14)) if _is_applied else 14,
                key="rsi_period"
            ),
            "oversold": st.sidebar.slider(
                "超卖阈值", 15.0, 35.0, 30.0, 1.0, key="rsi_oversold"
            ),
            "overbought": st.sidebar.slider(
                "超买阈值", 65.0, 85.0, 70.0, 1.0, key="rsi_overbought"
            ),
            "adx_threshold": st.sidebar.slider(
                "ADX震荡市阈值", 15.0, 40.0, 25.0, 0.5, key="rsi_adx"
            ),
            "position_size": st.sidebar.slider(
                "RSI仓位比例", 0.05, 0.5, 0.2, 0.05, key="rsi_pos"
            ),
        }

    if "spread" in selected_strategies:
        st.sidebar.subheader("跨期套利参数")
        near_symbol = st.sidebar.text_input("近月合约代码", value="RB2401", key="spread_near")
        far_symbol = st.sidebar.text_input("远月合约代码", value="RB2405", key="spread_far")
        strategy_params["spread"] = {
            "spread_ma_period": st.sidebar.slider(
                "价差均线周期", 5, 40, 20, key="spread_ma"
            ),
            "spread_entry_threshold": st.sidebar.slider(
                "入场阈值(标准差)", 1.0, 4.0, 2.0, 0.5, key="spread_thresh"
            ),
            "position_size": st.sidebar.slider(
                "套利仓位比例", 0.05, 0.3, 0.15, 0.05, key="spread_pos"
            ),
            "near_symbol": near_symbol,
            "far_symbol": far_symbol,
        }

    st.sidebar.divider()
    st.sidebar.subheader(" 组合管理")
    dynamic_weighting = st.sidebar.checkbox("动态调整策略权重（基于市场状态）", value=True, key="dynamic_weighting")
    total_allocation = st.sidebar.slider("总资金利用率上限", 0.3, 1.0, 0.8, 0.05, key="total_allocation")

    st.sidebar.divider()
    st.sidebar.subheader(" 展期设置")
    rollover_mode = st.sidebar.selectbox(
        "展期模式",
        ["liquidity", "time", "spread"],
        format_func=lambda x: {
            "liquidity": "流动性触发",
            "time": "时间触发",
            "spread": "价差触发",
        }.get(x, x),
    )
    rollover_params = {}
    if rollover_mode == "time":
        rollover_params["days_before_expiry"] = st.sidebar.slider(
            "到期前展期天数", 1, 15, 5, key="rollover_days"
        )
    elif rollover_mode == "liquidity":
        rollover_params["liquidity_ratio"] = st.sidebar.slider(
            "持仓量比值阈值", 1.0, 3.0, 1.5, 0.1, key="rollover_liq"
        )
    elif rollover_mode == "spread":
        rollover_params["spread_threshold"] = st.sidebar.slider(
            "价差阈值(元)", 5.0, 100.0, 20.0, 5.0, key="rollover_spread"
        )
        rollover_params["max_rollover_delay"] = st.sidebar.slider(
            "最大展期延迟天数", 1, 30, 10, key="rollover_delay"
        )

    st.sidebar.divider()
    st.sidebar.subheader("🛡️ 风控设置")
    risk_params = {
        "stop_loss_pct": st.sidebar.slider("单笔止损(%)", 0.5, 10.0, 2.0, 0.5) / 100,
        "max_position_pct": st.sidebar.slider("单合约最大仓位(%)", 5, 50, 20) / 100,
        "max_total_position_pct": st.sidebar.slider("总仓位上限(%)", 20, 100, 40) / 100,
        "rollover_cost_tolerance": st.sidebar.slider(
            "展期成本容忍度(元)", 10.0, 200.0, 50.0, 10.0
        ),
        "daily_loss_limit": st.sidebar.slider("日内最大亏损上限(%)", 0.5, 10.0, 3.0, 0.5) / 100,
    }

    st.sidebar.divider()
    st.sidebar.subheader("🔍 参数优化")
    enable_optimization = st.sidebar.checkbox("启用参数优化")
    optimize_metric = st.sidebar.selectbox(
        "优化目标", ["sharpe", "total_return_pct", "profit_factor"], key="opt_metric"
    )

    st.sidebar.divider()
    st.sidebar.subheader("💸 交易成本")
    commission = st.sidebar.number_input(
        "手续费率 (如 0.0001 表示万分之1)", min_value=0.0, max_value=0.01,
        value=0.0001, step=0.0001, format="%.4f", key="commission",
    )
    slippage = st.sidebar.number_input(
        "滑点 (如 0.0002 表示万分之2)", min_value=0.0, max_value=0.01,
        value=0.0002, step=0.0001, format="%.4f", key="slippage",
    )

    st.sidebar.divider()
    st.sidebar.subheader("📥 输出选项")
    save_trades = st.sidebar.checkbox("保存交易记录为 CSV", value=True, key="save_trades")

    if st.session_state.data_loaded:
        st.sidebar.divider()
        st.sidebar.subheader("📌 当前回测范围")
        df_cur = st.session_state.pybroker_df
        if df_cur is not None:
            syms = df_cur["symbol"].unique().tolist()
            date_min = str(df_cur["date"].min())[:10] if "date" in df_cur.columns else "N/A"
            date_max = str(df_cur["date"].max())[:10] if "date" in df_cur.columns else "N/A"
            st.sidebar.metric("品种数", len(syms))
            st.sidebar.caption(f"日期: {date_min} ~ {date_max}")
            st.sidebar.caption(f"数据行: {len(df_cur):,}")
            st.sidebar.caption(f"筛选状态: {'✅ 已筛选' if st.session_state.get('data_filtered') else '⚠️ 全量数据'}")

    return {
        "selected_strategies": selected_strategies,
        "strategy_params": strategy_params,
        "rollover_mode": rollover_mode,
        "rollover_params": rollover_params,
        "risk_params": risk_params,
        "enable_optimization": enable_optimization,
        "optimize_metric": optimize_metric,
        "dynamic_weighting": dynamic_weighting,
        "total_allocation": total_allocation,
        "commission": commission,
        "slippage": slippage,
        "save_trades": save_trades,
    }