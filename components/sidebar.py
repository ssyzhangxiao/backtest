"""侧边栏策略配置模块。"""
import streamlit as st
from core.strategies import STRATEGY_REGISTRY


def render_strategy_config() -> dict:
    """渲染策略配置面板（侧边栏），返回配置字典。"""
    st.sidebar.header("⚙️ 策略配置")

    selected_strategies = st.sidebar.multiselect(
        "选择策略",
        list(STRATEGY_REGISTRY.keys()),
        default=["ts_momentum"],
        format_func=lambda x: {
            "ts_momentum": "📈 时序动量",
            "roll_yield": "🔄 展期收益",
            "alpha019": "📉 Alpha019",
            "alpha032": "📊 Alpha032",
        }.get(x, x),
    )

    strategy_params = {}

    if "ts_momentum" in selected_strategies:
        st.sidebar.subheader("时序动量参数")
        _applied = st.session_state.get("applied_optimized_params", {})
        _is_applied = st.session_state.get("optimized_strategy") == "ts_momentum"
        strategy_params["ts_momentum"] = {
            "window": st.sidebar.slider(
                "动量窗口", 5, 60,
                int(_applied.get("window", 20)) if _is_applied else 20,
                key="tsm_window"
            ),
            "position_size": st.sidebar.slider(
                "仓位比例", 0.05, 0.5, 0.2, 0.05, key="tsm_pos"
            ),
        }

    if "roll_yield" in selected_strategies:
        st.sidebar.subheader("展期收益参数")
        _applied = st.session_state.get("applied_optimized_params", {})
        _is_applied = st.session_state.get("optimized_strategy") == "roll_yield"
        strategy_params["roll_yield"] = {
            "lookback": st.sidebar.slider(
                "回看窗口", 5, 60,
                int(_applied.get("lookback", 20)) if _is_applied else 20,
                key="ry_lookback"
            ),
            "entry_threshold": st.sidebar.slider(
                "入场阈值(%)", 0.5, 5.0, 2.0, 0.5, key="ry_entry"
            ),
            "exit_threshold": st.sidebar.slider(
                "出场阈值(%)", 0.1, 2.0, 0.5, 0.1, key="ry_exit"
            ),
            "position_size": st.sidebar.slider(
                "仓位比例", 0.05, 0.5, 0.2, 0.05, key="ry_pos"
            ),
        }

    if "alpha019" in selected_strategies:
        st.sidebar.subheader("Alpha019参数")
        _applied = st.session_state.get("applied_optimized_params", {})
        _is_applied = st.session_state.get("optimized_strategy") == "alpha019"
        strategy_params["alpha019"] = {
            "short_window": st.sidebar.slider(
                "短期窗口", 3, 20,
                int(_applied.get("short_window", 7)) if _is_applied else 7,
                key="a019_short"
            ),
            "long_window": st.sidebar.slider(
                "长期窗口", 60, 360,
                int(_applied.get("long_window", 250)) if _is_applied else 250,
                key="a019_long"
            ),
            "position_size": st.sidebar.slider(
                "仓位比例", 0.05, 0.5, 0.2, 0.05, key="a019_pos"
            ),
        }

    if "alpha032" in selected_strategies:
        st.sidebar.subheader("Alpha032参数")
        _applied = st.session_state.get("applied_optimized_params", {})
        _is_applied = st.session_state.get("optimized_strategy") == "alpha032"
        strategy_params["alpha032"] = {
            "ma_window": st.sidebar.slider(
                "均线窗口", 3, 20,
                int(_applied.get("ma_window", 7)) if _is_applied else 7,
                key="a032_ma"
            ),
            "corr_window": st.sidebar.slider(
                "相关性窗口", 60, 360,
                int(_applied.get("corr_window", 230)) if _is_applied else 230,
                key="a032_corr"
            ),
            "position_size": st.sidebar.slider(
                "仓位比例", 0.05, 0.5, 0.2, 0.05, key="a032_pos"
            ),
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
