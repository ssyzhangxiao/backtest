"""
商品期货多策略量化回测系统 - Streamlit 前端入口。

运行方式：
    streamlit run app.py

功能模块：
1. 数据导入：上传CSV、预览、合约列表
2. 策略配置：侧边栏，支持多策略选择、参数设置、展期模式、优化开关、风控设置
3. 回测运行与结果展示：绩效卡片、资金曲线、展期统计、持仓明细、交易记录
4. 优化结果展示：参数热力图、最佳参数表、验证集对比
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import pybroker
from pybroker import Strategy, StrategyConfig

from core.data_loader import DataLoader
from core.environment import EnvironmentAdapter
from core.strategies import STRATEGY_REGISTRY, create_strategy
from core.rollover import RolloverManager, RolloverMode
from core.risk_manager import RiskManager
from core.optimizer import ParameterOptimizer
from utils.plots import PlotManager
from utils.metrics import MetricsCalculator


st.set_page_config(
    page_title="期货量化回测系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

_PYBROKER_COLUMNS_REGISTERED = False


@st.cache_data(show_spinner=False)
def load_data_cached(data_dir: str, file_pattern: str):
    loader = DataLoader(data_dir)
    loader.load_csv_files(file_pattern=file_pattern)
    loader.identify_dominant_contracts()
    loader.build_continuous_series()
    return loader


@st.cache_data(show_spinner=False)
def compute_env_cached(pybroker_df: pd.DataFrame):
    env_adapter = EnvironmentAdapter()
    return env_adapter.compute_for_pybroker(pybroker_df)


def register_pybroker_columns():
    global _PYBROKER_COLUMNS_REGISTERED
    if not _PYBROKER_COLUMNS_REGISTERED:
        pybroker.register_columns(
            "open_interest",
            "is_dominant",
            "dominant_symbol",
            "prev_dominant_symbol",
            "rollover_flag",
            "product",
            "env_atr",
            "env_adx",
            "env_plus_di",
            "env_minus_di",
            "env_market_regime",
            "env_trend_score",
            "env_compression_score",
            "env_momentum_score",
            "env_liquidity_score",
            "env_bearish_exhaustion",
            "env_bullish_exhaustion",
            "env_weight_trend",
            "env_weight_reversal",
            "env_weight_spread",
        )
        _PYBROKER_COLUMNS_REGISTERED = True


def init_session_state():
    """初始化 Streamlit session state。"""
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "backtest_run" not in st.session_state:
        st.session_state.backtest_run = False
    if "backtest_result" not in st.session_state:
        st.session_state.backtest_result = None
    if "pybroker_df" not in st.session_state:
        st.session_state.pybroker_df = None
    if "data_loader" not in st.session_state:
        st.session_state.data_loader = None
    if "env_df" not in st.session_state:
        st.session_state.env_df = None


def render_data_import():
    """渲染数据导入模块。"""
    st.header("📁 数据导入")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("数据源设置")
        data_dir = st.text_input("数据目录路径", value=DATA_DIR)

        file_pattern = st.text_input(
            "文件匹配模式",
            value="*.csv",
            help="如 SHFE.RB.csv 仅加载螺纹钢，*.csv 加载全部",
        )

        _uploaded_files = st.file_uploader(
            "上传CSV文件",
            type=["csv"],
            accept_multiple_files=True,
            help="支持两种格式：1)合约格式(date,symbol,open_interest) 2)品种格式(datetime,position)",
        )

        if st.button("加载数据", type="primary"):
            try:
                with st.spinner("正在加载数据..."):
                    loader = load_data_cached(data_dir, file_pattern)

                    st.session_state.data_loader = loader
                    st.session_state.data_loaded = True

                    pybroker_df = loader.get_pybroker_df()
                    env_df = compute_env_cached(pybroker_df)
                    st.session_state.pybroker_df = env_df
                    st.session_state.env_df = env_df

                    st.session_state.backtest_run = False
                    st.session_state.backtest_result = None

                st.success(f"数据加载成功！模式: {loader.data_mode}")
            except Exception as e:
                st.error(f"数据加载失败: {e}")

    with col2:
        if st.session_state.data_loaded and st.session_state.data_loader:
            loader = st.session_state.data_loader
            summary = loader.get_data_summary()

            st.subheader("数据概览")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("品种/合约数量", summary.get("total_symbols", 0))
            with col_b:
                st.metric("数据起始", summary.get("date_range", ("N/A", ""))[0])
            with col_c:
                st.metric("数据截止", summary.get("date_range", ("", "N/A"))[1])

            data_mode = summary.get("data_mode", "unknown")
            st.info(f"数据模式: {'品种指数' if data_mode == 'product' else '合约展期'}")

            st.subheader("品种列表")
            for product, info in summary.get("products", {}).items():
                st.write(f"**{product}**: {info['contracts']} 个品种/合约")

            if st.checkbox("预览原始数据"):
                st.dataframe(loader.all_contracts.head(20), use_container_width=True)

            if data_mode == "contract":
                if st.checkbox("预览展期信息"):
                    rollover = loader.get_rollover_dates()
                    if not rollover.empty:
                        st.dataframe(rollover, use_container_width=True)
                    else:
                        st.info("未检测到展期事件")


def render_strategy_config():
    """渲染策略配置面板（侧边栏）。"""
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
        strategy_params["dual_ma"] = {
            "short_ma": st.sidebar.slider("短期均线周期", 3, 15, 5, key="dual_short"),
            "long_ma": st.sidebar.slider("长期均线周期", 10, 60, 20, key="dual_long"),
            "adx_threshold": st.sidebar.slider(
                "ADX趋势阈值", 15.0, 40.0, 25.0, 0.5, key="dual_adx"
            ),
            "position_size": st.sidebar.slider(
                "仓位比例", 0.05, 0.5, 0.3, 0.05, key="dual_pos"
            ),
        }

    if "rsi" in selected_strategies:
        st.sidebar.subheader("RSI参数")
        strategy_params["rsi"] = {
            "rsi_period": st.sidebar.slider("RSI周期", 5, 30, 14, key="rsi_period"),
            "oversold": st.sidebar.slider(
                "超卖阈值", 15.0, 35.0, 30.0, 1.0, key="rsi_oversold"
            ),
            "overbought": st.sidebar.slider(
                "超买阈值", 65.0, 85.0, 70.0, 1.0, key="rsi_overbought"
            ),
            "position_size": st.sidebar.slider(
                "RSI仓位比例", 0.05, 0.5, 0.2, 0.05, key="rsi_pos"
            ),
        }

    if "spread" in selected_strategies:
        st.sidebar.subheader("跨期套利参数")
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
        }

    st.sidebar.divider()
    st.sidebar.subheader("🔄 展期设置")
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

    st.sidebar.divider()
    st.sidebar.subheader("🛡️ 风控设置")
    risk_params = {
        "stop_loss_pct": st.sidebar.slider("单笔止损(%)", 0.5, 10.0, 2.0, 0.5) / 100,
        "max_position_pct": st.sidebar.slider("单合约最大仓位(%)", 5, 50, 20) / 100,
        "max_total_position_pct": st.sidebar.slider("总仓位上限(%)", 20, 100, 40) / 100,
        "rollover_cost_tolerance": st.sidebar.slider(
            "展期成本容忍度(元)", 10.0, 200.0, 50.0, 10.0
        ),
    }

    st.sidebar.divider()
    st.sidebar.subheader("🔍 参数优化")
    enable_optimization = st.sidebar.checkbox("启用参数优化")
    optimize_metric = st.sidebar.selectbox(
        "优化目标", ["sharpe", "total_return_pct", "profit_factor"], key="opt_metric"
    )

    return {
        "selected_strategies": selected_strategies,
        "strategy_params": strategy_params,
        "rollover_mode": rollover_mode,
        "rollover_params": rollover_params,
        "risk_params": risk_params,
        "enable_optimization": enable_optimization,
        "optimize_metric": optimize_metric,
    }


def run_backtest(config: dict):
    if not st.session_state.data_loaded or st.session_state.pybroker_df is None:
        st.error("请先加载数据！")
        return None

    register_pybroker_columns()

    df = st.session_state.pybroker_df

    symbols = df["symbol"].unique().tolist()

    initial_cash = st.session_state.get("initial_cash", 1_000_000)
    pybroker_config = StrategyConfig(initial_cash=initial_cash)

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


def render_backtest_results(result):
    """
    渲染回测结果。

    Args:
        result: PyBroker 回测结果
    """
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

    if data_mode == "contract":
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📈 资金曲线", "📉 回撤分析", "🔄 展期统计", "📋 交易记录", "📊 月度收益"]
        )
    else:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📈 资金曲线", "📉 回撤分析", "📋 交易记录", "📊 月度收益"]
        )

    with tab1:
        if portfolio_df is not None and not portfolio_df.empty:
            fig = PlotManager.plot_equity_curve(portfolio_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("无资金曲线数据")

    with tab2:
        if portfolio_df is not None and not portfolio_df.empty:
            fig = PlotManager.plot_drawdown(portfolio_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("无回撤数据")

    if data_mode == "contract":
        with tab3:
            if st.session_state.data_loader:
                loader = st.session_state.data_loader
                rollover_dates = loader.get_rollover_dates()

                if not rollover_dates.empty:
                    fig = PlotManager.plot_rollover_timeline(rollover_dates)
                    st.plotly_chart(fig, use_container_width=True)

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

                    st.dataframe(rollover_dates, use_container_width=True)
                else:
                    st.info("未检测到展期事件")

        trade_tab = tab4
        monthly_tab = tab5
    else:
        trade_tab = tab3
        monthly_tab = tab4

    with trade_tab:
        if trades_df is not None and not trades_df.empty:
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        elif orders_df is not None and not orders_df.empty:
            st.subheader("订单记录")
            st.dataframe(orders_df, use_container_width=True, hide_index=True)
        else:
            st.info("无交易记录")

    with monthly_tab:
        if portfolio_df is not None and not portfolio_df.empty:
            fig = PlotManager.plot_monthly_returns(portfolio_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("无月度收益数据")

    st.divider()
    st.subheader("完整指标表")
    if all_metrics:
        metrics_display = pd.DataFrame(
            [{"指标": k, "值": v} for k, v in all_metrics.items()]
        )
        st.dataframe(metrics_display, use_container_width=True, hide_index=True)


def render_optimization(config: dict):
    """
    渲染参数优化模块。

    Args:
        config: 策略配置
    """
    st.header("🔍 参数优化")

    if not st.session_state.data_loaded:
        st.warning("请先加载数据！")
        return

    strat_name = (
        config["selected_strategies"][0] if config["selected_strategies"] else "dual_ma"
    )
    strat_class = STRATEGY_REGISTRY.get(strat_name)

    if strat_class is None:
        st.error("请选择至少一个策略")
        return

    st.subheader(f"优化策略: {strat_name}")

    col1, col2 = st.columns(2)
    with col1:
        if strat_name == "dual_ma":
            short_ma_range = st.text_input(
                "短期均线搜索范围", "3,5,7,10", key="opt_short"
            )
            long_ma_range = st.text_input(
                "长期均线搜索范围", "15,20,30,40", key="opt_long"
            )
            try:
                param_grid = {
                    "short_ma": [int(x.strip()) for x in short_ma_range.split(",")],
                    "long_ma": [int(x.strip()) for x in long_ma_range.split(",")],
                }
            except ValueError:
                st.error("参数范围格式错误，请输入逗号分隔的整数，如: 3,5,7,10")
                return
        elif strat_name == "rsi":
            rsi_period_range = st.text_input(
                "RSI周期搜索范围", "7,10,14,21", key="opt_rsi"
            )
            try:
                param_grid = {
                    "rsi_period": [int(x.strip()) for x in rsi_period_range.split(",")],
                }
            except ValueError:
                st.error("参数范围格式错误，请输入逗号分隔的整数，如: 7,10,14,21")
                return
        else:
            st.info("跨期套利策略暂不支持参数优化")
            return

    with col2:
        optimize_mode = st.radio("优化模式", ["网格搜索", "滚动优化"], key="opt_mode")
        metric = config.get("optimize_metric", "sharpe")

    if st.button("开始优化", type="primary"):
        optimizer = ParameterOptimizer(param_grid=param_grid, metric=metric)

        df = st.session_state.pybroker_df
        symbols = df["symbol"].unique().tolist()

        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current, total):
            if total:
                progress_bar.progress(current / total)
                status_text.text(f"正在优化: {current}/{total}")

        try:
            if optimize_mode == "网格搜索":
                results_df = optimizer.grid_search(
                    strategy_class=strat_class,
                    data=df,
                    symbols=symbols,
                    progress_callback=progress_callback,
                )
            else:
                results_df = optimizer.rolling_optimize(
                    strategy_class=strat_class,
                    data=df,
                    symbols=symbols,
                    progress_callback=progress_callback,
                )

            progress_bar.progress(1.0)
            status_text.text("优化完成！")

            if not results_df.empty:
                st.subheader("优化结果")

                best_params = optimizer.get_best_params()
                if best_params:
                    st.success(f"最佳参数: {best_params}")

                st.dataframe(
                    results_df.head(20), use_container_width=True, hide_index=True
                )

                if len(param_grid) >= 2:
                    param_keys = list(param_grid.keys())
                    fig = PlotManager.plot_param_heatmap(
                        results_df, param_keys[0], param_keys[1], metric
                    )
                    st.plotly_chart(fig, use_container_width=True)

                save_path = os.path.join(DATA_DIR, "..", "optimization_results.json")
                optimizer.save_results(save_path)
                st.info(f"优化结果已保存到 {save_path}")
            else:
                st.warning("优化未产生有效结果")

        except Exception as e:
            st.error(f"优化失败: {e}")
            import traceback

            st.code(traceback.format_exc())


def render_data_analysis():
    """渲染数据分析模块。"""
    st.header("📉 数据分析")

    if not st.session_state.data_loaded or st.session_state.env_df is None:
        st.warning("请先加载数据！")
        return

    df = st.session_state.env_df

    tab1, tab2, tab3 = st.tabs(["K线图", "市场状态", "价差分析"])

    with tab1:
        symbols = df["symbol"].unique().tolist()
        selected_symbol = st.selectbox("选择合约", symbols, key="analysis_symbol")

        symbol_data = df[df["symbol"] == selected_symbol].copy()
        if not symbol_data.empty:
            fig = PlotManager.plot_price_with_signals(symbol_data)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if "env_market_regime" in df.columns:
            dominant_data = df[df["is_dominant"]].copy()
            if not dominant_data.empty:
                fig = PlotManager.plot_regime_overlay(
                    dominant_data.rename(columns={"env_market_regime": "market_regime"})
                )
                st.plotly_chart(fig, use_container_width=True)

                regime_counts = dominant_data["env_market_regime"].value_counts()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("趋势市天数", regime_counts.get("trend", 0))
                with col2:
                    st.metric("震荡市天数", regime_counts.get("range", 0))

                if "env_trend_score" in dominant_data.columns:
                    st.subheader("趋势强度分数")
                    ts = dominant_data["env_trend_score"]
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("平均趋势分数", f"{ts.mean():.3f}")
                    with col_b:
                        st.metric("最大趋势分数", f"{ts.max():.3f}")
                    with col_c:
                        st.metric("趋势分数 > 0.5 占比", f"{(ts > 0.5).mean():.1%}")

                if "env_bearish_exhaustion" in dominant_data.columns:
                    bearish_count = dominant_data["env_bearish_exhaustion"].sum()
                    bullish_count = dominant_data["env_bullish_exhaustion"].sum()
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("看跌衰竭信号", int(bearish_count))
                    with col_b:
                        st.metric("看涨衰竭信号", int(bullish_count))

                if "env_weight_trend" in dominant_data.columns:
                    st.subheader("动态策略权重")
                    wt = dominant_data["env_weight_trend"].mean()
                    wr = dominant_data["env_weight_reversal"].mean()
                    ws = dominant_data["env_weight_spread"].mean()
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("趋势策略平均权重", f"{wt:.2%}")
                    with col_b:
                        st.metric("反转策略平均权重", f"{wr:.2%}")
                    with col_c:
                        st.metric("套利策略平均权重", f"{ws:.2%}")

    with tab3:
        if st.session_state.data_loader:
            loader = st.session_state.data_loader
            products = loader.get_product_symbols()

            for product, syms in products.items():
                if len(syms) >= 2:
                    st.subheader(f"{product} 价差分析")
                    product_data = df[df["product"] == product]

                    pivot_close = product_data.pivot_table(
                        values="close", index="date", columns="symbol"
                    )

                    if pivot_close.shape[1] >= 2:
                        cols = pivot_close.columns.tolist()
                        near = cols[0]
                        far = cols[-1]
                        spread = pivot_close[near] - pivot_close[far]

                        import plotly.graph_objects as go

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=spread.index,
                                y=spread.values,
                                mode="lines",
                                name=f"{near}-{far} 价差",
                            )
                        )
                        fig.update_layout(
                            title=f"{product} 跨期价差",
                            xaxis_title="日期",
                            yaxis_title="价差",
                            template="plotly_white",
                            height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)


def main():
    """主函数入口。"""
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
            st.warning("请先在'数据导入'页面加载数据！")
        else:
            st.subheader("当前配置")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**策略**: ", ", ".join(config["selected_strategies"]))
            with col2:
                st.write("**展期模式**: ", config["rollover_mode"])
            with col3:
                st.write("**初始资金**: ", f"{st.session_state.initial_cash:,.0f}")

            if st.button("▶️ 开始回测", type="primary", use_container_width=True):
                with st.spinner("回测运行中..."):
                    result = run_backtest(config)
                    if result is not None:
                        st.session_state.backtest_result = result
                        st.session_state.backtest_run = True
                        st.success("回测完成！")

            if st.session_state.backtest_run and st.session_state.backtest_result:
                render_backtest_results(st.session_state.backtest_result)

    elif page == "🔍 参数优化":
        if config.get("enable_optimization"):
            render_optimization(config)
        else:
            st.info("请在左侧勾选'启用参数优化'")


if __name__ == "__main__":
    main()
