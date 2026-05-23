"""参数优化页面模块。"""
import os
import time

import streamlit as st
import pandas as pd
import plotly.express as px

from core.optimizer import ParameterOptimizer
from core.strategies import STRATEGY_REGISTRY
from utils.plots import PlotManager
from config import DATA_DIR


def render_optimization(config: dict):
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
            short_ma_values = st.multiselect(
                "短期均线周期", options=[3, 5, 7, 10, 12, 15],
                default=[5, 10], key="opt_short"
            )
            long_ma_values = st.multiselect(
                "长期均线周期", options=[15, 20, 25, 30, 40, 50],
                default=[20, 30], key="opt_long"
            )
            if not short_ma_values or not long_ma_values:
                st.warning("请至少选择一个参数值")
                return
            param_grid = {
                "short_ma": short_ma_values,
                "long_ma": long_ma_values,
            }
        elif strat_name == "rsi":
            rsi_period_values = st.multiselect(
                "RSI周期", options=[5, 7, 10, 14, 21, 28],
                default=[10, 14, 21], key="opt_rsi"
            )
            if not rsi_period_values:
                st.warning("请至少选择一个参数值")
                return
            param_grid = {
                "rsi_period": rsi_period_values,
            }
        else:
            st.info("跨期套利策略暂不支持参数优化")
            return

    with col2:
        optimize_mode = st.radio("优化模式", ["网格搜索", "滚动优化"], key="opt_mode")
        metric = config.get("optimize_metric", "sharpe")
        train_days = st.number_input("训练窗口（交易日）", min_value=21, max_value=504, value=252, step=21, key="opt_train_days")
        test_days = st.number_input("测试窗口（交易日）", min_value=5, max_value=126, value=21, step=5, key="opt_test_days")

    if st.button("开始优化", type="primary"):
        optimizer = ParameterOptimizer(param_grid=param_grid, metric=metric)

        df = st.session_state.pybroker_df
        symbols = df["symbol"].unique().tolist()

        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

        def progress_callback(current, total):
            if total:
                progress_bar.progress(current / total)
                elapsed = time.time() - start_time
                if current > 0:
                    eta = elapsed / current * (total - current)
                    status_text.text(
                        f"优化进度: {current}/{total}，预计剩余 {eta:.1f} 秒"
                    )
                else:
                    status_text.text(f"优化进度: {current}/{total}")

        try:
            if optimize_mode == "网格搜索":
                results_df = optimizer.grid_search(
                    strategy_class=strat_class,
                    data=df,
                    symbols=symbols,
                    progress_callback=progress_callback,
                )
            else:
                train_months_approx = max(1, train_days // 21)
                test_months_approx = max(1, test_days // 21)
                results_df = optimizer.rolling_optimize(
                    strategy_class=strat_class,
                    data=df,
                    symbols=symbols,
                    train_months=train_months_approx,
                    test_months=test_months_approx,
                    progress_callback=progress_callback,
                )

            progress_bar.progress(1.0)
            status_text.text("优化完成！")

            if not results_df.empty:
                st.subheader("优化结果")

                best_params = optimizer.get_best_params()
                if best_params:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.success(f"最佳参数: {best_params}")
                    with col_b:
                        if st.button("📌 应用到回测", key="apply_to_backtest"):
                            st.session_state["applied_optimized_params"] = best_params
                            st.session_state["optimized_strategy"] = strat_name
                            st.info("参数已应用到回测配置，请前往\"运行回测\"页面执行回测")

                st.dataframe(
                    results_df.head(20), use_container_width=True, hide_index=True
                )

                param_cols = list(param_grid.keys())

                if len(param_cols) >= 2:
                    with st.expander("二维参数热力图", expanded=True):
                        fig_heat = PlotManager.plot_param_heatmap(
                            results_df, param_cols[0], param_cols[1], metric
                        )
                        st.plotly_chart(fig_heat, use_container_width=True)

                if len(param_cols) >= 2:
                    with st.expander("3D 参数曲面图"):
                        fig_3d = PlotManager.plot_surface_3d(
                            results_df, param_cols[0], param_cols[1], metric
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)

                if len(param_cols) >= 2:
                    with st.expander("平行坐标图"):
                        fig_parallel = PlotManager.plot_parallel_coordinate(
                            results_df, param_cols, metric
                        )
                        st.plotly_chart(fig_parallel, use_container_width=True)

                for pc in param_cols:
                    with st.expander(f"参数扫描: {pc}"):
                        extra_metrics = [c for c in ["total_return_pct", "max_drawdown_pct", "profit_factor"]
                                         if c in results_df.columns and c != metric]
                        fig_scan = PlotManager.plot_param_scan(
                            results_df, pc, metric=metric,
                            extra_metrics=extra_metrics if extra_metrics else None,
                        )
                        st.plotly_chart(fig_scan, use_container_width=True)

                if len(param_cols) >= 2:
                    with st.expander("参数重要性"):
                        fig_imp = PlotManager.plot_param_importance(
                            results_df, param_cols, metric
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

                if len(param_cols) == 2:
                    with st.expander("等高线图"):
                        pivot = results_df.pivot_table(
                            values=metric, index=param_cols[1],
                            columns=param_cols[0], aggfunc="mean"
                        )
                        contour_fig = px.contour(
                            x=pivot.columns, y=pivot.index, z=pivot.values,
                            labels={"x": param_cols[0], "y": param_cols[1], "z": metric},
                            title=f"等高线图 ({metric})"
                        )
                        st.plotly_chart(contour_fig, use_container_width=True)

                if optimize_mode == "滚动优化":
                    st.subheader("滚动优化详情")

                    for param in param_cols:
                        if param in results_df.columns:
                            with st.expander(f"参数稳定性: {param}"):
                                fig_stab = PlotManager.plot_param_stability(
                                    results_df, param
                                )
                                st.plotly_chart(fig_stab, use_container_width=True)

                    if metric in results_df.columns:
                        with st.expander(f"测试集 {metric} 曲线"):
                            fig_metric = px.line(
                                results_df, x="test_end", y=metric,
                                title=f"测试集 {metric} 曲线"
                            )
                            st.plotly_chart(fig_metric, use_container_width=True)

                    display_cols = ["test_start", "test_end"] + param_cols + [metric]
                    available_cols = [c for c in display_cols if c in results_df.columns]
                    st.dataframe(
                        results_df[available_cols].head(20),
                        use_container_width=True, hide_index=True
                    )

                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 下载优化结果为 CSV",
                    csv, "optimization_results.csv", "text/csv"
                )

                json_str = results_df.to_json(orient="records", force_ascii=False, indent=2)
                st.download_button(
                    "📥 下载优化结果为 JSON",
                    json_str, "optimization_results.json", "application/json"
                )

                save_path = os.path.join(DATA_DIR, "..", "optimization_results.json")
                optimizer.save_results(save_path)
                st.info(f"优化结果已保存到 {save_path}")
            else:
                st.warning("优化未产生有效结果")

        except Exception as e:
            st.error(f"优化失败: {e}")
            import traceback
            st.code(traceback.format_exc())