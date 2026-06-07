"""
P0 整改（2026-06-07）：core/strategies/ 已整体删除，
ParameterOptimizer 依赖的 strategy_class 接口已废弃。
本页面改为调用 Pipeline 编排器（规则17 单一公共入口），
具体参数优化由 runner.optimization.* 公共系统执行。

P1 整改（2026-06-07）：参数名称与候选值从 StrategyProfile.param_ranges
动态获取，避免在 UI 层硬编码参数名/值与子策略体系脱耦（规则17）。
"""
import os

import streamlit as st
import plotly.express as px

from core.config import DATA_DIR, StrategyLibrary
from core.config.strategy_profiles import STRATEGY_NAMES
from utils.plots import PlotManager


# ---------------------------------------------------------------------------
# P1 整改：从 StrategyProfile.param_ranges 动态生成 multiselect
# ---------------------------------------------------------------------------
# Fallback 候选值：仅在 StrategyProfile.param_ranges 中未找到对应参数时使用
_PARAM_FALLBACKS: dict[str, list] = {
    "trend":         {"window": [5, 10, 15, 20, 30, 40, 60]},
    "term_structure": {
        "lookback": [5, 10, 15, 20, 30, 40, 60],
        "entry_threshold": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    },
    "mean_reversion": {
        "short_window": [3, 5, 7, 10, 14],
        "long_window": [120, 180, 250, 300, 360],
    },
    "vol_breakout": {
        "ma_window": [3, 5, 7, 10, 14],
        "corr_window": [120, 180, 230, 300, 360],
    },
}

# 各参数默认选中的下标（取 param_ranges 中前 2 个值）
_DEFAULT_PARAM_PICKS: dict[str, dict[str, list]] = {
    "trend":         {"window": [10, 20]},
    "term_structure": {"lookback": [10, 20], "entry_threshold": [1.5, 2.0]},
    "mean_reversion": {"short_window": [5, 7], "long_window": [180, 250]},
    "vol_breakout":   {"ma_window": [5, 7], "corr_window": [180, 230]},
}


def _get_param_choices(strategy: str, param_name: str) -> list:
    """
    从 StrategyProfile.param_ranges 读取候选值，未命中时回退到 _PARAM_FALLBACKS。
    """
    lib = StrategyLibrary()
    profile = lib.get_profile(strategy)
    if profile is not None and param_name in profile.param_ranges:
        return list(profile.param_ranges[param_name])
    return list(_PARAM_FALLBACKS.get(strategy, {}).get(param_name, []))


def _get_default_picks(strategy: str, param_name: str) -> list:
    """从 _DEFAULT_PARAM_PICKS 读取默认值；若不可用，回退到候选值的前 2 个。"""
    explicit = _DEFAULT_PARAM_PICKS.get(strategy, {}).get(param_name)
    if explicit is not None:
        return explicit
    choices = _get_param_choices(strategy, param_name)
    return choices[1:3] if len(choices) >= 3 else choices[:2]


def _param_multiselect(
    label: str,
    key: str,
    strategy: str,
    param_name: str,
) -> list:
    """统一的参数多选器：候选值/默认值均来自 StrategyProfile。"""
    choices = _get_param_choices(strategy, param_name)
    default = _get_default_picks(strategy, param_name)
    return st.multiselect(
        label,
        options=choices,
        default=default,
        key=key,
    )


def render_optimization(config: dict):
    st.header("🔍 参数优化")

    if not st.session_state.data_loaded:
        st.warning("请先加载数据！")
        return

    strat_name = (
        config["selected_strategies"][0] if config["selected_strategies"] else "trend"
    )
    if strat_name not in STRATEGY_NAMES:
        st.error(f"未知策略: {strat_name}")
        return

    st.subheader(f"优化策略: {strat_name}")

    col1, col2 = st.columns(2)
    with col1:
        # P1 整改（2026-06-07）：参数候选值/默认值从 StrategyProfile.param_ranges 动态获取
        # 候选参数定义：(显示标签, Streamlit key, StrategyProfile.param_ranges 中的 key)
        _PARAM_DEFS: dict[str, list[tuple[str, str, str]]] = {
            "trend": [("动量窗口", "opt_trend_window", "window")],
            "term_structure": [
                ("回看窗口", "opt_ts_lookback", "lookback"),
                ("入场阈值(%)", "opt_ts_entry", "entry_threshold"),
            ],
            "mean_reversion": [
                ("短期窗口", "opt_mr_short", "short_window"),
                ("长期窗口", "opt_mr_long", "long_window"),
            ],
            "vol_breakout": [
                ("均线窗口", "opt_vb_ma", "ma_window"),
                ("相关性窗口", "opt_vb_corr", "corr_window"),
            ],
        }

        param_defs = _PARAM_DEFS.get(strat_name)
        if not param_defs:
            st.info("该策略暂不支持参数优化")
            return

        param_grid: dict[str, list] = {}
        for label, key, param_name in param_defs:
            values = _param_multiselect(
                label=label,
                key=key,
                strategy=strat_name,
                param_name=param_name,
            )
            if not values:
                st.warning("请至少选择一个参数值")
                return
            param_grid[param_name] = values

    with col2:
        optimize_mode = st.radio("优化模式", ["网格搜索", "滚动优化"], key="opt_mode")
        metric = config.get("optimize_metric", "sharpe")
        train_days = st.number_input("训练窗口（交易日）", min_value=21, max_value=504, value=252, step=21, key="opt_train_days")
        test_days = st.number_input("测试窗口（交易日）", min_value=5, max_value=126, value=21, step=5, key="opt_test_days")

    if st.button("开始优化", type="primary"):
        # P0 整改（2026-06-07）：不再使用 ParameterOptimizer(strategy_class=...) 接口
        # （该接口依赖已删除的子策略类）。改为委托 Pipeline 编排器执行参数优化，
        # 符合规则 17（不重复造轮子，统一公共入口）。
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("正在启动 Pipeline 参数优化...")

        try:
            from runner.pipeline import Pipeline

            pipe = Pipeline(config.get("config_path", "config.yaml"))
            pipe.with_config(
                optimize_metric=metric,
            )

            # Pipeline.optimize() 内部使用 runner.optimization.grid_search_single_strategy
            # 完成实际优化（规则17 公共系统调用）
            results_df = pipe.optimize(
                strategy=strat_name,
                param_grid=param_grid,
                mode="grid" if optimize_mode == "网格搜索" else "rolling",
                progress_callback=lambda cur, tot: (
                    progress_bar.progress(cur / tot) if tot else None
                ),
            )

            progress_bar.progress(1.0)
            status_text.text("优化完成！")

            if not results_df.empty:
                st.subheader("优化结果")

                best_params = (
                    results_df.iloc[0].to_dict() if metric in results_df.columns else None
                )
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
                st.info(f"优化结果建议保存到 {save_path}（请使用 run_optimize.py CLI）")
            else:
                st.warning("优化未产生有效结果")

        except Exception as e:
            st.error(f"优化失败: {e}")
            import traceback
            st.code(traceback.format_exc())