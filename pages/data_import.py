"""数据导入页面模块。"""
import streamlit as st
import pandas as pd

from config import DATA_DIR
from utils.session_state import load_data_cached, compute_env_cached


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
                    st.session_state.pybroker_df_full = env_df.copy()
                    st.session_state.pybroker_df = env_df.copy()
                    st.session_state.env_df = env_df.copy()
                    st.session_state.data_filtered = False

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

            st.divider()
            st.subheader("数据切片")
            symbol_options = loader.all_contracts["symbol"].unique().tolist()
            selected_symbols = st.multiselect(
                "选择回测合约（默认全选）", symbol_options, default=symbol_options,
                key="selected_symbols",
            )

            date_col = "date" if "date" in loader.all_contracts.columns else loader.all_contracts.columns[0]
            _min_date = loader.all_contracts[date_col].min()
            _max_date = loader.all_contracts[date_col].max()
            if pd.notna(_min_date) and pd.notna(_max_date):
                start_date = st.date_input("回测开始日期", value=_min_date, key="bt_start_date")
                end_date = st.date_input("回测结束日期", value=_max_date, key="bt_end_date")

            full_df = st.session_state.get("pybroker_df_full", loader.get_pybroker_df())
            filtered_df = full_df.copy()
            filter_applied = False

            if selected_symbols and len(selected_symbols) > 0:
                if set(selected_symbols) != set(filtered_df["symbol"].unique()):
                    filtered_df = filtered_df[filtered_df["symbol"].isin(selected_symbols)]
                    filter_applied = True

            if pd.notna(_min_date) and pd.notna(_max_date):
                filtered_df["date"] = pd.to_datetime(filtered_df["date"])
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date)
                if start_ts > filtered_df["date"].min() or end_ts < filtered_df["date"].max():
                    filtered_df = filtered_df[(filtered_df["date"] >= start_ts) &
                                           (filtered_df["date"] <= end_ts)]
                    filter_applied = True

            need_update = (
                filter_applied or
                not st.session_state.get("data_filtered") or
                set(st.session_state.pybroker_df["symbol"].unique()) != set(filtered_df["symbol"].unique())
            )
            if need_update:
                filtered_env_df = compute_env_cached(filtered_df)
                st.session_state.pybroker_df = filtered_env_df.copy()
                st.session_state.env_df = filtered_env_df.copy()
                st.session_state.data_filtered = filter_applied
                st.session_state.filtered_symbols_count = filtered_df["symbol"].nunique()
                st.session_state.filtered_date_start = str(filtered_df["date"].dt.date.min())
                st.session_state.filtered_date_end = str(filtered_df["date"].dt.date.max())

            if st.checkbox("预览原始数据"):
                preview_df = loader.all_contracts.copy()
                if selected_symbols and len(selected_symbols) > 0:
                    preview_df = preview_df[preview_df["symbol"].isin(selected_symbols)]
                if pd.notna(_min_date) and pd.notna(_max_date):
                    preview_df[date_col] = pd.to_datetime(preview_df[date_col])
                    preview_df = preview_df[(preview_df[date_col] >= pd.Timestamp(start_date)) &
                                           (preview_df[date_col] <= pd.Timestamp(end_date))]

                st.dataframe(preview_df.head(20), use_container_width=True)
                st.metric("预览数据行数", f"{len(preview_df):,}")
                st.metric("回测数据行数", f"{len(st.session_state.pybroker_df):,}")

            if data_mode == "contract":
                if st.checkbox("预览展期信息"):
                    rollover = loader.get_rollover_dates()
                    if not rollover.empty:
                        st.dataframe(rollover, use_container_width=True)
                    else:
                        st.info("未检测到展期事件")