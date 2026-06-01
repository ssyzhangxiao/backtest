"""数据导入页面模块。"""
import streamlit as st

from core.config import DATA_DIR
from utils.session_state import load_data_cached, load_tqsdk_cached, compute_env_cached


def _get_tqsdk_defaults():
    """从config.yaml读取TqSdk凭证，回退到环境变量。"""
    import os
    phone = os.environ.get("TQSDK_PHONE", "")
    password = os.environ.get("TQSDK_PASSWORD", "")
    if not phone or not password:
        try:
            import yaml
            with open("config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            data_cfg = cfg.get("data", {})
            phone = phone or data_cfg.get("tqsdk_phone", "")
            password = password or data_cfg.get("tqsdk_password", "")
        except Exception:
            pass
    return phone, password


_DEFAULT_PHONE, _DEFAULT_PASSWORD = _get_tqsdk_defaults()

# 品种分组（用于 TqSdk 模式下的多选）
_SYMBOL_GROUPS = {
    "黑色系": ["SHFE.RB", "SHFE.HC", "DCE.I", "DCE.J", "DCE.JM"],
    "农产品": ["DCE.M", "DCE.C", "DCE.A", "DCE.P", "DCE.Y",
               "CZCE.SR", "CZCE.CF", "CZCE.OI", "CZCE.RM"],
    "化工":   ["CZCE.TA", "CZCE.MA", "CZCE.FG", "CZCE.SA",
               "DCE.EG", "DCE.PP", "DCE.L", "DCE.V"],
    "有色":   ["SHFE.CU", "SHFE.AL", "SHFE.ZN", "SHFE.NI", "SHFE.SN"],
    "贵金属": ["SHFE.AU", "SHFE.AG"],
    "股指":   ["CFFEX.IF", "CFFEX.IC", "CFFEX.IH"],
    "能源":   ["INE.SC", "INE.NR"],
}
_ALL_SYMBOLS = sorted({s for g in _SYMBOL_GROUPS.values() for s in g})

_MODE_LABELS = {
    "tqsdk": "TqSdk 连续合约",
    "contract": "独立合约（含展期）",
    "product": "品种指数",
}


def _do_load(loader):
    """加载数据后的公共处理。"""
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


def _render_tqsdk_settings():
    """TqSdk 模式的设置表单。"""
    phone = st.text_input("快期账号（手机号）", value=_DEFAULT_PHONE)
    password = st.text_input("快期密码", value=_DEFAULT_PASSWORD, type="password")

    st.caption("独立合约 + 展期模式 — 下载真实合约后识别主力序列，完整支持展期处理")

    st.caption("选择品种（勾选分组）")
    selected = []
    cols = st.columns(3)
    for i, (group_name, syms) in enumerate(_SYMBOL_GROUPS.items()):
        with cols[i % 3]:
            if st.checkbox(group_name, value=True, key=f"tq_grp_{group_name}"):
                selected.extend(syms)

    if not selected:
        selected = _ALL_SYMBOLS

    data_length = st.slider(
        "每品种 K 线数量",
        min_value=500, max_value=10000, value=5000, step=500,
        help="数量越大可回溯时间越长（5000约回溯到2015年），加载也越慢",
    )

    if st.button("从 TqSdk 加载", type="primary"):
        with st.spinner("正在从 TqSdk 获取真实合约数据..."):
            loader = load_tqsdk_cached(
                phone=phone, password=password,
                symbols=tuple(selected),
                data_length=data_length,
            )
            _do_load(loader)
        st.success(f"加载成功！{len(selected)} 个品种，模式: {loader.data_mode}")


def _render_csv_settings():
    """CSV 模式的设置表单。"""
    data_dir = st.text_input("数据目录路径", value=DATA_DIR)
    file_pattern = st.text_input(
        "文件匹配模式", value="*.csv",
        help="如 SHFE.RB.csv 仅加载螺纹钢，*.csv 加载全部",
    )
    _uploaded_files = st.file_uploader(
        "上传 CSV 文件", type=["csv"], accept_multiple_files=True,
        help="格式：1)合约格式(date,symbol,open_interest) 2)品种格式(datetime,position)",
    )
    if st.button("从 CSV 加载", type="primary"):
        with st.spinner("正在加载 CSV 数据..."):
            loader = load_data_cached(data_dir, file_pattern)
            _do_load(loader)
        st.success(f"加载成功！模式: {loader.data_mode}")


def _render_data_overview():
    """渲染数据概览面板。"""
    if not st.session_state.data_loaded or not st.session_state.data_loader:
        return
    loader = st.session_state.data_loader
    summary = loader.get_data_summary()

    st.subheader("数据概览")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("品种数量", summary.get("total_symbols", 0))
    with col_b:
        dr = summary.get("date_range", ("N/A", ""))
        st.metric("数据起始", dr[0])
    with col_c:
        dr = summary.get("date_range", ("", "N/A"))
        st.metric("数据截止", dr[1])

    data_mode = summary.get("data_mode", "unknown")
    st.info(f"数据源: {_MODE_LABELS.get(data_mode, data_mode)}")

    st.subheader("品种列表")
    for product, info in summary.get("products", {}).items():
        st.write(f"**{product}**: {info['contracts']} 个品种/合约")


def render_data_import():
    st.header("数据导入")

    source_mode = st.radio(
        "数据源",
        options=["TqSdk（在线真实数据）", "CSV 文件（本地数据）"],
        horizontal=True,
        index=0,
        help="TqSdk 从天勤量化获取交易所真实合约数据，CSV 从本地文件加载",
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("设置")
        if source_mode.startswith("TqSdk"):
            _render_tqsdk_settings()
        else:
            _render_csv_settings()

    with col2:
        _render_data_overview()
