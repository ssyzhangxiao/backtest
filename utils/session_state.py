import streamlit as st
import pandas as pd
import pybroker
from typing import Optional

from config import _PYBROKER_COLUMNS

_PYBROKER_COLUMNS_REGISTERED = False


@st.cache_data(show_spinner=False)
def load_data_cached(data_dir: str, file_pattern: str):
    """CSV 模式：从本地文件加载数据。"""
    from core.data_loader import DataLoader
    loader = DataLoader(data_dir)
    loader.load_csv_files(file_pattern=file_pattern)
    loader.identify_dominant_contracts()
    loader.build_continuous_series()
    return loader


@st.cache_data(show_spinner=False)
def load_tqsdk_cached(
    phone: str, password: str,
    symbols: Optional[tuple] = None,
    data_length: int = 2000,
):
    """TqSdk 模式：从 TqSdk 获取独立合约 + 展期数据。"""
    from core.data_loader_tqsdk import TqSdkDataSource
    sym_list = list(symbols) if symbols else None
    loader = TqSdkDataSource(
        phone=phone, password=password,
        symbols=sym_list, data_length=data_length,
    )
    loader.load_from_tqsdk()
    loader.identify_dominant_contracts()
    loader.build_continuous_series()
    return loader


@st.cache_data(show_spinner=False)
def compute_env_cached(pybroker_df: pd.DataFrame):
    from core.environment import EnvironmentAdapter
    env_adapter = EnvironmentAdapter()
    return env_adapter.compute_for_pybroker(pybroker_df)


def register_pybroker_columns():
    global _PYBROKER_COLUMNS_REGISTERED
    if not _PYBROKER_COLUMNS_REGISTERED:
        pybroker.register_columns(*_PYBROKER_COLUMNS)
        _PYBROKER_COLUMNS_REGISTERED = True


def init_session_state():
    defaults = {
        "data_loaded": False,
        "backtest_run": False,
        "backtest_result": None,
        "pybroker_df": None,
        "data_loader": None,
        "env_df": None,
        "initial_cash": 1_000_000,
        "applied_optimized_params": {},
        "optimized_strategy": None,
    }
    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val