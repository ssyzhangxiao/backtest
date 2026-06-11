"""
共享 pytest fixture：合成数据 + BacktestConfig + PyBrokerDataSource。

用于回测修复点测试（6 处"重建 BacktestConfig 丢 factor_weights"bug）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ───────────────────────────────────────────────────────────
# 注册自定义 mark
# ───────────────────────────────────────────────────────────
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: 慢测试（>30s），CI 跳过")


# ───────────────────────────────────────────────────────────
# 数据集规模：CI 跑得快 + 5 子策略能开仓
# ───────────────────────────────────────────────────────────
_TEST_N_BARS = 300
_TEST_SYMBOLS = ("RB", "CU", "AU", "I", "M", "P")
_TINY_N_BARS = 80
_TINY_SYMBOLS = ("RB", "CU")


def _make_synth_ohlcv(n: int, seed: int) -> pd.DataFrame:
    """合成一个品种的 OHLCV + OI + 期限结构列。"""
    rng = np.random.default_rng(seed)
    # 随机游走生成 close
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.integers(1000, 5000, n)
    open_interest = rng.integers(50000, 80000, n)
    # 期限结构列（占位即可）
    far_close = close * 1.01
    far_price = close * 1.01
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "symbol": "TEMP",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "open_interest": open_interest,
            "spread": close * 0.001,
            "near_far_ratio": 1.01,
            "near_close": close,
            "far_close": far_close,
            "dominant_close": close,
            "far_price": far_price,
            "far_oi": open_interest * 0.5,
            "is_dominant": True,
            "delivery_exclude": False,
            "gap_weight": 0.5,
            "roll_map": 0.0,
            "near_price": close,
        }
    )


@pytest.fixture(scope="session")
def synth_all_df() -> pd.DataFrame:
    """全品种合成数据（6 品种 × 300 bars）。"""
    frames = []
    for i, sym in enumerate(_TEST_SYMBOLS):
        df = _make_synth_ohlcv(_TEST_N_BARS, seed=42 + i)
        df["symbol"] = sym
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def synth_ds(synth_all_df):
    """PyBrokerDataSource 实例。"""
    from core.engine.pybroker_data_source import PyBrokerDataSource

    return PyBrokerDataSource(synth_all_df)


@pytest.fixture(scope="session")
def synth_lib():
    """StrategyLibrary 实例。"""
    from core.config.strategy_profiles import StrategyLibrary

    return StrategyLibrary()


@pytest.fixture(scope="session")
def synth_config():
    """最小可用 BacktestConfig（含 factor_weights + stop_loss_pct + rebalance_days）。"""
    from core.config import BacktestConfig

    return BacktestConfig(
        initial_cash=1_000_000.0,
        commission_rate=0.0001,
        slippage_rate=0.001,
        # 关键字段：5 子策略权重（防止 0 trade bug）
        factor_weights={
            "trend": 0.2,
            "term_structure": 0.2,
            "mean_reversion": 0.2,
            "vol_breakout": 0.2,
            "composite_resonance": 0.2,
        },
        stop_loss_pct=0.05,
        max_position_pct=0.3,
        max_total_position_pct=0.8,
        rebalance_days=5,
        symbols=list(_TEST_SYMBOLS),
        strategy_names=[
            "trend",
            "term_structure",
            "mean_reversion",
            "vol_breakout",
            "composite_resonance",
        ],
        full_start="2020-01-01",
        full_end=str(
            pd.Timestamp("2020-01-01") + pd.Timedelta(days=_TEST_N_BARS - 1)
        ).split()[0],
        train_start="2020-01-01",
        train_end=str(pd.Timestamp("2020-01-01") + pd.Timedelta(days=200)).split()[0],
        test_start=str(pd.Timestamp("2020-01-01") + pd.Timedelta(days=201)).split()[0],
        test_end=str(
            pd.Timestamp("2020-01-01") + pd.Timedelta(days=_TEST_N_BARS - 1)
        ).split()[0],
        in_sample_end=str(pd.Timestamp("2020-01-01") + pd.Timedelta(days=200)).split()[
            0
        ],
        out_sample_start=str(
            pd.Timestamp("2020-01-01") + pd.Timedelta(days=201)
        ).split()[0],
    )


@pytest.fixture
def small_pspace_vol_breakout() -> dict:
    """vol_breakout 最小参数空间（2×2 加速 CI）。"""
    return {"ma_window": [10, 20], "corr_window": [40, 60]}


# ───────────────────────────────────────────────────────────
# tiny fixture：2 品种 × 80 bars，CI 跑 < 10s
# ───────────────────────────────────────────────────────────
def _make_tiny_ohlcv(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.integers(1000, 5000, n)
    open_interest = rng.integers(50000, 80000, n)
    far_close = close * 1.01
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "symbol": "TEMP",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "open_interest": open_interest,
            "spread": close * 0.001,
            "near_far_ratio": 1.01,
            "near_close": close,
            "far_close": far_close,
            "dominant_close": close,
            "far_price": close * 1.01,
            "far_oi": open_interest * 0.5,
            "is_dominant": True,
            "delivery_exclude": False,
            "gap_weight": 0.5,
            "roll_map": 0.0,
            "near_price": close,
        }
    )


@pytest.fixture(scope="session")
def tiny_all_df() -> pd.DataFrame:
    """tiny 数据集：2 品种 × 80 bars。"""
    frames = []
    for i, sym in enumerate(_TINY_SYMBOLS):
        df = _make_tiny_ohlcv(_TINY_N_BARS, seed=42 + i)
        df["symbol"] = sym
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def tiny_ds(tiny_all_df):
    """tiny PyBrokerDataSource。"""
    from core.engine.pybroker_data_source import PyBrokerDataSource

    return PyBrokerDataSource(tiny_all_df)


@pytest.fixture(scope="session")
def tiny_config():
    """tiny BacktestConfig：2 品种 + 5 子策略权重。"""
    from core.config import BacktestConfig

    return BacktestConfig(
        initial_cash=1_000_000.0,
        commission_rate=0.0001,
        slippage_rate=0.001,
        factor_weights={
            "trend": 0.2,
            "term_structure": 0.2,
            "mean_reversion": 0.2,
            "vol_breakout": 0.2,
            "composite_resonance": 0.2,
        },
        stop_loss_pct=0.05,
        max_position_pct=0.3,
        max_total_position_pct=0.8,
        rebalance_days=5,
        symbols=list(_TINY_SYMBOLS),
        strategy_names=[
            "trend",
            "term_structure",
            "mean_reversion",
            "vol_breakout",
            "composite_resonance",
        ],
        full_start="2020-01-01",
        full_end=str(
            pd.Timestamp("2020-01-01") + pd.Timedelta(days=_TINY_N_BARS - 1)
        ).split()[0],
        train_start="2020-01-01",
        train_end=str(pd.Timestamp("2020-01-01") + pd.Timedelta(days=50)).split()[0],
        test_start=str(pd.Timestamp("2020-01-01") + pd.Timedelta(days=51)).split()[0],
        test_end=str(
            pd.Timestamp("2020-01-01") + pd.Timedelta(days=_TINY_N_BARS - 1)
        ).split()[0],
        in_sample_end=str(pd.Timestamp("2020-01-01") + pd.Timedelta(days=50)).split()[
            0
        ],
        out_sample_start=str(
            pd.Timestamp("2020-01-01") + pd.Timedelta(days=51)
        ).split()[0],
    )
