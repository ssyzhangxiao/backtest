"""
UnifiedFactorPool + SignalAbstractionLayer 单元测试。
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.execution.factor_pool import (
    ALL_SIGNAL_NAMES,
    CTA_SIGNAL_NAMES,
    DEFAULT_FACTOR_NAMES,
    UnifiedFactorPool,
)
from core.execution.signal_abstraction import (
    DEFAULT_CTA_WEIGHTS,
    SignalAbstractionLayer,
    SignalMode,
)


# ── fixtures ──


@pytest.fixture
def mini_ohlcv() -> pd.DataFrame:
    """50 bar 的迷你 OHLCV 数据。"""
    np.random.seed(42)
    n = 50
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": close * 0.99,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.randint(10000, 50000, n),
    })


@pytest.fixture
def signal_layer(mini_ohlcv: pd.DataFrame) -> SignalAbstractionLayer:
    """预加载了 mini_ohlcv 的 SignalAbstractionLayer。"""
    pool = UnifiedFactorPool()
    layer = SignalAbstractionLayer(pool, default_mode="cross_sectional", cta_weight=0.5)
    return layer


# ══════════════════════════════════════════════════════════════
# UnifiedFactorPool
# ══════════════════════════════════════════════════════════════


class TestUnifiedFactorPoolInit:
    """UnifiedFactorPool 初始化测试。"""

    def test_init(self):
        pool = UnifiedFactorPool()
        assert pool._cache == {}
        assert pool._cta_wrapper is None

    def test_all_signal_names(self):
        assert len(ALL_SIGNAL_NAMES) == len(DEFAULT_FACTOR_NAMES) + len(CTA_SIGNAL_NAMES)
        assert ALL_SIGNAL_NAMES[:5] == DEFAULT_FACTOR_NAMES
        for cta_name in CTA_SIGNAL_NAMES:
            assert cta_name in ALL_SIGNAL_NAMES

    def test_cta_signal_names(self):
        assert "carry" in CTA_SIGNAL_NAMES
        assert "vol_mean_reversion" in CTA_SIGNAL_NAMES
        assert "donchian_breakout" in CTA_SIGNAL_NAMES
        assert "momentum_ma" in CTA_SIGNAL_NAMES
        assert "tsi_garch" in CTA_SIGNAL_NAMES
        assert "pair_trading" in CTA_SIGNAL_NAMES
        assert len(CTA_SIGNAL_NAMES) == 6

    def test_compute_all_returns_dataframe(self, mini_ohlcv: pd.DataFrame):
        pool = UnifiedFactorPool()
        result = pool.compute_all(mini_ohlcv, "TEST")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mini_ohlcv)
        for name in ALL_SIGNAL_NAMES:
            assert name in result.columns

    def test_compute_all_caches_result(self, mini_ohlcv: pd.DataFrame):
        pool = UnifiedFactorPool()
        result1 = pool.compute_all(mini_ohlcv, "TEST")
        result2 = pool.compute_all(mini_ohlcv, "TEST")
        assert result1 is result2  # 同一引用（缓存）

    def test_compute_all_different_symbols(self, mini_ohlcv: pd.DataFrame):
        pool = UnifiedFactorPool()
        r1 = pool.compute_all(mini_ohlcv, "SYM_A")
        r2 = pool.compute_all(mini_ohlcv, "SYM_B")
        assert r1 is not r2

    def test_clear_cache(self, mini_ohlcv: pd.DataFrame):
        pool = UnifiedFactorPool()
        pool.compute_all(mini_ohlcv, "TEST")
        assert "TEST" in pool._cache
        pool.clear_cache("TEST")
        assert "TEST" not in pool._cache

    def test_clear_cache_all(self, mini_ohlcv: pd.DataFrame):
        pool = UnifiedFactorPool()
        pool.compute_all(mini_ohlcv, "A")
        pool.compute_all(mini_ohlcv, "B")
        pool.clear_cache()
        assert pool._cache == {}

    def test_compute_signals_for_bar(self, mini_ohlcv: pd.DataFrame):
        pool = UnifiedFactorPool()
        signals = pool.compute_signals_for_bar(mini_ohlcv, "TEST", bar_idx=-1)
        assert isinstance(signals, dict)
        for name in ALL_SIGNAL_NAMES:
            assert name in signals
        # 信号值应为 float 且在 [-1, 1] 范围内或接近
        for v in signals.values():
            assert isinstance(v, float)

    def test_compute_signals_for_bar_out_of_range(self, mini_ohlcv: pd.DataFrame):
        pool = UnifiedFactorPool()
        signals = pool.compute_signals_for_bar(mini_ohlcv, "TEST", bar_idx=999)
        assert all(v == 0.0 for v in signals.values())


class TestUnifiedFactorPoolShortData:
    """短数据边缘情况测试。"""

    def test_too_short_data(self):
        pool = UnifiedFactorPool()
        short = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "volume": [1000] * 5,
        })
        result = pool.compute_all(short, "TEST")
        assert isinstance(result, pd.DataFrame)

    def test_missing_columns(self):
        pool = UnifiedFactorPool()
        bad_df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(Exception):
            pool.compute_all(bad_df, "TEST")


# ══════════════════════════════════════════════════════════════
# SignalAbstractionLayer
# ══════════════════════════════════════════════════════════════


class TestSignalAbstractionLayerInit:
    """初始化测试。"""

    def test_default_params(self):
        pool = UnifiedFactorPool()
        layer = SignalAbstractionLayer(pool)
        assert layer.mode == "cross_sectional"
        assert layer.cta_weight == 0.5

    def test_custom_params(self):
        pool = UnifiedFactorPool()
        layer = SignalAbstractionLayer(pool, default_mode="hybrid", cta_weight=0.3)
        assert layer.mode == "hybrid"
        assert layer.cta_weight == 0.3


class TestSignalMode:
    """SignalMode 枚举测试。"""

    def test_values(self):
        assert SignalMode.CROSS_SECTIONAL.value == "cross_sectional"
        assert SignalMode.CTA.value == "cta"
        assert SignalMode.HYBRID.value == "hybrid"

    def test_all_modes(self):
        modes = set(m.value for m in SignalMode)
        assert modes == {"cross_sectional", "cta", "hybrid"}


class TestGetCrossSectionalSignals:
    """横截面信号测试。"""

    def test_returns_five_signals(self, signal_layer, mini_ohlcv):
        signals = signal_layer.get_cross_sectional_signals("TEST", mini_ohlcv, -1)
        assert isinstance(signals, dict)
        assert len(signals) == 5
        for name in DEFAULT_FACTOR_NAMES:
            assert name in signals

    def test_signals_are_clipped(self, signal_layer, mini_ohlcv):
        signals = signal_layer.get_cross_sectional_signals("TEST", mini_ohlcv, -1)
        for v in signals.values():
            assert -1.0 <= v <= 1.0

    def test_early_bar_handling(self, signal_layer, mini_ohlcv):
        """前 30 bar 数据不足应仍返回有效值。"""
        signals = signal_layer.get_cross_sectional_signals("TEST", mini_ohlcv, 20)
        assert isinstance(signals, dict)
        # 早期信号可能为 0（数据不足，无有效信号）
        for name in DEFAULT_FACTOR_NAMES:
            assert name in signals


class TestGetCTACompositeSignal:
    """CTA 复合信号测试。"""

    def test_returns_single_float(self, signal_layer, mini_ohlcv):
        sig = signal_layer.get_cta_composite_signal("TEST", mini_ohlcv, -1)
        assert isinstance(sig, float)
        assert -1.0 <= sig <= 1.0

    def test_custom_weights(self, signal_layer, mini_ohlcv):
        custom = {"carry": 1.0}
        sig = signal_layer.get_cta_composite_signal(
            "TEST", mini_ohlcv, -1, weights=custom,
        )
        assert isinstance(sig, float)

    def test_default_weights_match(self, signal_layer, mini_ohlcv):
        """默认权重应与模块级常量一致。"""
        sig_default = signal_layer.get_cta_composite_signal("TEST", mini_ohlcv, -1)
        sig_explicit = signal_layer.get_cta_composite_signal(
            "TEST", mini_ohlcv, -1, weights=DEFAULT_CTA_WEIGHTS,
        )
        assert abs(sig_default - sig_explicit) < 1e-6


class TestGetHybridSignal:
    """混合信号测试。"""

    def test_default_cta_weight(self, signal_layer, mini_ohlcv):
        sig = signal_layer.get_hybrid_signal("TEST", mini_ohlcv, -1, cross_section_z=0.5)
        assert isinstance(sig, float)
        assert -1.0 <= sig <= 1.0

    def test_custom_cta_weight(self, signal_layer, mini_ohlcv):
        sig = signal_layer.get_hybrid_signal(
            "TEST", mini_ohlcv, -1, cross_section_z=0.5, cta_weight=0.2,
        )
        assert isinstance(sig, float)

    def test_pure_cross_section(self, signal_layer, mini_ohlcv):
        sig = signal_layer.get_hybrid_signal(
            "TEST", mini_ohlcv, -1, cross_section_z=0.5, cta_weight=0.0,
        )
        assert abs(sig - 0.5) < 1e-6

    def test_pure_cta(self, signal_layer, mini_ohlcv):
        sig = signal_layer.get_hybrid_signal(
            "TEST", mini_ohlcv, -1, cross_section_z=0.0, cta_weight=1.0,
        )
        assert isinstance(sig, float)

    def test_instance_cta_weight_used(self, mini_ohlcv):
        pool = UnifiedFactorPool()
        layer = SignalAbstractionLayer(pool, cta_weight=0.3)
        # 不传 cta_weight 应使用实例的 0.3
        sig = layer.get_hybrid_signal("TEST", mini_ohlcv, -1, cross_section_z=0.5)
        assert isinstance(sig, float)

    def test_clipping(self, signal_layer, mini_ohlcv):
        """极端值应被 clip。"""
        sig = signal_layer.get_hybrid_signal(
            "TEST", mini_ohlcv, -1, cross_section_z=100.0, cta_weight=0.0,
        )
        assert sig == 1.0
        sig = signal_layer.get_hybrid_signal(
            "TEST", mini_ohlcv, -1, cross_section_z=-100.0, cta_weight=0.0,
        )
        assert sig == -1.0
