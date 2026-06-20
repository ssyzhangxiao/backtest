"""基差动量因子（basis_momentum）单元测试（规则6）。"""

import unittest

import numpy as np
import pandas as pd

from core.factors.basis_momentum import (
    compute_basis_momentum,
    compute_basis_momentum_series,
)


class TestBasisMomentumArray(unittest.TestCase):
    """compute_basis_momentum（数组版）测试。"""

    def test_basis_increasing_positive_signal(self):
        """基差走强（远月 - 近月 扩大）→ 信号为正（做多）。"""
        n = 50
        close = np.full(n, 100.0)
        # far_close 从 110 增加到 130（basis 扩大 → 基差动量↑）
        far_close = np.linspace(110, 130, n)
        signal = compute_basis_momentum(close, far_close, basis_window=10)
        # 后期应持续为正
        self.assertGreater(signal[-1], 0)

    def test_basis_decreasing_negative_signal(self):
        """基差走弱（远月 - 近月 收窄）→ 信号为负（做空）。"""
        n = 50
        close = np.full(n, 100.0)
        # far_close 从 130 下降到 110（basis 收窄 → 基差动量↓）
        far_close = np.linspace(130, 110, n)
        signal = compute_basis_momentum(close, far_close, basis_window=10)
        # 后期应持续为负
        self.assertLess(signal[-1], 0)

    def test_signal_bounded_in_range(self):
        """信号应在 [-1, 1] 区间内。"""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n)) * 0.1
        far_close = close + np.linspace(0, 20, n) + np.random.randn(n) * 0.5
        signal = compute_basis_momentum(close, far_close, basis_window=20)
        finite = signal[np.isfinite(signal)]
        self.assertGreaterEqual(finite.min(), -1.0)
        self.assertLessEqual(finite.max(), 1.0)

    def test_warmup_zero(self):
        """前 basis_window 个 bar 应为 0。"""
        n = 50
        close = np.full(n, 100.0)
        far_close = np.linspace(110, 130, n)
        window = 20
        signal = compute_basis_momentum(close, far_close, basis_window=window)
        np.testing.assert_array_equal(signal[:window], np.zeros(window))

    def test_far_close_all_nan_returns_zero(self):
        """远月全 NaN → 返回零信号（品种无次主力）。"""
        close = np.full(50, 100.0)
        far_close = np.full(50, np.nan)
        signal = compute_basis_momentum(close, far_close, basis_window=10)
        np.testing.assert_array_equal(signal, np.zeros(50))

    def test_none_far_close_returns_zero(self):
        """far_close=None → 返回零信号。"""
        close = np.full(50, 100.0)
        signal = compute_basis_momentum(close, None, basis_window=10)
        np.testing.assert_array_equal(signal, np.zeros(50))

    def test_empty_array(self):
        """空数组应返回空数组，不抛异常。"""
        signal = compute_basis_momentum(np.array([]), np.array([]), basis_window=10)
        self.assertEqual(len(signal), 0)

    def test_short_array_returns_zero(self):
        """长度 < basis_window 的数组应全为 0。"""
        close = np.full(5, 100.0)
        far_close = np.full(5, 110.0)
        signal = compute_basis_momentum(close, far_close, basis_window=10)
        np.testing.assert_array_equal(signal, np.zeros(5))

    def test_zero_close_handled(self):
        """close=0 应避免除零，使用兜底（返回有限值或 0）。"""
        n = 50
        close = np.zeros(n)  # 全 0（异常情况）
        far_close = np.linspace(110, 130, n)
        # 不应抛异常，结果有限
        signal = compute_basis_momentum(close, far_close, basis_window=10)
        self.assertEqual(len(signal), n)
        # 不应有 inf
        self.assertTrue(np.all(np.isfinite(signal[10:])))


class TestBasisMomentumSeries(unittest.TestCase):
    """compute_basis_momentum_series（Series 版）测试。"""

    def test_series_alignment(self):
        """两个 Series 应按索引对齐。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        close = pd.Series(np.full(30, 100.0), index=dates)
        far_close = pd.Series(np.linspace(110, 130, 30), index=dates)
        signal = compute_basis_momentum_series(close, far_close, basis_window=10)
        self.assertEqual(len(signal), 30)
        self.assertEqual(signal.index.equals(close.index), True)
        self.assertEqual(signal.name, "basis_momentum")
        # warmup 后应有非零信号
        self.assertGreater(signal.iloc[-1], 0)

    def test_far_close_missing_dates(self):
        """far_close 缺少日期应填充 NaN。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        close = pd.Series(np.full(30, 100.0), index=dates)
        # far_close 只覆盖前 15 天
        far_close = pd.Series(
            np.linspace(110, 115, 15),
            index=dates[:15],
        )
        signal = compute_basis_momentum_series(close, far_close, basis_window=10)
        # 后期（无 far_close 数据）应全为 0
        self.assertEqual(len(signal), 30)

    def test_empty_close_returns_empty(self):
        """空 close 应返回空 Series。"""
        signal = compute_basis_momentum_series(
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            basis_window=10,
        )
        self.assertEqual(len(signal), 0)

    def test_none_far_close_returns_zero_signal(self):
        """far_close=None 应返回全 0 信号。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        close = pd.Series(np.full(30, 100.0), index=dates)
        signal = compute_basis_momentum_series(close, None, basis_window=10)
        np.testing.assert_array_equal(signal.values, np.zeros(30))


if __name__ == "__main__":
    unittest.main()
