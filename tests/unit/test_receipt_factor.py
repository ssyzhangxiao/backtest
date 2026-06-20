"""仓单变化率因子（receipt_factor）单元测试（规则6）。

复用 ReceiptFetcher 的实现，做一层薄包装测试：
  - 数组版与 Series 版输出一致
  - 输入合法性（None / 空 / 长度不足）
  - 信号方向：仓单下降 → 做多
"""

import unittest

import numpy as np
import pandas as pd

from core.factors.receipt_factor import (
    compute_receipt_factor,
    compute_receipt_factor_signal,
)


class TestReceiptFactorArray(unittest.TestCase):
    """数组版测试。"""

    def test_array_matches_series_implementation(self):
        """数组版输出应与直接调用模块级函数一致。"""
        n = 50
        receipt = pd.Series(np.linspace(100, 50, n))
        from core.data.receipt_fetcher import get_receipt_change_signal
        expected = get_receipt_change_signal(receipt, window=10).to_numpy()
        actual = compute_receipt_factor(receipt.values, window=10)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_increasing_receipt_negative_signal(self):
        """仓单上升 → 信号为负（做空）。"""
        n = 50
        receipt = np.linspace(50, 100, n)
        signal = compute_receipt_factor(receipt, window=10)
        self.assertLess(signal[-1], 0)

    def test_decreasing_receipt_positive_signal(self):
        """仓单下降 → 信号为正（做多）。"""
        n = 50
        receipt = np.linspace(100, 50, n)
        signal = compute_receipt_factor(receipt, window=10)
        self.assertGreater(signal[-1], 0)

    def test_signal_bounded(self):
        """信号应在 [-1, 1]。"""
        np.random.seed(42)
        receipt = 100 + np.cumsum(np.random.randn(100))
        receipt = np.abs(receipt) + 1.0
        signal = compute_receipt_factor(receipt, window=20)
        finite = signal[np.isfinite(signal)]
        self.assertGreaterEqual(finite.min(), -1.0)
        self.assertLessEqual(finite.max(), 1.0)

    def test_empty_array(self):
        """空数组应返回空数组。"""
        signal = compute_receipt_factor(np.array([]), window=10)
        self.assertEqual(len(signal), 0)

    def test_short_array(self):
        """长度 < window 的数组应全为 0。"""
        signal = compute_receipt_factor(np.array([10, 20, 30]), window=10)
        np.testing.assert_array_equal(signal, np.zeros(3))


class TestReceiptFactorSeries(unittest.TestCase):
    """Series 版测试。"""

    def test_series_output(self):
        """Series 版输出索引与输入一致。"""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        receipt = pd.Series(np.linspace(100, 50, 50), index=dates)
        signal = compute_receipt_factor_signal(receipt, window=10)
        self.assertEqual(len(signal), 50)
        self.assertEqual(signal.index.equals(receipt.index), True)
        self.assertEqual(signal.name, "receipt_factor")
        # 仓单下降 → 信号为正
        self.assertGreater(signal.iloc[-1], 0)

    def test_empty_series(self):
        """空 Series 应返回空 Series，name 仍为 'receipt_factor'。"""
        signal = compute_receipt_factor_signal(pd.Series(dtype=float), window=10)
        self.assertEqual(len(signal), 0)
        self.assertEqual(signal.name, "receipt_factor")

    def test_none_series(self):
        """None 应返回空 Series。"""
        signal = compute_receipt_factor_signal(None, window=10)
        self.assertEqual(len(signal), 0)


if __name__ == "__main__":
    unittest.main()
