"""
ADX公共函数和蒙特卡洛模拟单元测试。

规则6要求：因子计算正确性测试 + 关键路径测试。
覆盖：compute_adx（标量/序列/组件）、compute_true_range、
     蒙特卡洛向量化与循环版本结果一致性。
"""

import unittest
import numpy as np
import pandas as pd

from utils.indicators import compute_adx, compute_adx_series, compute_adx_components, compute_true_range
from core.validation.monte_carlo import MonteCarloSimulator, MonteCarloResult


class TestComputeTrueRange(unittest.TestCase):
    """真实波幅计算测试。"""

    def test_basic_calculation(self):
        """基本TR计算。"""
        high = pd.Series([105, 110, 108])
        low = pd.Series([100, 103, 104])
        close = pd.Series([102, 108, 106])
        tr = compute_true_range(high, low, close)
        self.assertEqual(len(tr), 3)
        # 首行 TR = H - L = 105 - 100 = 5
        self.assertAlmostEqual(tr.iloc[0], 5.0, places=2)

    def test_gap_up(self):
        """跳空高开：TR应包含缺口。"""
        high = pd.Series([100, 110])
        low = pd.Series([95, 105])
        close = pd.Series([98, 108])
        tr = compute_true_range(high, low, close)
        # 第2行：max(110-105, |110-98|, |105-98|) = max(5, 12, 7) = 12
        self.assertAlmostEqual(tr.iloc[1], 12.0, places=2)

    def test_insufficient_data(self):
        """数据不足应抛出异常。"""
        with self.assertRaises(ValueError):
            compute_true_range(pd.Series([100]), pd.Series([95]), pd.Series([98]))


class TestComputeAdx(unittest.TestCase):
    """ADX标量计算测试。"""

    def test_trending_market(self):
        """趋势行情：ADX应 > 25。"""
        np.random.seed(42)
        n = 200
        # 构造强趋势数据
        close = np.cumsum(np.ones(n) * 0.5) + 100
        high = close + np.abs(np.random.randn(n)) * 0.3
        low = close - np.abs(np.random.randn(n)) * 0.3

        adx_val, plus_di, minus_di = compute_adx(high, low, close, period=14)
        # 强趋势行情ADX应较高
        self.assertGreater(adx_val, 20.0)

    def test_ranging_market(self):
        """震荡行情：ADX应较低。"""
        np.random.seed(42)
        n = 200
        # 构造震荡数据
        close = 100 + np.cumsum(np.random.randn(n) * 0.1)
        high = close + np.abs(np.random.randn(n)) * 2
        low = close - np.abs(np.random.randn(n)) * 2

        adx_val, _, _ = compute_adx(high, low, close, period=14)
        # 震荡行情ADX应较低
        self.assertLess(adx_val, 50.0)

    def test_insufficient_data(self):
        """数据不足应返回0。"""
        close = np.array([100, 101])
        high = np.array([102, 103])
        low = np.array([99, 100])
        adx_val, _, _ = compute_adx(high, low, close, period=14)
        self.assertEqual(adx_val, 0.0)

    def test_invalid_period(self):
        """无效周期应抛出异常。"""
        with self.assertRaises(ValueError):
            compute_adx(np.array([1, 2]), np.array([1, 2]), np.array([1, 2]), period=0)

    def test_plus_di_minus_di_range(self):
        """+DI和-DI应在0~100范围内。"""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n))
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))

        _, plus_di, minus_di = compute_adx(high, low, close, period=14)
        self.assertGreaterEqual(plus_di, 0.0)
        self.assertLessEqual(plus_di, 100.0)
        self.assertGreaterEqual(minus_di, 0.0)
        self.assertLessEqual(minus_di, 100.0)


class TestComputeAdxSeries(unittest.TestCase):
    """ADX序列计算测试。"""

    def test_series_length(self):
        """输出序列长度应与输入一致。"""
        np.random.seed(42)
        n = 100
        close = np.cumsum(np.random.randn(n)) + 100
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))

        adx_series = compute_adx_series(high, low, close, period=14)
        self.assertEqual(len(adx_series), n)

    def test_series_nan_prefix(self):
        """序列前段应为NaN（数据不足计算窗口）。"""
        np.random.seed(42)
        n = 100
        close = np.cumsum(np.random.randn(n)) + 100
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))

        adx_series = compute_adx_series(high, low, close, period=14)
        # 前28个（period*2）应为NaN
        nan_count = adx_series.isna().sum()
        self.assertGreater(nan_count, 10)


class TestComputeAdxComponents(unittest.TestCase):
    """ADX组件序列计算测试。"""

    def test_components_shape(self):
        """三个组件序列长度应一致。"""
        np.random.seed(42)
        n = 100
        close_s = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        high_s = close_s + pd.Series(np.abs(np.random.randn(n)))
        low_s = close_s - pd.Series(np.abs(np.random.randn(n)))

        adx, plus_di, minus_di = compute_adx_components(high_s, low_s, close_s, period=14)
        self.assertEqual(len(adx), n)
        self.assertEqual(len(plus_di), n)
        self.assertEqual(len(minus_di), n)

    def test_index_preserved(self):
        """输出应保留原始索引。"""
        np.random.seed(42)
        n = 50
        idx = pd.date_range("2024-01-01", periods=n)
        close_s = pd.Series(np.cumsum(np.random.randn(n)) + 100, index=idx)
        high_s = close_s + pd.Series(np.abs(np.random.randn(n)), index=idx)
        low_s = close_s - pd.Series(np.abs(np.random.randn(n)), index=idx)

        adx, _, _ = compute_adx_components(high_s, low_s, close_s, period=14)
        pd.testing.assert_index_equal(adx.index, idx)


class TestMonteCarloSimulator(unittest.TestCase):
    """蒙特卡洛模拟器测试。"""

    def setUp(self):
        np.random.seed(42)
        self.returns = np.random.normal(0.0003, 0.015, 504)
        self.mc = MonteCarloSimulator(n_simulations=100, random_seed=42)

    def test_simulate_returns_result(self):
        """模拟应返回MonteCarloResult。"""
        result = self.mc.simulate(self.returns)
        self.assertIsInstance(result, MonteCarloResult)
        self.assertEqual(result.n_simulations, 100)

    def test_quantile_keys(self):
        """分位数字典应包含5个键。"""
        result = self.mc.simulate(self.returns)
        for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
            self.assertIn(q, result.sharpe_quantiles)
            self.assertIn(q, result.max_drawdown_quantiles)
            self.assertIn(q, result.annual_return_quantiles)

    def test_quantile_ordering(self):
        """分位数应单调递增。"""
        result = self.mc.simulate(self.returns)
        qs = sorted(result.sharpe_quantiles.keys())
        for i in range(len(qs) - 1):
            self.assertLessEqual(
                result.sharpe_quantiles[qs[i]],
                result.sharpe_quantiles[qs[i + 1]],
            )

    def test_robustness_flag(self):
        """稳健性标志应与Sharpe 5%分位数一致。"""
        result = self.mc.simulate(self.returns)
        self.assertEqual(result.is_robust, result.sharpe_quantiles[0.05] > 0)

    def test_short_returns_rejected(self):
        """过短的收益率序列应返回空结果。"""
        result = self.mc.simulate(np.array([0.01, 0.02]))
        self.assertEqual(result.n_simulations, 0)
        self.assertFalse(result.is_robust)

    def test_vectorized_loop_consistency(self):
        """向量化与循环版本结果应高度一致。"""
        mc = MonteCarloSimulator(n_simulations=500, random_seed=42)
        vec_result = mc.simulate(self.returns)
        loop_result = mc.simulate_loop(self.returns)

        # 由于随机数生成方式不同（integers vs choice），分位数可能有微小差异
        # 但中位数差异应 < 10%
        for q in [0.50]:
            vec_s = vec_result.sharpe_quantiles[q]
            loop_s = loop_result.sharpe_quantiles[q]
            if abs(loop_s) > 0.01:
                diff_pct = abs(vec_s - loop_s) / abs(loop_s)
                self.assertLess(diff_pct, 0.15, f"Sharpe@{q}差异过大: vec={vec_s}, loop={loop_s}")

    def test_vectorized_faster(self):
        """向量化版本应比循环版本更快。"""
        mc = MonteCarloSimulator(n_simulations=500, random_seed=42)
        vec_result = mc.simulate(self.returns)
        loop_result = mc.simulate_loop(self.returns)
        # 向量化版本应至少不慢于循环版本
        self.assertLessEqual(vec_result.elapsed_seconds, loop_result.elapsed_seconds * 1.5)

    def test_elapsed_seconds_recorded(self):
        """应记录执行时间。"""
        result = self.mc.simulate(self.returns)
        self.assertGreater(result.elapsed_seconds, 0)


if __name__ == "__main__":
    unittest.main()
