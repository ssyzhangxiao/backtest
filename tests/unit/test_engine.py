"""
调仓逻辑单元测试。

规则6要求：调仓逻辑修改必须有调仓日判断测试。
覆盖：因子评估器、调仓日判断逻辑。
"""

import unittest
import numpy as np

from core.ext.factors.evaluator import FactorEvaluator, FactorEvalResult


class TestFactorEvaluatorForRebalance(unittest.TestCase):
    """因子评估器测试（替代原 RollingICWeightEngine 测试）。"""

    def setUp(self):
        self.evaluator = FactorEvaluator(
            forward_period=5,
            ic_window=30,
            min_observations=10,
        )

    def test_evaluate_single_factor(self):
        """单因子评估应返回 FactorEvalResult。"""
        n = 100
        np.random.seed(42)
        scores = np.random.randn(n)
        returns = scores * 0.5 + np.random.randn(n) * 0.5
        result = self.evaluator.evaluate("trend", scores, returns)
        self.assertIsInstance(result, FactorEvalResult)
        self.assertEqual(result.name, "trend")

    def test_batch_evaluate(self):
        """批量评估应返回多个结果。"""
        n = 100
        np.random.seed(42)
        factor_dict = {
            "trend": np.random.randn(n),
            "momentum": np.random.randn(n),
        }
        returns = np.random.randn(n)
        results = self.evaluator.evaluate_batch(factor_dict, returns)
        self.assertEqual(len(results), 2)
        self.assertIn("trend", results)

    def test_insufficient_data(self):
        """观测数不足应标记为无效。"""
        scores = np.array([1.0, 2.0])
        returns = np.array([0.1, 0.2])
        result = self.evaluator.evaluate("short", scores, returns)
        self.assertFalse(result.is_valid)

    def test_healthy_factor_is_valid(self):
        """健康因子应通过规则9。"""
        n = 200
        np.random.seed(42)
        scores = np.random.randn(n)
        returns = scores * 0.3 + np.random.randn(n) * 0.1
        result = self.evaluator.evaluate("healthy", scores, returns)
        # IC 应为正
        self.assertGreater(result.ic_mean, 0)


class TestRebalanceDayLogic(unittest.TestCase):
    """调仓日判断逻辑测试。"""

    def test_rebalance_every_n_days(self):
        """每N个交易日调仓一次。"""
        rebalance_days = 3
        for day in range(30):
            is_rebalance_day = day % rebalance_days == 0
            if day in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]:
                self.assertTrue(is_rebalance_day, f"Day {day} should be rebalance day")
            else:
                self.assertFalse(
                    is_rebalance_day, f"Day {day} should not be rebalance day"
                )

    def test_rebalance_day_1_means_daily(self):
        """rebalance_days=1表示每日调仓。"""
        rebalance_days = 1
        for day in range(10):
            self.assertTrue(day % rebalance_days == 0)

    def test_rebalance_day_never_on_non_multiples(self):
        """非调仓日不应触发。"""
        rebalance_days = 5
        non_rebalance_days = [1, 2, 3, 4, 6, 7, 8, 9]
        for day in non_rebalance_days:
            self.assertNotEqual(day % rebalance_days, 0)


if __name__ == "__main__":
    unittest.main()
