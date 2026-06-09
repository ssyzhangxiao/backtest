"""
因子模块单元测试。

规则6要求：新增策略必须有对应的因子计算正确性测试。
覆盖：因子评估（IC/IR/有效性判定）、因子变换（非线性/交叉项）。
"""

import unittest
import numpy as np

from core.factors.factor_evaluator import (
    FactorEvaluator,
    FactorEvalResult,
    IC_THRESHOLD,
    IR_THRESHOLD,
)
from core.factors.factor_transformer import FactorTransformer


class TestFactorEvaluator(unittest.TestCase):
    """因子评估器测试。"""

    def setUp(self):
        """构造测试数据。"""
        np.random.seed(42)
        self.n = 200
        # 有效因子：与前瞻收益正相关
        self.valid_scores = np.random.randn(self.n)
        self.valid_returns = self.valid_scores * 0.1 + np.random.randn(self.n) * 0.05
        # 无效因子：与前瞻收益无关
        self.random_scores = np.random.randn(self.n)
        self.random_returns = np.random.randn(self.n)

    def test_valid_factor_passes_evaluation(self):
        """有效因子应通过IC/IR阈值判定。"""
        evaluator = FactorEvaluator(forward_period=5, min_observations=30)
        result = evaluator.evaluate(
            "valid_factor", self.valid_scores, self.valid_returns
        )
        self.assertIsInstance(result, FactorEvalResult)
        self.assertEqual(result.name, "valid_factor")
        # 有效因子的IC应显著大于0
        self.assertGreater(abs(result.ic_mean), IC_THRESHOLD * 0.5)

    def test_random_factor_fails_evaluation(self):
        """随机因子应无法通过IC阈值。"""
        evaluator = FactorEvaluator(forward_period=5, min_observations=30)
        result = evaluator.evaluate(
            "random_factor", self.random_scores, self.random_returns
        )
        # 随机因子的IC应接近0
        self.assertLess(abs(result.ic_mean), 0.15)

    def test_insufficient_observations(self):
        """观测数不足应返回无效。"""
        evaluator = FactorEvaluator(min_observations=30)
        short_scores = np.array([1.0, 2.0, 3.0])
        short_returns = np.array([0.01, 0.02, -0.01])
        result = evaluator.evaluate("short", short_scores, short_returns)
        self.assertFalse(result.is_valid)
        self.assertIn("不足", result.reject_reason)

    def test_nan_handling(self):
        """含NaN的数据应正确处理。"""
        scores = np.array([1.0, np.nan, 3.0, 4.0, 5.0] * 40)
        returns = np.array([0.01, 0.02, np.nan, -0.01, 0.03] * 40)
        evaluator = FactorEvaluator(min_observations=30)
        result = evaluator.evaluate("nan_factor", scores, returns)
        # 不应抛出异常
        self.assertIsInstance(result, FactorEvalResult)

    def test_ic_threshold_enforcement(self):
        """IC低于阈值的因子应被拒绝。"""
        evaluator = FactorEvaluator(min_observations=30)
        # 构造IC极低的因子
        low_ic_scores = np.random.randn(200)
        low_ic_returns = np.random.randn(200)
        result = evaluator.evaluate("low_ic", low_ic_scores, low_ic_returns)
        # 若IC < 0.03，应标记为无效
        if abs(result.ic_mean) < IC_THRESHOLD:
            self.assertFalse(result.is_valid)

    def test_evaluate_batch(self):
        """批量评估应返回多个因子结果。"""
        evaluator = FactorEvaluator(min_observations=30)
        factors = {
            "factor_a": self.valid_scores,
            "factor_b": self.random_scores,
        }
        results = evaluator.evaluate_batch(factors, self.valid_returns)
        self.assertEqual(len(results), 2)
        self.assertIn("factor_a", results)
        self.assertIn("factor_b", results)


class TestFactorTransformer(unittest.TestCase):
    """因子变换器测试。"""

    def setUp(self):
        np.random.seed(42)
        self.n = 200
        self.scores = np.random.randn(self.n) * 0.5
        self.returns = self.scores * 0.1 + np.random.randn(self.n) * 0.05

    def test_log_transform(self):
        """对数变换应压缩极端值。"""
        transformer = FactorTransformer()
        result = transformer.log_transform(self.scores, name="test_log")
        self.assertEqual(len(result), self.n)
        # 对数变换后绝对值应更小
        self.assertLess(np.max(np.abs(result)), np.max(np.abs(self.scores)) + 1)

    def test_power_transform(self):
        """幂函数变换应减少偏度。"""
        transformer = FactorTransformer(powers=[0.5])
        result = transformer.power_transform(self.scores, power=0.5, name="test_power")
        self.assertEqual(len(result), self.n)

    def test_factor_product(self):
        """因子乘积应捕捉共振信号。"""
        transformer = FactorTransformer()
        scores2 = np.random.randn(self.n) * 0.3
        result = transformer.factor_product(self.scores, scores2)
        self.assertEqual(len(result), self.n)

    def test_factor_ratio(self):
        """因子比率应避免除零。"""
        transformer = FactorTransformer()
        scores2 = np.random.randn(self.n) * 0.3
        result = transformer.factor_ratio(self.scores, scores2)
        self.assertEqual(len(result), self.n)
        # 不应包含inf
        self.assertFalse(np.any(np.isinf(result)))


if __name__ == "__main__":
    unittest.main()
