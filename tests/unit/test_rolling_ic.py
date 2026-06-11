"""因子评估器单元测试（替代原 test_rolling_ic.py 和 test_factor_decay.py）。

原 core.engine.rolling_ic 和 core.engine.factor_decay 已废弃，
功能由 core.ext.factors.evaluator.FactorEvaluator 统一提供。
"""

import numpy as np
import pytest

from core.ext.factors.evaluator import FactorEvaluator, FactorEvalResult


class TestFactorEvaluatorInit:
    """初始化测试。"""

    def test_default_init(self):
        evaluator = FactorEvaluator()
        assert evaluator.forward_period == 5
        assert evaluator.ic_window == 60

    def test_custom_config(self):
        evaluator = FactorEvaluator(forward_period=10, ic_window=30, min_observations=10)
        assert evaluator.forward_period == 10
        assert evaluator.ic_window == 30
        assert evaluator.min_observations == 10


class TestICComputation:
    """IC计算测试。"""

    def test_perfect_positive_correlation(self):
        evaluator = FactorEvaluator(min_observations=10)
        n = 100
        scores = np.arange(n, dtype=float)
        returns = np.arange(n, dtype=float) * 0.1
        result = evaluator.evaluate("trend", scores, returns)
        assert result.ic_mean > 0.9
        assert result.is_valid or result.ic_mean > 0.5

    def test_no_correlation(self):
        evaluator = FactorEvaluator(min_observations=10)
        np.random.seed(42)
        n = 200
        scores = np.random.randn(n)
        returns = np.random.randn(n)
        result = evaluator.evaluate("random", scores, returns)
        assert abs(result.ic_mean) < 0.3

    def test_constant_factor_yields_zero_ic(self):
        evaluator = FactorEvaluator(min_observations=10)
        n = 100
        scores = np.full(n, 0.5)
        returns = np.arange(n, dtype=float) * 0.1
        result = evaluator.evaluate("const", scores, returns)
        # 常数因子 IC 应为 0 或接近 0
        assert abs(result.ic_mean) < 0.1

    def test_insufficient_observations(self):
        evaluator = FactorEvaluator(min_observations=50)
        scores = np.array([1.0, 2.0, 3.0])
        returns = np.array([0.1, 0.2, 0.3])
        result = evaluator.evaluate("short", scores, returns)
        assert not result.is_valid
        assert "不足" in result.reject_reason

    def test_nan_handling(self):
        evaluator = FactorEvaluator(min_observations=10)
        n = 100
        scores = np.random.randn(n)
        returns = np.random.randn(n)
        scores[::10] = np.nan  # 每10个插入一个NaN
        result = evaluator.evaluate("with_nan", scores, returns)
        # 应能正常处理NaN
        assert isinstance(result, FactorEvalResult)


class TestBatchEvaluation:
    """批量评估测试。"""

    def test_evaluate_batch(self):
        evaluator = FactorEvaluator(min_observations=10)
        n = 100
        factor_dict = {
            "trend": np.random.randn(n),
            "momentum": np.random.randn(n),
        }
        returns = np.random.randn(n)
        results = evaluator.evaluate_batch(factor_dict, returns)
        assert len(results) == 2
        assert "trend" in results
        assert "momentum" in results
        assert isinstance(results["trend"], FactorEvalResult)

    def test_evaluate_batch_with_correlated_factors(self):
        evaluator = FactorEvaluator(min_observations=10)
        n = 200
        base = np.random.randn(n)
        factor_dict = {
            "factor_a": base + np.random.randn(n) * 0.1,
            "factor_b": base + np.random.randn(n) * 0.1,
        }
        returns = base * 0.5 + np.random.randn(n) * 0.5
        results = evaluator.evaluate_batch(factor_dict, returns)
        # 两个因子都应与收益有正相关
        assert results["factor_a"].ic_mean > 0
        assert results["factor_b"].ic_mean > 0


class TestDecayDetection:
    """衰减检测测试（替代原 test_factor_decay.py）。"""

    def test_healthy_factor(self):
        evaluator = FactorEvaluator(min_observations=10)
        n = 200
        scores = np.random.randn(n) * 0.1 + 0.5
        returns = scores + np.random.randn(n) * 0.3
        result = evaluator.evaluate("healthy", scores, returns)
        # IC 应为正且较大
        assert result.ic_mean > 0

    def test_dead_factor(self):
        evaluator = FactorEvaluator(min_observations=10)
        n = 200
        scores = np.random.randn(n) * 0.005
        returns = np.random.randn(n)
        result = evaluator.evaluate("dead", scores, returns)
        # IC 应接近 0
        assert abs(result.ic_mean) < 0.3

    def test_result_summary(self):
        evaluator = FactorEvaluator(min_observations=10)
        n = 100
        scores = np.random.randn(n)
        returns = np.random.randn(n)
        result = evaluator.evaluate("test", scores, returns)
        summary = result.summary()
        assert "test" in summary
        assert "IC=" in summary
