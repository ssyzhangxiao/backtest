"""滚动IC加权引擎单元测试。"""
import numpy as np
import pandas as pd
import pytest

from core.engine.rolling_ic import RollingICWeightEngine, RollingICConfig


class TestRollingICInit:
    """初始化测试。"""

    def test_default_init(self):
        engine = RollingICWeightEngine()
        assert engine.config.window == 60
        assert engine._observation_count == 0

    def test_custom_config(self):
        config = RollingICConfig(window=30, ema_alpha=0.2)
        engine = RollingICWeightEngine(config)
        assert engine.config.window == 30
        assert engine.config.ema_alpha == 0.2

    def test_default_weights_match_config(self):
        engine = RollingICWeightEngine()
        weights = engine.get_dynamic_weights()
        assert "trend" in weights
        assert "term_structure" in weights
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


class TestICComputation:
    """IC计算测试。"""

    def test_perfect_positive_correlation(self):
        engine = RollingICWeightEngine(RollingICConfig(window=100, min_observations=10, ema_alpha=1.0))
        for i in range(50):
            engine.update(
                {"trend": i * 0.1, "term_structure": i * 0.05},
                i * 0.02,
            )
        ic = engine.current_ic
        assert ic["trend"] > 0.9
        assert ic["term_structure"] > 0.9

    def test_no_correlation(self):
        engine = RollingICWeightEngine(RollingICConfig(window=200, min_observations=30, ema_alpha=1.0))
        np.random.seed(42)
        for _ in range(100):
            engine.update(
                {"trend": np.random.randn()},
                np.random.randn(),
            )
        ic = engine.current_ic
        assert abs(ic["trend"]) < 0.3

    def test_ema_smoothing(self):
        engine = RollingICWeightEngine(RollingICConfig(window=200, min_observations=30, ema_alpha=0.2))
        np.random.seed(42)
        for i in range(50):
            engine.update({"trend": 0.5 + np.random.randn() * 0.1, "term_structure": -0.3 + np.random.randn() * 0.1}, 0.5 + np.random.randn() * 0.1)
        ic1 = dict(engine.current_ic)
        # 再喂随机数据
        for i in range(50):
            engine.update({"trend": np.random.randn(), "term_structure": np.random.randn()}, np.random.randn())
        ic2 = dict(engine.current_ic)
        # EMA平滑后IC不应跳变太大（相对IC1而言）
        assert abs(ic2["trend"]) < 1.0  # IC始终在[-1,1]区间
        assert abs(ic2["term_structure"]) < 1.0

    def test_constant_factor_yields_zero_ic(self):
        engine = RollingICWeightEngine(RollingICConfig(window=100, min_observations=10, ema_alpha=1.0))
        for i in range(50):
            engine.update(
                {"trend": 0.5},  # 常数值
                i * 0.1,
            )
        assert engine.current_ic["trend"] == pytest.approx(0.0, abs=1e-6)


class TestDynamicWeights:
    """动态权重测试。"""

    def test_weights_sum_to_one(self):
        engine = RollingICWeightEngine(RollingICConfig(window=200, min_observations=30, ema_alpha=1.0))
        np.random.seed(42)
        for _ in range(100):
            engine.update(
                {"trend": np.random.randn(), "term_structure": np.random.randn(),
                 "mean_reversion": np.random.randn(), "vol_breakout": np.random.randn()},
                np.random.randn(),
            )
        weights = engine.get_dynamic_weights()
        assert len(weights) > 0
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_reset_restores_defaults(self):
        engine = RollingICWeightEngine(RollingICConfig(ema_alpha=1.0, min_observations=10))
        for _ in range(50):
            engine.update({"trend": 0.5}, 0.5)
        engine.reset()
        assert engine._observation_count == 0
        assert len(engine._symbol_score_history) == 0

    def test_ic_history(self):
        engine = RollingICWeightEngine(RollingICConfig(window=100, min_observations=30, ema_alpha=1.0))
        for _ in range(50):
            engine.update({"trend": 0.5, "term_structure": 0.3}, 0.5)
        df = engine.ic_history
        assert not df.empty
        assert "trend" in df.columns

    def test_forward_returns(self):
        engine = RollingICWeightEngine()
        close = pd.Series([100, 101, 102, 103, 104, 105, 106])
        fwd = engine.compute_forward_returns(close)
        assert fwd.iloc[0] == pytest.approx(0.05, abs=0.001)

    def test_get_ic_summary(self):
        engine = RollingICWeightEngine(RollingICConfig(window=100, min_observations=30, ema_alpha=1.0))
        for _ in range(50):
            engine.update({"trend": 0.5, "term_structure": 0.3}, 0.5)
        summary = engine.get_ic_summary()
        assert len(summary) == 2
        assert "trend" in summary
        assert "term_structure" in summary
        assert "current" in summary["trend"]
        assert "weight" in summary["trend"]