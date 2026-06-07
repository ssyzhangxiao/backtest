"""
FactorScoringEngine 单元测试。

验证调仓日判断、因子得分计算、综合得分合成等核心逻辑。
"""

import pytest
from unittest.mock import MagicMock

from core.engine.switch_engine import FactorScoringEngine, ScoringConfig
from core.config.strategy_profiles import StrategyLibrary
from core.config import DEFAULT_FACTOR_WEIGHTS


class TestFactorScoringEngineRenaming:
    """验证重命名一致性。"""

    def test_new_name_instantiation(self):
        """使用新名称实例化。"""
        engine = FactorScoringEngine(StrategyLibrary())
        assert isinstance(engine, FactorScoringEngine)


class TestIsRebalanceDay:
    """验证调仓日判断逻辑。"""

    def _fresh_engine(self, days: int = 3):
        """每个测试获取独立engine，避免setup_method状态污染。"""
        from datetime import date
        engine = FactorScoringEngine(StrategyLibrary(), ScoringConfig(rebalance_days=days))
        engine.mark_rebalanced(date(2024, 1, 1))
        return engine

    def test_rebalance_day_1(self):
        """基准日+3天应为调仓日。"""
        from datetime import timedelta, date
        engine = self._fresh_engine(3)
        assert engine.is_rebalance_day(date(2024, 1, 1) + timedelta(days=3)) is True

    def test_rebalance_day_4(self):
        """基准日+6天应为调仓日（每3天）。"""
        from datetime import timedelta, date
        engine = self._fresh_engine(3)
        assert engine.is_rebalance_day(date(2024, 1, 1) + timedelta(days=6)) is True

    def test_not_rebalance_day_2(self):
        """基准日+1天不应为调仓日。"""
        from datetime import timedelta, date
        engine = self._fresh_engine(3)
        assert engine.is_rebalance_day(date(2024, 1, 1) + timedelta(days=1)) is False

    def test_not_rebalance_day_3(self):
        """基准日+2天不应为调仓日。"""
        from datetime import timedelta, date
        engine = self._fresh_engine(3)
        assert engine.is_rebalance_day(date(2024, 1, 1) + timedelta(days=2)) is False

    def test_rebalance_day_7(self):
        """基准日+9天应为调仓日。"""
        from datetime import timedelta, date
        engine = self._fresh_engine(3)
        assert engine.is_rebalance_day(date(2024, 1, 1) + timedelta(days=9)) is True

    def test_rebalance_freq_5(self):
        """5天周期：基准日、+5天为调仓日。"""
        from datetime import timedelta, date
        engine = self._fresh_engine(5)
        assert engine.is_rebalance_day(date(2024, 1, 1) + timedelta(days=5)) is True
        assert engine.is_rebalance_day(date(2024, 1, 1) + timedelta(days=4)) is False


class TestComputeCompositeScore:
    """验证综合得分计算。"""

    def setup_method(self):
        self.engine = FactorScoringEngine(StrategyLibrary(), ScoringConfig())

    def test_all_positive_scores(self):
        """所有因子为正时，综合得分为正。"""
        scores = {"trend": 0.5, "term_structure": 0.3, "mean_reversion": 0.4, "vol_breakout": 0.2}
        result = self.engine.compute_composite_score("TEST", scores)
        assert result > 0

    def test_all_negative_scores(self):
        """所有因子为负时，综合得分为负。"""
        scores = {"trend": -0.5, "term_structure": -0.3, "mean_reversion": -0.4, "vol_breakout": -0.2}
        result = self.engine.compute_composite_score("TEST", scores)
        assert result < 0

    def test_equal_weights_average(self):
        """等权时综合得分等于各因子简单平均。"""
        scores = {"trend": 0.4, "term_structure": 0.4, "mean_reversion": 0.4, "vol_breakout": 0.4}
        result = self.engine.compute_composite_score("TEST", scores)
        assert abs(result - 0.4) < 1e-6

    def test_empty_scores(self):
        """空得分返回0。"""
        result = self.engine.compute_composite_score("TEST", {})
        assert result == 0.0

    def test_partial_scores(self):
        """部分因子缺失时仍可计算。"""
        scores = {"trend": 0.5, "mean_reversion": 0.3}
        result = self.engine.compute_composite_score("TEST", scores)
        assert isinstance(result, float)


class TestExtractIndicator:
    """验证指标提取逻辑。"""

    def test_extract_from_series(self):
        """从 pandas Series 提取最后一个值。"""
        import pandas as pd
        ctx = MagicMock()
        ctx.indicator.return_value = pd.Series([1.0, 2.0, 3.0])
        result = FactorScoringEngine.extract_indicator(ctx, "test_ind")
        assert result == 3.0

    def test_extract_from_list(self):
        """从列表提取最后一个值。"""
        ctx = MagicMock()
        ctx.indicator.return_value = [10.0, 20.0, 30.0]
        result = FactorScoringEngine.extract_indicator(ctx, "test_ind")
        assert result == 30.0

    def test_extract_failure_returns_none(self):
        """提取失败返回 None。"""
        ctx = MagicMock()
        ctx.indicator.side_effect = Exception("not found")
        result = FactorScoringEngine.extract_indicator(ctx, "missing_ind")
        assert result is None


class TestScoringConfig:
    """验证 ScoringConfig 默认值。"""

    def test_default_weights(self):
        """默认权重应与 config.DEFAULT_FACTOR_WEIGHTS 一致。"""
        config = ScoringConfig()
        assert config.factor_weights == DEFAULT_FACTOR_WEIGHTS

    def test_default_rebalance_days(self):
        """默认调仓周期为3天。"""
        config = ScoringConfig()
        assert config.rebalance_days == 3
