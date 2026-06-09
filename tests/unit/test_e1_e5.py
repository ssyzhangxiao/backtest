"""
测试 e1_e5.py 模块
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from runner.backtest.experiments.e1_e5 import (
    WeightedSignalFusion,
    _calculate_rolling_volatility,
    _calculate_risk_parity_weights,
    PortfolioResult,
)


class TestWeightedSignalFusion:
    """测试加权信号融合类"""

    def test_combine_signals_equal_weights(self):
        """测试等权信号融合"""
        fusion = WeightedSignalFusion({"a": 0.5, "b": 0.5})
        result = fusion.combine({"a": 10, "b": 20})
        assert result == 15.0

    def test_combine_signals_unequal_weights(self):
        """测试不等权信号融合"""
        fusion = WeightedSignalFusion({"a": 0.3, "b": 0.7})
        result = fusion.combine({"a": 10, "b": 20})
        assert result == 17.0

    def test_combine_signals_missing_strategy(self):
        """测试缺少策略的情况"""
        fusion = WeightedSignalFusion({"a": 0.5, "b": 0.5})
        result = fusion.combine({"a": 10})
        assert result == 10.0  # 只有 a 的信号，权重归一化后全部在 a


class TestCalculateRollingVolatility:
    """测试滚动波动率计算"""

    def test_calculate_rolling_volatility(self):
        """测试正常计算"""
        # 创建收益率序列
        dates = pd.date_range(start="2020-01-01", periods=100)
        returns = pd.Series(np.random.normal(0, 0.01, 100), index=dates)

        # 计算滚动波动率
        vol = _calculate_rolling_volatility(returns, window=20)

        # 验证结果
        assert isinstance(vol, pd.Series)
        assert len(vol) == 100
        # 前 window-1 个值可能是 NaN
        assert not vol.isna().all()

    def test_calculate_rolling_volatility_short_series(self):
        """测试短序列计算"""
        dates = pd.date_range(start="2020-01-01", periods=10)
        returns = pd.Series(np.random.normal(0, 0.01, 10), index=dates)

        vol = _calculate_rolling_volatility(returns, window=20)

        assert isinstance(vol, pd.Series)
        # 由于序列太短，min_periods 会被应用
        assert not vol.isna().all()


class TestCalculateRiskParityWeights:
    """测试风险平价权重计算"""

    def test_calculate_risk_parity_weights(self):
        """测试正常计算"""
        # 创建策略收益率
        dates = pd.date_range(start="2020-01-01", periods=100)
        strategy_returns = {
            "strategy1": pd.Series(np.random.normal(0, 0.01, 100), index=dates),
            "strategy2": pd.Series(np.random.normal(0, 0.02, 100), index=dates),
            "strategy3": pd.Series(np.random.normal(0, 0.03, 100), index=dates),
        }

        # 计算风险平价权重
        weights_df = _calculate_risk_parity_weights(strategy_returns, window=20)

        # 验证结果
        assert isinstance(weights_df, pd.DataFrame)
        assert list(weights_df.columns) == ["strategy1", "strategy2", "strategy3"]

        # 验证权重总和接近 1
        row_sums = weights_df.sum(axis=1)
        assert all((row_sums >= 0.999) & (row_sums <= 1.001) | row_sums.isna())

        # 验证波动率越高的策略权重越低（使用最后一行的平均权重）
        avg_weights = weights_df.mean()
        # strategy1 波动率最低，应该权重最高
        assert avg_weights["strategy1"] > avg_weights["strategy2"]
        assert avg_weights["strategy2"] > avg_weights["strategy3"]

    def test_calculate_risk_parity_weights_single_strategy(self):
        """测试单策略情况"""
        dates = pd.date_range(start="2020-01-01", periods=100)
        strategy_returns = {
            "strategy1": pd.Series(np.random.normal(0, 0.01, 100), index=dates),
        }

        weights_df = _calculate_risk_parity_weights(strategy_returns, window=20)

        assert isinstance(weights_df, pd.DataFrame)
        assert list(weights_df.columns) == ["strategy1"]
        # 单策略权重应该总是 1
        assert (weights_df["strategy1"] == 1.0).all()


class TestPortfolioResult:
    """测试 PortfolioResult 类型"""

    def test_portfolio_result_creation(self):
        """测试创建 PortfolioResult"""
        metrics = {"total_return": 0.1, "sharpe": 1.5}
        equity = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10),
                "equity": np.linspace(1000000, 1100000, 10),
            }
        )

        # PortfolioResult 是 TypedDict，所以我们创建一个符合的字典
        result: PortfolioResult = {
            "metrics": metrics,
            "equity": equity,
        }

        assert "metrics" in result
        assert "equity" in result
        assert isinstance(result["equity"], pd.DataFrame)
