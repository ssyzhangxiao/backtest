"""
BacktestConfig 单元测试。

验证配置加载、YAML 解析、废弃字段移除等。
"""

import os
import tempfile

import yaml

from core.config import BacktestConfig, DEFAULT_FACTOR_WEIGHTS, INITIAL_CASH


class TestDefaultConfig:
    """验证默认配置值。"""

    def test_default_initial_cash(self):
        """默认初始资金为 100 万。"""
        config = BacktestConfig()
        assert config.initial_cash == 1_000_000

    def test_default_factor_weights(self):
        """P1-1 整改：默认因子权重为空字典（强制用户显式配置）。"""
        config = BacktestConfig()
        assert config.factor_weights == {}

    def test_default_rebalance_days(self):
        """默认调仓周期为3天。"""
        config = BacktestConfig()
        assert config.rebalance_days == 3

    def test_no_deprecated_fields(self):
        """不应包含废弃字段。"""
        config = BacktestConfig()
        assert not hasattr(config, "fusion_mode")
        assert not hasattr(config, "regime_filter_enabled")


class TestFromYaml:
    """验证从 YAML 加载配置。"""

    def test_load_from_yaml(self):
        """从 YAML 文件加载配置。"""
        yaml_content = {
            "backtest": {
                "initial_capital": 500000,
                "rebalance_freq": 5,
                "commission": 0.0005,
                "slippage": 0.0003,
                "stop_loss_pct": 0.08,
            },
            "factor_weights": {
                "trend": 0.3,
                "term_structure": 0.3,
                "mean_reversion": 0.2,
                "vol_breakout": 0.2,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = f.name

        try:
            config = BacktestConfig.from_yaml(yaml_path)
            assert config.initial_cash == 500000
            assert config.rebalance_days == 5
            assert config.commission_rate == 0.0005
            assert config.stop_loss_pct == 0.08
            assert config.factor_weights["trend"] == 0.3
        finally:
            os.unlink(yaml_path)

    def test_load_from_empty_yaml(self):
        """空 YAML 使用默认值。P1-1 整改：factor_weights 缺省时为空字典。"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            yaml_path = f.name

        try:
            config = BacktestConfig.from_yaml(yaml_path)
            assert config.initial_cash == INITIAL_CASH
            # P1-1 整改：缺省 factor_weights 时为空字典
            assert config.factor_weights == {}
        finally:
            os.unlink(yaml_path)


class TestDefaultFactorWeights:
    """验证因子权重常量。"""

    def test_five_factors(self):
        """应有5个因子。"""
        assert len(DEFAULT_FACTOR_WEIGHTS) == 5

    def test_weights_sum_to_one(self):
        """权重之和应为1。"""
        assert abs(sum(DEFAULT_FACTOR_WEIGHTS.values()) - 1.0) < 1e-6

    def test_expected_factors(self):
        """应包含预期的5个因子。"""
        expected = {"trend", "term_structure", "mean_reversion", "vol_breakout", "composite_resonance"}
        assert set(DEFAULT_FACTOR_WEIGHTS.keys()) == expected
