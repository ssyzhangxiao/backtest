"""
测试 config_utils.py 模块。

仅测试保留的辅助函数，已删除的 Pydantic 配置模型和
validate_backtest_config / _merge_with_defaults 不再测试
（配置验证改用 core.config.BacktestConfig.from_yaml()）。
"""

import pytest

from runner.common.config_utils import (
    get_backtest_config,
    get_walkforward_config,
    get_montecarlo_config,
    get_factors_list,
    get_missing_data_method,
)


class TestConfigUtils:
    """配置辅助函数测试"""

    def test_get_backtest_config(self):
        """测试获取 backtest 配置"""
        config = {
            "backtest": {
                "initial_capital": 1000000,
                "commission": 0.0001,
            }
        }

        result = get_backtest_config(config)
        assert result["initial_capital"] == 1000000
        assert result["commission"] == 0.0001

    def test_get_backtest_config_empty(self):
        """测试空配置返回空字典"""
        result = get_backtest_config({})
        assert result == {}

    def test_get_walkforward_config(self):
        """测试获取 walkforward 配置"""
        config = {
            "walk_forward": {
                "window": 252,
                "step": 63,
            }
        }

        result = get_walkforward_config(config)
        assert result["window"] == 252
        assert result["step"] == 63

    def test_get_walkforward_config_compatibility(self):
        """测试 walkforward 配置兼容性（train_bars/test_bars → window/step）"""
        config = {
            "walk_forward": {
                "train_bars": 252,
                "test_bars": 63,
            }
        }

        result = get_walkforward_config(config)
        assert result["window"] == 252
        assert result["step"] == 63

    def test_get_montecarlo_config(self):
        """测试获取 montecarlo 配置"""
        config = {
            "monte_carlo": {
                "n_simulations": 1000,
                "random_seed": 42,
                "bankruptcy_threshold": 0.8,
            }
        }

        result = get_montecarlo_config(config)
        assert result["n_simulations"] == 1000
        assert result["random_seed"] == 42
        assert result["bankruptcy_threshold"] == 0.8

    def test_get_factors_list_from_weights(self):
        """测试从 factor_weights 获取因子列表"""
        config = {
            "factor_weights": {
                "momentum": 0.5,
                "value": 0.3,
                "quality": 0.2,
            }
        }

        result = get_factors_list(config)
        assert set(result) == {"momentum", "value", "quality"}

    def test_get_factors_list_from_config(self):
        """测试从 factors 配置获取因子列表"""
        config = {
            "factors": {
                "list": ["factor1", "factor2", "factor3"],
            }
        }

        result = get_factors_list(config)
        assert result == ["factor1", "factor2", "factor3"]

    def test_get_factors_list_default(self):
        """测试默认因子列表：返回 5 子策略（2026-06 架构调整：子策略优先于具体因子名）"""
        result = get_factors_list({})
        assert len(result) == 5
        assert "trend" in result
        assert "vol_breakout" in result

    def test_get_missing_data_method(self):
        """测试获取缺失值处理方法"""
        config = {
            "backtest": {
                "missing_data_method": "ffill",
            }
        }

        result = get_missing_data_method(config)
        assert result == "ffill"

    def test_get_missing_data_method_default(self):
        """测试默认缺失值处理方法"""
        result = get_missing_data_method({"backtest": {}})
        assert result == "fill_zero"
