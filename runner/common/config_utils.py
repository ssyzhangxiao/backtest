"""
配置辅助函数。

提供配置字典的快捷访问函数，供 runner/ 层使用。
核心配置模型统一使用 core.config.BacktestConfig（规则2、17），
本模块不再重复定义配置类。

已删除的重复内容：
  - Pydantic BacktestConfig / FullConfig 等（与 core/config/ 重复）
  - validate_backtest_config() / _merge_with_defaults()（改用 BacktestConfig.from_yaml()）
  - load_config_from_yaml()（改用 BacktestConfig.from_yaml()）
"""

from pathlib import Path
from typing import Any, Dict, List


def get_backtest_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取回测配置。

    Args:
        config: 完整配置字典

    Returns:
        backtest 配置字典
    """
    return config.get("backtest", {})


def get_walkforward_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取 Walk-Forward 配置。

    兼容 window/step 和 train_bars/test_bars 两种命名。

    Args:
        config: 完整配置字典

    Returns:
        walk_forward 配置字典
    """
    wf_config = config.get("walk_forward", {})
    # 兼容性：同时支持 window/step 和 train_bars/test_bars
    if "window" not in wf_config and "train_bars" in wf_config:
        wf_config["window"] = wf_config["train_bars"]
    if "step" not in wf_config and "test_bars" in wf_config:
        wf_config["step"] = wf_config["test_bars"]
    return wf_config


def get_montecarlo_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取蒙特卡洛配置。

    Args:
        config: 完整配置字典

    Returns:
        monte_carlo 配置字典
    """
    return config.get("monte_carlo", {})


def get_factors_list(config: Dict[str, Any]) -> List[str]:
    """
    从配置获取因子列表。

    优先级：factor_weights 键 > factors.list > 默认列表。

    Args:
        config: 完整配置字典

    Returns:
        因子名称列表
    """
    # 先从 factor_weights 取键
    factor_weights = config.get("factor_weights", {})
    if factor_weights:
        return list(factor_weights.keys())

    # 备选：从 factors 配置取
    factors_config = config.get("factors", {})
    if "list" in factors_config:
        return factors_config["list"]

    # 默认因子
    return ["ts_momentum", "roll_yield", "alpha019", "alpha032"]


def get_missing_data_method(config: Dict[str, Any]) -> str:
    """
    获取缺失值处理方法。

    Args:
        config: 完整配置字典

    Returns:
        缺失值处理方法字符串
    """
    backtest = config.get("backtest", {})
    return backtest.get("missing_data_method", "fill_zero")
