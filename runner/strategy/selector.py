"""
策略选择模块。

委托 core/config/strategy_profiles.py 的 StrategyLibrary，
消除重复#4、#5。
"""

from typing import Any, Dict, List

from loguru import logger

from core.config.strategy_profiles import StrategyLibrary


def get_strategy_names(config: Dict[str, Any]) -> List[str]:
    """
    从配置中提取策略名称列表。

    优先使用 StrategyLibrary，配置中无策略时回退到 StrategyLibrary.list_all()。

    Args:
        config: 原始配置字典

    Returns:
        策略名称列表
    """
    strategies = config.get("strategies", [])
    names = [s["name"] for s in strategies if isinstance(s, dict) and "name" in s]
    if not names:
        lib = StrategyLibrary()
        names = [p.name for p in lib.list_all()]
    return names


def get_param_spaces(
    lib: StrategyLibrary,
    strategy_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    获取各策略的参数搜索空间。

    委托 StrategyLibrary.get_profile().param_ranges，
    消除重复#5。

    Args:
        lib: 策略库实例
        strategy_names: 策略名称列表

    Returns:
        {策略名: {参数名: 参数范围}} 字典
    """
    param_spaces: Dict[str, Dict[str, Any]] = {}
    for sname in strategy_names:
        profile = lib.get_profile(sname)
        if profile is not None and profile.param_ranges:
            param_spaces[sname] = dict(profile.param_ranges)
    return param_spaces
