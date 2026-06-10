"""
策略选择模块。

委托 core/config/strategy_profiles.py 的 StrategyLibrary，
消除重复#4、#5。
"""

from typing import Any, Dict, List

from core.config.strategy_profiles import (
    StrategyLibrary,
    SUB_STRATEGY_NAMES,
)

# 横截面组合模式标识（仅在显式组合实验中使用，不作为单品种基线）
_COMPOSITE_MODE_NAMES = frozenset({"cross_sectional"})


def get_strategy_names(config: Dict[str, Any]) -> List[str]:
    """
    从配置中提取子策略名称列表。

    优先使用 StrategyLibrary，配置中无策略时回退到 SUB_STRATEGY_NAMES（5 个原子子策略）。

    P1 整改（2026-06-10）：默认排除横截面组合模式（cross_sectional），
    它是上层 PortfolioManager 组合实验使用的概念，不应在单品种基线回测中
    作为独立策略执行（其 Profile 没有可调用的指标构建器，会导致 equity 恒为初始值）。

    Args:
        config: 原始配置字典

    Returns:
        子策略名称列表
    """
    strategies = config.get("strategies", [])
    names = [s["name"] for s in strategies if isinstance(s, dict) and "name" in s]
    if not names:
        names = list(SUB_STRATEGY_NAMES)
    # 过滤横截面组合模式
    return [n for n in names if n not in _COMPOSITE_MODE_NAMES]


def get_composite_mode_names() -> List[str]:
    """获取横截面组合模式名称列表（用于组合实验）。"""
    return list(_COMPOSITE_MODE_NAMES)


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
