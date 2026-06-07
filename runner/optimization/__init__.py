"""优化层：参数搜索与选择。"""

from dataclasses import replace
from typing import Any, TypeVar

T = TypeVar("T")


def copy_config(config: T, **overrides: Any) -> T:
    """
    复制 BacktestConfig 并按需覆盖字段（规则17 共享工具）。

    整改原因：grid_search.py / window_search.py 此前只复制 3 个字段（initial_cash /
    commission_rate / slippage_rate），遗漏 stop_loss_pct / max_position_pct /
    use_cross_section / rebalance_days 等影响策略行为的配置。本函数基于 dataclasses.replace
    复制所有字段后应用 overrides，调用方无需关心字段清单变化。

    Args:
        config: 任意 dataclass 实例（典型为 BacktestConfig）
        **overrides: 需要覆盖的字段

    Returns:
        新的实例（不修改原对象）
    """
    if hasattr(config, "__dataclass_fields__"):
        return replace(config, **overrides)
    # 非 dataclass 防御性回退：尝试 setattr 复制
    import copy

    new = copy.copy(config)
    for k, v in overrides.items():
        setattr(new, k, v)
    return new


__all__ = ["copy_config"]

