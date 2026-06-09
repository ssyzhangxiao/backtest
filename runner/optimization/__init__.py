"""优化层：参数搜索与选择。"""

from dataclasses import replace
from typing import Any, TypeVar

T = TypeVar("T")


def copy_config(config: T, **overrides: Any) -> T:
    """
    复制 dataclass 配置并按需覆盖字段（规则17 共享工具）。

    与 ``dataclasses.replace`` 的关系：
      本函数是 ``dataclasses.replace`` 的"超集+防御性回退"封装，核心逻辑完全复用
      ``dataclasses.replace``。两者等价，但 ``copy_config`` 在以下场景更健壮：
        1. **非 dataclass 防御性回退**：当 ``config`` 不是 dataclass 时（罕见但可能），
           自动用 ``copy.copy`` + ``setattr`` 复制，避免抛 ``AttributeError``。
        2. **统一入口**：调用方无需判断 ``hasattr(config, '__dataclass_fields__')``，
           集中维护"如何复制配置对象"的语义。

    整改历史：
      - grid_search.py / window_search.py 此前只复制 3 个字段（initial_cash /
        commission_rate / slippage_rate），遗漏 stop_loss_pct / max_position_pct /
        use_cross_section / rebalance_days 等影响策略行为的配置。
      - 本函数基于 ``dataclasses.replace`` 复制所有字段后应用 overrides，调用方
        无需关心字段清单变化（新增字段自动支持）。

    Args:
        config: 任意 dataclass 实例（典型为 ``BacktestConfig``）
        **overrides: 需要覆盖的字段，以 ``field=value`` 形式传入

    Returns:
        新的实例（不修改原对象），类型与 ``config`` 相同

    Examples:
        基本用法：覆盖回测初始资金::

            from core.config import BacktestConfig
            cfg = BacktestConfig.from_yaml("config.yaml")
            cfg_small = copy_config(cfg, initial_cash=50_000)

        多字段覆盖::

            cfg_opt = copy_config(
                cfg,
                initial_cash=100_000,
                commission_rate=0.0005,
                stop_loss_pct=0.08,
                rebalance_days=3,
            )

        与 ``dataclasses.replace`` 等价（dataclass 场景）::

            from dataclasses import replace
            assert copy_config(cfg, x=1) == replace(cfg, x=1)

        非 dataclass 防御性回退::

            class LegacyCfg:
                def __init__(self, x): self.x = x
            legacy = LegacyCfg(1)
            copy_config(legacy, x=2)  # LegacyCfg(x=2)，不抛异常
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

