"""CTA 策略注册表。

可插拔设计：新增 CTA 策略只需实现 CTABaseStrategy 并注册到这里。
"""

from __future__ import annotations

from typing import Any, Dict, Type

from core.strategies.cta.base import CTABaseStrategy

# 注册表：{策略名: 策略类}
CTA_STRATEGY_REGISTRY: Dict[str, Type[CTABaseStrategy]] = {}


def register_cta_strategy(name: str, strategy_cls: Type[CTABaseStrategy]) -> None:
    """注册 CTA 策略类。"""
    if name in CTA_STRATEGY_REGISTRY:
        raise ValueError(f"CTA 策略 {name} 已注册")
    CTA_STRATEGY_REGISTRY[name] = strategy_cls


def get_cta_strategy(name: str, config: Dict[str, Any] = None) -> CTABaseStrategy:
    """获取 CTA 策略实例。

    Args:
        name: 策略名（必须已注册）
        config: 策略配置

    Returns:
        CTABaseStrategy 实例
    """
    cls = CTA_STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"未知 CTA 策略: {name}，已注册: {list(CTA_STRATEGY_REGISTRY.keys())}"
        )
    return cls(config or {})


__all__ = [
    "CTA_STRATEGY_REGISTRY",
    "register_cta_strategy",
    "get_cta_strategy",
]
