"""
策略注册表 — 向后兼容层。

实际实现已迁移到 core.strategy_registry。
此模块仅保留导入重定向。
"""

from core.strategy_registry import (
    STRATEGY_REGISTRY,
    get_strategy_class,
    create_strategy,
)

__all__ = ["STRATEGY_REGISTRY", "get_strategy_class", "create_strategy"]
