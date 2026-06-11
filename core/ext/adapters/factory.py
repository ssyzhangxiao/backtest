"""数据源适配器工厂（规则21.3）。

设计：
    - @register_adapter("name") 装饰器：注册适配器类到全局注册表
    - create_data_source(name, **kwargs) 工厂：按名字创建实例
    - list_adapters() 工具：列出所有已注册适配器（用于错误提示和文档生成）

禁止：
    - 在 create_data_source 中硬编码 if/elif name == "x"（规则21.3）
    - 直接 import 具体适配器（必须通过工厂调用，避免硬编码依赖）
"""

from __future__ import annotations

from typing import Dict, List, Type

from .base import DataSourceAdapter


_DATA_SOURCE_REGISTRY: Dict[str, Type[DataSourceAdapter]] = {}


def register_adapter(name: str):
    """装饰器：注册适配器到工厂。

    Usage:
        @register_adapter("tqsdk")
        class TqSdkAdapter(DataSourceAdapter):
            name = "tqsdk"
            ...

    Args:
        name: 适配器唯一名称（snake_case，如 "tqsdk" / "csv" / "akshare"）

    Raises:
        ValueError: name 已注册时（防止覆盖）
    """
    def deco(cls: Type[DataSourceAdapter]) -> Type[DataSourceAdapter]:
        if name in _DATA_SOURCE_REGISTRY:
            raise ValueError(
                f"数据源 '{name}' 已注册为 {_DATA_SOURCE_REGISTRY[name].__name__}，"
                f"重复注册 {cls.__name__}"
            )
        if not cls.name:
            cls.name = name
        _DATA_SOURCE_REGISTRY[name] = cls
        return cls
    return deco


def create_data_source(name: str, **kwargs) -> DataSourceAdapter:
    """按名字创建数据源适配器实例。

    Args:
        name: 已注册的数据源名称
        **kwargs: 传递给适配器 __init__ 的参数

    Returns:
        DataSourceAdapter 实例

    Raises:
        KeyError: name 未注册时
    """
    if name not in _DATA_SOURCE_REGISTRY:
        raise KeyError(
            f"未知数据源 '{name}'，已注册: {list_adapters()}。"
            f"如需新增，请在 core/ext/adapters/ 下实现 xxx_adapter.py 并使用 @register_adapter('xxx') 注册。"
        )
    return _DATA_SOURCE_REGISTRY[name](**kwargs)


def list_adapters() -> List[str]:
    """列出所有已注册的数据源名称。"""
    return sorted(_DATA_SOURCE_REGISTRY.keys())


__all__ = [
    "DataSourceAdapter",
    "register_adapter",
    "create_data_source",
    "list_adapters",
]
