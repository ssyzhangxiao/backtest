"""
YAML 工具函数（公共工具，避免重复实现）。

提供：
  - convert_numpy_types：递归将 numpy 类型转换为 Python 原生类型
  - dump_yaml / load_yaml：安全的 YAML 读写封装
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import yaml


def convert_numpy_types(obj: Any) -> Any:
    """
    递归将 numpy 类型转换为 Python 原生类型（避免 yaml.safe_dump 报错）。

    Args:
        obj: 任意 Python 对象

    Returns:
        转换后的对象（numpy.int64 → int, numpy.float64 → float, numpy.ndarray → list）

    Examples:
        >>> convert_numpy_types(np.int64(5))
        5
        >>> convert_numpy_types({"a": np.float64(0.1)})
        {'a': 0.1}
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [convert_numpy_types(item) for item in obj]
        return converted if not isinstance(obj, tuple) else tuple(converted)
    return obj


def dump_yaml(path: str, raw: dict, sort_keys: bool = False) -> None:
    """
    将字典写入 YAML 文件（自动处理 numpy 类型）。

    Args:
        path: 目标文件路径
        raw: 顶层字典
        sort_keys: 是否按键排序（默认 False，保持插入顺序）
    """
    safe_raw = convert_numpy_types(raw)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(safe_raw, f, default_flow_style=False, allow_unicode=True, sort_keys=sort_keys)


def load_yaml(path: str) -> dict:
    """
    从 YAML 文件加载字典（不存在时返回空字典）。

    Args:
        path: YAML 文件路径

    Returns:
        解析后的字典
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


__all__ = ["convert_numpy_types", "dump_yaml", "load_yaml"]
