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

    支持 `${VAR_NAME}` 环境变量替换：
      - `tqsdk_phone: ${TQSDK_PHONE}` → 展开为 `os.environ["TQSDK_PHONE"]`
      - 变量不存在时保留原样（不抛错）
      - 自动加载 .env 文件（调用 `load_dotenv()`）

    Args:
        path: YAML 文件路径

    Returns:
        解析后的字典
    """
    # 确保 .env 已加载，使 ${VAR} 展开生效
    try:
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv()
    except ImportError:
        pass
    import os, re

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return {}

    def _replace_var(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(0))

    text = re.sub(r"\$\{(\w+)\}", _replace_var, text)
    return yaml.safe_load(text) or {}


__all__ = ["convert_numpy_types", "dump_yaml", "load_yaml"]
