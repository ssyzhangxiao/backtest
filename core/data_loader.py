"""兼容层 — core.data_loader 已迁移到 core.data 包。

此文件仅提供向后兼容的导入重定向，将在 v0.2.0 删除。
新导入路径：from core.data import DataLoader
"""

import sys
import warnings

warnings.warn(
    "core.data_loader 已迁移到 core.data 包，"
    "请改用 'from core.data import DataLoader'。"
    "此兼容层将在 v0.2.0 删除。",
    DeprecationWarning,
    stacklevel=2,
)

from core.data import DataLoader
from core.data._constants import (
    CACHE_DIR,
    DAILY_SECONDS,
    DEFAULT_SYMBOLS,
    MAX_CONTRACTS_PER_PRODUCT,
    PRODUCT_EXCHANGE_MAP,
)

# 保持旧常量名兼容
_CACHE_DIR = CACHE_DIR
_DAILY_SECONDS = DAILY_SECONDS
_DEFAULT_SYMBOLS = DEFAULT_SYMBOLS
_MAX_CONTRACTS_PER_PRODUCT = MAX_CONTRACTS_PER_PRODUCT

__all__ = [
    "DataLoader",
    "CACHE_DIR",
    "_CACHE_DIR",
    "DAILY_SECONDS",
    "_DAILY_SECONDS",
    "DEFAULT_SYMBOLS",
    "_DEFAULT_SYMBOLS",
    "MAX_CONTRACTS_PER_PRODUCT",
    "_MAX_CONTRACTS_PER_PRODUCT",
    "PRODUCT_EXCHANGE_MAP",
]
