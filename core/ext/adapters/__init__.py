"""数据源适配器（adapters/）— 规则21。

使用方式：

    from core.ext.adapters import create_data_source, list_adapters

    # 按名字创建
    ds = create_data_source("csv", data_dir="/data/futures")
    ds = create_data_source("tqsdk", phone="...", password="...")

    # 列出已注册
    print(list_adapters())  # ['csv', 'tqsdk']

注册机制：
    - @register_adapter("name") 装饰器：注册到 core.ext.adapters.factory._DATA_SOURCE_REGISTRY
    - 触发注册：导入本包时，__init__.py 会显式 import 所有内置适配器
    - 新增适配器：写 xxx_adapter.py，在本 __init__.py 中追加 import 即可

复用约束（规则21.4）：
    - 必须继承 DataSourceAdapter（core/ext/adapters/base.py）
    - 必须委托 core.data_loader.DataLoader 加载数据
    - 不得绕过 DataLoader 直接读取本地 CSV / 调 tqsdk

规划适配器：
    - tqsdk_adapter.py   ✅ 已实现（pip install tqsdk）
    - csv_adapter.py     ✅ 已实现（核心包内置）
    - akshare_adapter.py 规划中（pip install akshare）
    - rqdata_adapter.py  规划中（pip install rqdatac）
"""

from __future__ import annotations

# 触发注册：导入即注册到 _DATA_SOURCE_REGISTRY
from .base import DataSourceAdapter
from .factory import register_adapter, create_data_source, list_adapters

# 内置适配器（核心包内置，无需额外依赖）
from . import csv_adapter  # noqa: F401
from . import tqsdk_adapter  # noqa: F401


__all__ = [
    "DataSourceAdapter",
    "register_adapter",
    "create_data_source",
    "list_adapters",
]
