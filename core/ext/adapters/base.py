"""数据源适配器抽象基类（规则21）。

设计目标：
    1. 解耦数据源选择与使用：调用方只依赖 DataSourceAdapter 接口
    2. 支持可选安装：TqSdk/AKShare/RQData 各自独立 extras
    3. 复用核心：不重写数据加载逻辑，adapters 内部委托给 core.data_loader.DataLoader

实现约束（规则21.4）：
    - 子类必须委托 core.data_loader.DataLoader 加载数据，不得绕过
    - 子类必须实现 core.data_provider.DataProvider 的 3 个方法
    - 第三方依赖必须在对应 extras 中声明（requirements-data-sources.txt 等）

参考：
    .trae/rules/01-basics/21-ext-directory.md#214-复用约束不重复造轮子
"""

from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional

import pandas as pd

from core.data_provider import DataProvider


class DataSourceAdapter(DataProvider):
    """数据源适配器抽象基类（规则21.3）。

    所有 core/ext/adapters/xxx_adapter.py 必须继承本类并通过 @register_adapter("name") 注册。

    与 core.data_provider.DataProvider 的差异：
        - DataProvider 面向"已加载的数据源实例"
        - DataSourceAdapter 面向"按数据源类型可选加载的工厂层"
    """

    name: str = ""  # 子类必须设置唯一名称

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        """子类初始化：构造底层 DataLoader 或对应数据源实例。

        第三方依赖（tqsdk/akshare 等）的 import 必须放在 __init__ 内，
        以满足规则21.2：第三方依赖在适配器按需加载，未安装时立即报错。
        """
        ...

    # DataProvider 3 个方法由子类在委托 DataLoader 时实现
    def get_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        raise NotImplementedError

    def get_universe(
        self,
        date: Optional[str] = None,
        min_volume: float = 50000,
        max_margin: float = 5000,
    ) -> List[str]:
        raise NotImplementedError

    def validate_data(
        self,
        df: pd.DataFrame,
        min_rows: int = 100,
        max_missing: float = 0.05,
    ) -> bool:
        raise NotImplementedError
