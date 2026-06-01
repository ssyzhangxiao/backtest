"""
数据提供者抽象接口。

解耦数据获取逻辑，使回测引擎不依赖具体数据源（TqSdk/CSV/数据库）。
DataLoader 实现该接口，内部依赖 TqSdk。
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd


class DataProvider(ABC):
    """数据提供者抽象基类。"""

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """
        获取K线数据。

        Args:
            symbol: 品种代码
            start_date: 起始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            timeframe: K线周期

        Returns:
            DataFrame，至少包含: date, open, high, low, close, volume
        """
        ...

    @abstractmethod
    def get_universe(
        self,
        date: Optional[str] = None,
        min_volume: float = 50000,
        max_margin: float = 5000,
    ) -> List[str]:
        """
        获取品种池。

        Args:
            date: 日期 (YYYY-MM-DD)，None 表示最新
            min_volume: 最低日均成交量
            max_margin: 最高保证金门槛

        Returns:
            品种代码列表
        """
        ...

    @abstractmethod
    def validate_data(
        self,
        df: pd.DataFrame,
        min_rows: int = 100,
        max_missing: float = 0.05,
    ) -> bool:
        """
        校验数据质量。

        Args:
            df: 待校验数据
            min_rows: 最少行数
            max_missing: 最大缺失率

        Returns:
            True 表示数据合格

        Raises:
            ValueError: 数据不合格时抛出
        """
        ...
