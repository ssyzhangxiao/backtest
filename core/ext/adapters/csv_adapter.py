"""本地 CSV 数据源适配器（规则21）。

内部委托 core.data_loader.DataLoader(data_source="csv")，不重写数据加载逻辑（规则21.4）。

依赖：无（核心 requirements.txt 即可）
按需安装：核心包已包含
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import pandas as pd

from core.data_loader import DataLoader
from core.ext.adapters.base import DataSourceAdapter
from core.ext.adapters.factory import register_adapter


@register_adapter("csv")
class CsvAdapter(DataSourceAdapter):
    """本地 CSV 数据源适配器。

    内部直接使用 DataLoader(data_source="csv")，所有加载/展期/格式检测逻辑复用。
    DataLoader 的 DeprecationWarning 内部吞掉（迁移路径自身不报警）。
    """

    name = "csv"

    def __init__(
        self,
        data_dir: str,
        symbols: Optional[List[str]] = None,
        enable_cache: bool = True,
        cache_ttl_hours: int = 24,
        **kwargs,
    ) -> None:
        if not data_dir:
            raise ValueError("CsvAdapter 必须传入 data_dir 参数（CSV 文件目录）")
        # 内部使用 data_source='csv'，静默 DataLoader 的弃用警告（规则 22.3）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self._loader = DataLoader(
                data_source="csv",
                data_dir=data_dir,
                symbols=symbols,
                enable_cache=enable_cache,
                cache_ttl_hours=cache_ttl_hours,
                **kwargs,
            )

    def get_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        # 懒加载：首次访问时自动加载（避免每次都加载）
        if self._loader.full_df is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                self._loader.load_data()
        return self._loader.get_bars(symbol, start_date, end_date, timeframe)

    def get_universe(
        self,
        date: Optional[str] = None,
        min_volume: float = 50000,
        max_margin: float = 5000,
    ) -> List[str]:
        if self._loader.full_df is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                self._loader.load_data()
        return self._loader.get_universe(date, min_volume, max_margin)

    def validate_data(
        self,
        df: pd.DataFrame,
        min_rows: int = 100,
        max_missing: float = 0.05,
    ) -> bool:
        return self._loader.validate_data(df, min_rows, max_missing)


__all__ = ["CsvAdapter"]
