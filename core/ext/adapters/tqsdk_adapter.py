"""TqSdk 数据源适配器（规则21）。

内部委托 core.data.data_loader.DataLoader(data_source="tqsdk")，不重写数据加载逻辑（规则21.4）。

依赖：tqsdk（pip install tqsdk）
按需安装：pip install -r requirements-data-sources.txt
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import pandas as pd

# 2026-06-20：使用完整模块路径 core.data.data_loader（替代 core.data_loader 顶层 re-export），
# 避免依赖 core/__init__.py 的间接导出，更明确模块归属。
from core.data.data_loader import DataLoader
from core.ext.adapters.base import DataSourceAdapter
from core.ext.adapters.factory import register_adapter


@register_adapter("tqsdk")
class TqSdkAdapter(DataSourceAdapter):
    """TqSdk 在线数据源适配器。

    内部直接使用 DataLoader(data_source="tqsdk")，所有加载/缓存/重连逻辑复用。
    """

    name = "tqsdk"

    def __init__(
        self,
        phone: Optional[str] = None,
        password: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        data_length: int = 2000,
        enable_cache: bool = True,
        cache_ttl_hours: int = 24,
        **kwargs,
    ) -> None:
        # 第三方 import 放在 __init__ 内，按需加载
        try:
            import tqsdk  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "TqSdkAdapter 需要 tqsdk，请执行：pip install tqsdk "
                "或 pip install -r requirements-data-sources.txt"
            ) from e

        # 内部使用 data_source='tqsdk'，静默 DataLoader 的弃用警告（规则 22.3）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self._loader = DataLoader(
                data_source="tqsdk",
                phone=phone,
                password=password,
                symbols=symbols,
                data_length=data_length,
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
        return self._loader.get_bars(symbol, start_date, end_date, timeframe)

    def get_universe(
        self,
        date: Optional[str] = None,
        min_volume: float = 50000,
        max_margin: float = 5000,
    ) -> List[str]:
        return self._loader.get_universe(date, min_volume, max_margin)

    def validate_data(
        self,
        df: pd.DataFrame,
        min_rows: int = 100,
        max_missing: float = 0.05,
    ) -> bool:
        return self._loader.validate_data(df, min_rows, max_missing)


__all__ = ["TqSdkAdapter"]
