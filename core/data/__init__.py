"""数据加载器包。

拆分自原 core/data_loader.py（1166行），按职责分为4个子模块：
- _constants.py       常量映射（品种→交易所、默认品种、缓存配置）
- _tqsdk_mixin.py     TqSdk 数据源方法（连接/缓存/加载）
- _csv_mixin.py       CSV 数据源方法（格式检测/加载/后处理）
- _series_mixin.py    主力合约识别/连续序列/价差对
- data_loader.py      DataLoader 主类（Mixin 组合 + 输出接口）
"""

from core.data.data_loader import DataLoader
from core.data._constants import (
    CACHE_DIR,
    DAILY_SECONDS,
    DEFAULT_SYMBOLS,
    MAX_CONTRACTS_PER_PRODUCT,
    PRODUCT_EXCHANGE_MAP,
)

__all__ = [
    "DataLoader",
    "CACHE_DIR",
    "DAILY_SECONDS",
    "DEFAULT_SYMBOLS",
    "MAX_CONTRACTS_PER_PRODUCT",
    "PRODUCT_EXCHANGE_MAP",
]
