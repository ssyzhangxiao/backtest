"""
统一数据加载器模块。

支持两种数据源：
1. TqSdk 数据源：从 TqSdk 直接获取真实交易所合约数据，支持断线重连和数据缓存
2. 本地 CSV 数据源：读取本地 CSV 期货数据，支持合约模式和品种模式

输出格式兼容 PyBroker：包含 date, symbol, open, high, low, close, volume, open_interest,
以及辅助列 is_dominant, dominant_symbol, prev_dominant_symbol, rollover_flag, product。
"""

import logging
import warnings
from typing import Dict, List, Optional

import pandas as pd

from core.data_provider import DataProvider
from core.data._constants import DEFAULT_SYMBOLS
from core.data._tqsdk_mixin import TqsdkMixin
from core.data._csv_mixin import CsvMixin
from core.data._series_mixin import SeriesMixin

_logger = logging.getLogger(__name__)


class DataLoader(DataProvider, TqsdkMixin, CsvMixin, SeriesMixin):
    """
    统一数据加载器。

    支持两种数据源：
    1. TqSdk 数据源：从 TqSdk 直接获取真实交易所合约数据
    2. 本地 CSV 数据源：读取本地 CSV 期货数据

    对于 TqSdk 数据源，支持：
    - 断线自动重连
    - 数据缓存与本地存储
    - 完整的品种映射

    对于本地 CSV 数据源，支持：
    - 自动检测数据格式（合约模式/品种模式）
    - 合约模式：按合约分文件（如 RB2310.csv），包含具体合约数据，支持展期
    - 品种模式：按品种分文件（如 SHFE.RB.csv），品种连续指数数据，无展期

    Attributes:
        data_source: 数据源类型，'tqsdk' 或 'csv'
        data_dir: CSV文件所在目录路径（仅 CSV 模式）
        all_contracts: 所有合约的原始数据（合并后）
        dominant_map: 每日主力合约映射 {date: symbol}
        continuous_df: 展期法连续主力合约数据
        full_df: 包含所有合约的完整数据（用于跨期策略）
        data_mode: 数据模式，'contract' 或 'product'
        load_errors: 加载失败的文件及错误信息列表
    """

    PYBROKER_COLUMNS = (
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
        "is_dominant",
        "dominant_symbol",
        "prev_dominant_symbol",
        "rollover_flag",
        "product",
    )

    CONTRACT_REQUIRED_COLUMNS = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    ]
    PRODUCT_REQUIRED_COLUMNS = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "position",
    ]
    FORMAT_DETECT_ROWS = 5

    def __init__(
        self,
        data_source: str = "tqsdk",
        data_dir: Optional[str] = None,
        phone: Optional[str] = None,
        password: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        data_length: int = 2000,
        enable_cache: bool = True,
        cache_ttl_hours: int = 24,
        max_reconnect_attempts: int = 5,
        reconnect_delay_seconds: float = 2.0,
    ):
        """
        初始化数据加载器。

        .. deprecated::
            ``data_source`` 参数将在 v0.2.0 删除（规则 22 迁移阶段 3）。
            新接口：``from core.ext.adapters import create_data_source``：

            >>> from core.ext.adapters import create_data_source
            >>> ds = create_data_source("tqsdk", phone="...", password="...")
            >>> # 或
            >>> ds = create_data_source("csv", data_dir="/data/")

        Args:
            data_source: 数据源类型，'tqsdk' 或 'csv'
            data_dir: CSV文件所在目录路径（仅 CSV 模式）
            phone: TqSdk 账号手机号（仅 TqSdk 模式）
            password: TqSdk 账号密码（仅 TqSdk 模式）
            symbols: 品种代码列表，默认加载核心品种（仅 TqSdk 模式）
            data_length: 每个合约下载的 K 线数量（日线），仅 TqSdk 模式
            enable_cache: 是否启用数据缓存（仅 TqSdk 模式）
            cache_ttl_hours: 缓存有效期（小时）（仅 TqSdk 模式）
            max_reconnect_attempts: 最大重连次数（仅 TqSdk 模式）
            reconnect_delay_seconds: 重连延迟（秒）（仅 TqSdk 模式）
        """
        if data_source not in ("tqsdk", "csv"):
            raise ValueError(
                f"data_source={data_source!r} 已废弃（规则 22）。"
                f"新接口：create_data_source(name, **kwargs)。"
            )
        warnings.warn(
            f"DataLoader(data_source={data_source!r}) 已废弃（规则 22 迁移阶段 1），"
            f"将于 v0.2.0 删除。请改用 create_data_source({data_source!r}, ...) "
            f"——详见 .trae/notes/migration-audit-adapters.md",
            DeprecationWarning,
            stacklevel=2,
        )
        self.data_source = data_source.lower()
        self.data_dir = data_dir

        # TqSdk 相关配置
        self._phone = phone
        self._password = password
        self._symbols = symbols or DEFAULT_SYMBOLS
        self._data_length = data_length
        self._enable_cache = enable_cache
        self._cache_ttl_hours = cache_ttl_hours
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_delay_seconds = reconnect_delay_seconds
        self._api = None

        # 数据存储
        self.all_contracts: Optional[pd.DataFrame] = None
        self.dominant_map: Optional[pd.Series] = None
        self.continuous_df: Optional[pd.DataFrame] = None
        self.full_df: Optional[pd.DataFrame] = None
        self.data_mode: Optional[str] = None
        self.load_errors: List[Dict] = []
        self._product_symbols: Optional[Dict[str, List[str]]] = None

    # ------------------------------------------------------------------
    # 统一加载方法
    # ------------------------------------------------------------------

    def load_data(self, file_pattern: str = "*.csv", show_progress: bool = True):
        """
        统一加载数据的方法。

        .. deprecated::
            将于 v0.2.0 删除（规则 22 迁移阶段 3）。
            新接口：``create_data_source(name, ...).load()``。

        Args:
            file_pattern: 文件匹配模式（仅 CSV 模式）
            show_progress: 是否显示进度条（仅 TqSdk 模式）

        Returns:
            合并后的 DataFrame
        """
        warnings.warn(
            f"DataLoader.load_data() 已废弃（规则 22），data_source={self.data_source!r}。",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.data_source == "tqsdk":
            return self.load_from_tqsdk(show_progress=show_progress)
        elif self.data_source == "csv":
            return self.load_csv_files(file_pattern=file_pattern)
        else:
            raise ValueError(f"不支持的数据源类型: {self.data_source}")

    # ------------------------------------------------------------------
    # 输出接口
    # ------------------------------------------------------------------

    def get_pybroker_df(self) -> pd.DataFrame:
        """
        获取 PyBroker 兼容格式的 DataFrame（仅主力合约，按品种拼接连续序列）。

        非主力合约数据质量差且流动性不足，不应参与回测。
        spread/term_structure 所需的远月数据通过 far_close 列注入。

        关键：symbol 列使用品种代码（如 SHFE.RB）而非具体合约（如 SHFE.rb2401），
        使 PyBroker 按品种分组运行策略，避免换月时产生大量无意义交易。
        """
        if self.full_df is None:
            self.build_continuous_series()

        # 仅输出主力合约行
        dominant_df = self.full_df[self.full_df["is_dominant"]].copy()

        # 用品种代码替代具体合约名，实现连续序列
        if "product" in dominant_df.columns and self.data_mode == "contract":
            if "symbol" in dominant_df.columns:
                dominant_df["exchange"] = dominant_df["symbol"].str.split(".").str[0]
                dominant_df["symbol"] = (
                    dominant_df["exchange"] + "." + dominant_df["product"].str.upper()
                )
                dominant_df.drop(columns=["exchange"], inplace=True)

        available = [c for c in self.PYBROKER_COLUMNS if c in dominant_df.columns]
        # 如果有 spread 相关列也一并输出
        for col in ("far_symbol", "far_close", "spread"):
            if col in dominant_df.columns and col not in available:
                available.append(col)
        result = dominant_df[available].copy()
        return result.sort_values(["date", "symbol"]).reset_index(drop=True)

    def get_dominant_only_df(self) -> pd.DataFrame:
        """仅获取主力合约数据。"""
        if self.continuous_df is None:
            self.build_continuous_series()
        return self.continuous_df.copy()

    def get_rollover_dates(self) -> pd.DataFrame:
        """获取所有展期日期及对应的合约切换信息。"""
        if self.full_df is None:
            self.build_continuous_series()

        if self.data_mode == "product":
            return pd.DataFrame(
                columns=["date", "prev_dominant_symbol", "dominant_symbol"]
            )

        rollover = (
            self.full_df[self.full_df["rollover_flag"]]
            .drop_duplicates(subset=["date"])[
                ["date", "prev_dominant_symbol", "dominant_symbol"]
            ]
            .reset_index(drop=True)
        )
        return rollover

    @property
    def product_symbols(self) -> Dict[str, List[str]]:
        """获取各品种的合约代码列表（带缓存）。"""
        if self._product_symbols is None:
            if self.all_contracts is None:
                raise RuntimeError("请先调用 load_data() 加载数据")
            self._product_symbols = {
                p: sorted(g["symbol"].unique().tolist())
                for p, g in self.all_contracts.groupby("product")
            }
        return self._product_symbols

    def get_product_symbols(
        self, product: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """获取指定品种的合约代码列表。"""
        ps = self.product_symbols
        if product:
            return {product: ps.get(product, [])}
        return ps

    def get_data_summary(self) -> Dict:
        """获取数据摘要。"""
        if self.all_contracts is None:
            return {"status": "未加载数据"}

        products = self.product_symbols
        summary = {
            "data_source": self.data_source,
            "data_mode": self.data_mode,
            "total_symbols": self.all_contracts["symbol"].nunique(),
            "date_range": (
                str(self.all_contracts["date"].min().date()),
                str(self.all_contracts["date"].max().date()),
            ),
            "products": {
                p: {"contracts": len(syms), "symbols": syms[:5]}
                for p, syms in products.items()
            },
        }
        if self.full_df is not None and self.data_mode == "contract":
            summary["rollover_count"] = len(self.get_rollover_dates())
        if self.load_errors:
            summary["load_errors"] = self.load_errors
        return summary

    def close(self):
        """关闭连接（仅 TqSdk 模式）。"""
        if self.data_source == "tqsdk":
            self._close_api()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── DataProvider 接口实现 ──

    def get_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """DataProvider 接口：获取K线数据。"""
        df = self.get_pybroker_df()
        mask = (
            (df["symbol"] == symbol)
            & (df["date"] >= start_date)
            & (df["date"] <= end_date)
        )
        return df.loc[mask].reset_index(drop=True)

    def get_universe(
        self,
        date: Optional[str] = None,
        min_volume: float = 50000,
        max_margin: float = 5000,
    ) -> List[str]:
        """DataProvider 接口：获取品种池。"""
        df = self.get_pybroker_df()
        if date:
            df = df[df["date"] <= date]
        symbols = df["symbol"].unique().tolist()
        return symbols

    def validate_data(
        self,
        df: pd.DataFrame,
        min_rows: int = 100,
        max_missing: float = 0.05,
    ) -> bool:
        """DataProvider 接口：校验数据质量。"""
        if len(df) < min_rows:
            raise ValueError(f"数据行数不足: {len(df)} < {min_rows}")
        missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_rate > max_missing:
            raise ValueError(f"缺失率过高: {missing_rate:.2%} > {max_missing:.2%}")
        return True
