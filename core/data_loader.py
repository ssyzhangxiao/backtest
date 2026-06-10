"""
统一数据加载器模块。

支持两种数据源：
1. TqSdk 数据源：从 TqSdk 直接获取真实交易所合约数据，支持断线重连和数据缓存
2. 本地 CSV 数据源：读取本地 CSV 期货数据，支持合约模式和品种模式

输出格式兼容 PyBroker：包含 date, symbol, open, high, low, close, volume, open_interest,
以及辅助列 is_dominant, dominant_symbol, prev_dominant_symbol, rollover_flag, product。
"""

import os
import re
import glob
import time
import pickle
import logging
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 缓存目录配置
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 品种 → (TqSdk 交易所, TqSdk 品种代码) 完整映射
# ---------------------------------------------------------------------------
PRODUCT_EXCHANGE_MAP: Dict[str, Tuple[str, str]] = {
    # 上期所 SHFE
    "SHFE.RB": ("SHFE", "rb"),    "SHFE.AG": ("SHFE", "ag"),
    "SHFE.AU": ("SHFE", "au"),    "SHFE.AL": ("SHFE", "al"),
    "SHFE.ZN": ("SHFE", "zn"),    "SHFE.CU": ("SHFE", "cu"),
    "SHFE.NI": ("SHFE", "ni"),    "SHFE.SN": ("SHFE", "sn"),
    "SHFE.PB": ("SHFE", "pb"),    "SHFE.HC": ("SHFE", "hc"),
    "SHFE.BU": ("SHFE", "bu"),    "SHFE.RU": ("SHFE", "ru"),
    "SHFE.SS": ("SHFE", "ss"),    "SHFE.SP": ("SHFE", "sp"),
    "SHFE.BR": ("SHFE", "br"),    "SHFE.AO": ("SHFE", "ao"),
    # 大商所 DCE
    "DCE.M":  ("DCE", "m"),       "DCE.I":  ("DCE", "i"),
    "DCE.J":  ("DCE", "j"),       "DCE.JM": ("DCE", "jm"),
    "DCE.C":  ("DCE", "c"),       "DCE.CS": ("DCE", "cs"),
    "DCE.A":  ("DCE", "a"),       "DCE.B":  ("DCE", "b"),
    "DCE.P":  ("DCE", "p"),       "DCE.Y":  ("DCE", "y"),
    "DCE.L":  ("DCE", "l"),       "DCE.PP": ("DCE", "pp"),
    "DCE.V":  ("DCE", "v"),       "DCE.EB": ("DCE", "eb"),
    "DCE.EG": ("DCE", "eg"),      "DCE.PG": ("DCE", "pg"),
    "DCE.JD": ("DCE", "jd"),      "DCE.LH": ("DCE", "lh"),
    # 郑商所 CZCE
    "CZCE.TA": ("CZCE", "TA"),    "CZCE.MA": ("CZCE", "MA"),
    "CZCE.FG": ("CZCE", "FG"),    "CZCE.SA": ("CZCE", "SA"),
    "CZCE.SF": ("CZCE", "SF"),    "CZCE.SM": ("CZCE", "SM"),
    "CZCE.CF": ("CZCE", "CF"),    "CZCE.SR": ("CZCE", "SR"),
    "CZCE.OI": ("CZCE", "OI"),    "CZCE.RM": ("CZCE", "RM"),
    "CZCE.PF": ("CZCE", "PF"),    "CZCE.PX": ("CZCE", "PX"),
    "CZCE.SH": ("CZCE", "SH"),    "CZCE.UR": ("CZCE", "UR"),
    "CZCE.ZC": ("CZCE", "ZC"),    "CZCE.AP": ("CZCE", "AP"),
    "CZCE.CY": ("CZCE", "CY"),    "CZCE.PK": ("CZCE", "PK"),
    # 中金所 CFFEX
    "CFFEX.IF": ("CFFEX", "IF"),  "CFFEX.IC": ("CFFEX", "IC"),
    "CFFEX.IH": ("CFFEX", "IH"),  "CFFEX.IM": ("CFFEX", "IM"),
    "CFFEX.T":  ("CFFEX", "T"),   "CFFEX.TF": ("CFFEX", "TF"),
    "CFFEX.TS": ("CFFEX", "TS"),
    # 能源中心 INE
    "INE.SC": ("INE", "sc"),      "INE.NR": ("INE", "nr"),
    "INE.BC": ("INE", "bc"),      "INE.LU": ("INE", "lu"),
    "INE.EC": ("INE", "ec"),
    # 广期所 GFEX
    "GFEX.LC": ("GFEX", "LC"),    "GFEX.SI": ("GFEX", "SI"),
}


# ---------------------------------------------------------------------------
# 默认加载的核心品种
# ---------------------------------------------------------------------------
_DEFAULT_SYMBOLS = [
    "SHFE.RB", "SHFE.HC", "SHFE.AU", "SHFE.AG", "SHFE.CU",
    "DCE.M", "DCE.I", "DCE.J", "DCE.JM", "DCE.C",
    "DCE.P", "DCE.Y", "DCE.EG", "DCE.PP", "DCE.L",
    "CZCE.TA", "CZCE.MA", "CZCE.FG", "CZCE.SA", "CZCE.CF",
    "CZCE.OI", "CZCE.RM", "CZCE.SR", "CZCE.ZC",
    "CFFEX.IF", "CFFEX.IC", "CFFEX.IH",
    "INE.SC", "INE.NR",
]


# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------
_DAILY_SECONDS = 86400  # 86400 秒 = 1 天，用于 TqSdk K 线日线周期
_MAX_CONTRACTS_PER_PRODUCT = 20  # 每个品种最多下载的合约数


from core.data_provider import DataProvider


class DataLoader(DataProvider):
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
        "date", "symbol", "open", "high", "low", "close",
        "volume", "open_interest",
        "is_dominant", "dominant_symbol", "prev_dominant_symbol",
        "rollover_flag", "product",
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
        self.data_source = data_source.lower()
        self.data_dir = data_dir

        # TqSdk 相关配置
        self._phone = phone
        self._password = password
        self._symbols = symbols or _DEFAULT_SYMBOLS
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
    # TqSdk 相关方法
    # ------------------------------------------------------------------

    def _get_cache_path(self, symbol: str) -> Path:
        """获取品种数据的缓存路径"""
        sanitized = symbol.replace(".", "_").replace("-", "_")
        return CACHE_DIR / f"{sanitized}_{self._data_length}.pkl"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """检查缓存是否有效"""
        if not self._enable_cache:
            return False
        if not cache_path.exists():
            return False

        mtime = cache_path.stat().st_mtime
        hours_since = (time.time() - mtime) / 3600
        return hours_since < self._cache_ttl_hours

    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        cache_path = self._get_cache_path(symbol)
        if not self._is_cache_valid(cache_path):
            return None

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            _logger.info("缓存加载成功: %s", symbol)
            return data
        except Exception as e:
            _logger.warning("缓存加载失败: %s, 错误: %s", symbol, e)
            return None

    def _save_to_cache(self, symbol: str, data: pd.DataFrame):
        """保存数据到缓存"""
        if not self._enable_cache:
            return

        cache_path = self._get_cache_path(symbol)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            _logger.info("缓存保存成功: %s", symbol)
        except Exception as e:
            _logger.warning("缓存保存失败: %s, 错误: %s", symbol, e)

    def _create_api(self):
        """创建 TqSdk API 连接（带重连机制）"""
        from tqsdk import TqApi, TqAuth

        if not self._phone or not self._password:
            raise ValueError("TqSdk 认证信息缺失：请提供 phone 和 password")

        for attempt in range(self._max_reconnect_attempts + 1):
            try:
                _logger.info(
                    "连接 TqSdk (%d/%d)...",
                    attempt + 1, self._max_reconnect_attempts + 1,
                )
                self._api = TqApi(auth=TqAuth(self._phone, self._password))
                _logger.info("TqSdk 连接成功")
                return
            except PermissionError as e:
                raise PermissionError(
                    f"TqSdk 认证失败，请检查账号密码: {e}"
                ) from e
            except Exception as e:
                if attempt < self._max_reconnect_attempts:
                    delay = self._reconnect_delay_seconds * (2 ** attempt)
                    _logger.warning("连接失败: %s, %.1f 秒后重试...", e, delay)
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        f"连接 TqSdk 失败（尝试 {self._max_reconnect_attempts + 1} 次）: {e}"
                    ) from e

    def _close_api(self):
        """关闭 API 连接"""
        if self._api:
            try:
                self._api.close()
                self._api = None
                _logger.info("TqSdk 连接已关闭")
            except Exception as e:
                _logger.warning("关闭连接失败: %s", e)

    @staticmethod
    def _parse_contract_month(symbol: str) -> Tuple[int, int]:
        """从合约代码中提取年份和月份。"""
        m = re.search(r'(\d{2})(\d{2})$', symbol)
        if m:
            yy = int(m.group(1))
            mm = int(m.group(2))
            # 动态阈值：以当前年份+10为界，避免固定 50 在 2050 年后出错
            current_yy = pd.Timestamp.now().year % 100
            cutoff = (current_yy + 10) % 100
            year = 2000 + yy if yy <= cutoff else 1900 + yy
            return year, mm
        return 2099, 13

    def _load_one_product(self, product_symbol: str) -> pd.DataFrame:
        """加载单个品种的数据"""
        pe_info = PRODUCT_EXCHANGE_MAP.get(product_symbol)
        if pe_info is None:
            raise ValueError(f"未知品种: {product_symbol}")

        exchange_id, product_code = pe_info

        # 查询合约列表
        quotes = self._api.query_quotes(
            ins_class="FUTURE",
            product_id=product_code,
            exchange_id=exchange_id,
            expired=True,
        )
        self._api.wait_update(deadline=10)

        if not quotes:
            raise RuntimeError(f"未查到 {product_symbol} 的合约")

        # 按到期月份排序，取最近的 N 个合约
        sorted_contracts = sorted(quotes, key=self._parse_contract_month)[:_MAX_CONTRACTS_PER_PRODUCT]
        _logger.info("%s: 找到 %d 个合约", product_symbol, len(sorted_contracts))

        contract_dfs = []
        for ins_id in sorted_contracts:
            try:
                klines = self._api.get_kline_serial(
                    ins_id, _DAILY_SECONDS, data_length=self._data_length
                )
                self._api.wait_update(deadline=5)

                close_series = klines["close"]
                if len(close_series.dropna()) == 0:
                    continue

                df_contract = pd.DataFrame({
                    "date": pd.to_datetime(klines["datetime"], unit="ns", errors="coerce"),
                    "symbol": ins_id,
                    "product": product_code,
                    "open": klines["open"].astype(float),
                    "high": klines["high"].astype(float),
                    "low": klines["low"].astype(float),
                    "close": klines["close"].astype(float),
                    "volume": klines["volume"].astype(float),
                    "open_interest": klines["open_oi"].astype(float),
                })
                df_contract = df_contract.dropna(subset=["date", "close"])
                if not df_contract.empty:
                    contract_dfs.append(df_contract)
            except Exception as e:
                self.load_errors.append({
                    "symbol": product_symbol,
                    "contract": ins_id,
                    "error": f"加载合约K线失败: {e}",
                })
                warnings.warn(f"  加载合约 {ins_id} 失败: {e}")
                continue

        if not contract_dfs:
            raise RuntimeError(f"{product_symbol} 没有可用数据")

        product_df = pd.concat(contract_dfs, ignore_index=True)
        _logger.info("%s: 加载 %d 条记录", product_symbol, len(product_df))
        return product_df

    def load_from_tqsdk(self, show_progress: bool = True):
        """
        从 TqSdk 加载数据（带缓存和重连）。
        """
        if not self._phone or not self._password:
            raise ValueError(
                "load_from_tqsdk() 需要提供 phone 和 password，"
                "请在初始化 DataLoader 时传入"
            )

        _logger.info("=" * 60)
        _logger.info("TqSdk 数据加载")
        _logger.info("=" * 60)

        all_dfs = []
        self.load_errors = []

        # 品种循环迭代器（如果启用进度条则包装 tqdm）
        symbols_iter = self._symbols
        if show_progress:
            try:
                from tqdm import tqdm
                symbols_iter = tqdm(self._symbols, desc="加载品种", unit="品种")
            except ImportError:
                pass

        for symbol in symbols_iter:
            # 尝试从缓存加载
            cached = self._load_from_cache(symbol)
            if cached is not None:
                all_dfs.append(cached)
                continue

            # 缓存未命中或失效，从 TqSdk 加载
            try:
                self._create_api()
                df = self._load_one_product(symbol)
                self._save_to_cache(symbol, df)
                all_dfs.append(df)
                self._close_api()
            except Exception as e:
                self.load_errors.append({"symbol": symbol, "error": str(e)})
                warnings.warn(f"加载 {symbol} 失败: {e}")
                continue

        if not all_dfs:
            error_detail = "\n".join(
                f"  - {e['symbol']}: {e['error']}" for e in self.load_errors
            )
            raise RuntimeError(f"TqSdk 未加载到任何有效数据:\n{error_detail}")

        self.all_contracts = pd.concat(all_dfs, ignore_index=True)
        self.all_contracts = self.all_contracts.sort_values(["date", "symbol"]).reset_index(drop=True)
        self.data_mode = "contract"
        self._product_symbols = None

        if self.load_errors:
            warnings.warn(
                f"部分品种/合约加载失败 ({len(self.load_errors)}/{len(self._symbols)}): "
                + "; ".join(
                    f"{e.get('symbol', '?')}"
                    f"{('/' + e['contract']) if 'contract' in e else ''}"
                    f": {e['error']}"
                    for e in self.load_errors
                )
            )

        _logger.info("汇总: 共加载 %d 行数据", len(self.all_contracts))
        return self.all_contracts

    # ------------------------------------------------------------------
    # CSV 相关方法
    # ------------------------------------------------------------------

    def _detect_format_from_df(self, df: pd.DataFrame) -> str:
        """
        基于 DataFrame 列名检测数据格式。
        """
        cols = set(df.columns)
        contract_cols = set(self.CONTRACT_REQUIRED_COLUMNS)
        product_cols = {"datetime", "open", "high", "low", "close"}

        if contract_cols.issubset(cols):
            return "contract"
        elif product_cols.issubset(cols):
            return "product"
        elif "date" in cols and "symbol" in cols:
            return "contract"
        else:
            return "product"

    def _load_product_csv(
        self, filepath: str, df: Optional[pd.DataFrame] = None
    ) -> Optional[pd.DataFrame]:
        """
        加载品种汇总格式的CSV文件。
        """
        try:
            if df is None:
                df = pd.read_csv(filepath)
            required = {"datetime", "open", "high", "low", "close"}
            if not required.issubset(set(df.columns)):
                return None

            rename_map = {"datetime": "date"}
            if "position" in df.columns:
                rename_map["position"] = "open_interest"
            elif "open_interest" not in df.columns:
                df["open_interest"] = 0
                warnings.warn(
                    f"{filepath}: 缺少 position/open_interest 列，"
                    "主力合约识别将回退到 volume"
                )

            df = df.rename(columns=rename_map)

            if "symbol" not in df.columns:
                basename = os.path.basename(filepath).replace(".csv", "")
                df["symbol"] = basename

            if "volume" not in df.columns:
                df["volume"] = 0

            standard_cols = [
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
            ]
            available_cols = [c for c in standard_cols if c in df.columns]
            df = df[available_cols].copy()

            for col in ["open", "high", "low", "close", "volume", "open_interest"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            return df
        except Exception as e:
            _logger.warning("加载 CSV 文件失败: %s", e)
            return None

    def _try_load_csv_file(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        尝试加载单个CSV文件并转为标准格式。

        根据检测到的格式（contract/product）自动处理，
        加载失败将错误记录到 self.load_errors。

        Returns:
            标准格式的 DataFrame，加载失败返回 None
        """
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                self.load_errors.append({"file": filepath, "error": "文件为空"})
                return None

            fmt = self._detect_format_from_df(df)
            if self.data_mode is None:
                self.data_mode = fmt

            if fmt == "contract":
                missing = set(self.CONTRACT_REQUIRED_COLUMNS) - set(df.columns)
                if missing:
                    self.load_errors.append(
                        {"file": filepath, "error": f"缺少必要列: {missing}"}
                    )
                    return None
                return df
            else:
                loaded = self._load_product_csv(filepath, df)
                if loaded is not None and not loaded.empty:
                    return loaded
                self.load_errors.append(
                    {"file": filepath, "error": "品种格式转换失败"}
                )
                return None
        except Exception as e:
            self.load_errors.append({"file": filepath, "error": str(e)})
            return None

    def _finalize_loaded_dfs(self, dfs: list, file_count: int) -> pd.DataFrame:
        """
        对已加载的 DataFrame 列表进行后处理：
        合并、日期转换、排序、添加 product 列、赋值 self.all_contracts。
        """
        if not dfs:
            error_summary = "\n".join(
                f"  - {e['file']}: {e['error']}" for e in self.load_errors
            )
            raise ValueError(f"没有成功加载任何CSV文件。加载错误:\n{error_summary}")

        if self.load_errors:
            warnings.warn(
                f"部分文件加载失败 ({len(self.load_errors)}/{file_count}): "
                + "; ".join(f"{e['file']}: {e['error']}" for e in self.load_errors)
            )

        combined = pd.concat(dfs, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        invalid_dates = combined["date"].isna()
        if invalid_dates.any():
            warnings.warn(f"发现 {invalid_dates.sum()} 行无效日期，已剔除")
            combined = combined[~invalid_dates]

        combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)

        if self.data_mode == "product":
            combined["product"] = combined["symbol"]
        else:
            product_extracted = combined["symbol"].str.extract(r"^([A-Za-z]+)")[0]
            combined["product"] = product_extracted.fillna(combined["symbol"])

        self._product_symbols = None
        self.all_contracts = combined
        return combined

    def load_csv_files(self, file_pattern: str = "*.csv") -> pd.DataFrame:
        """
        读取目录下所有匹配的CSV文件并合并。
        """
        if not self.data_dir or not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        pattern = os.path.join(self.data_dir, file_pattern)
        files = sorted(glob.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"未找到匹配 '{file_pattern}' 的CSV文件: {self.data_dir}"
            )

        self.load_errors = []
        self.data_mode = None
        dfs = []

        for f in files:
            loaded = self._try_load_csv_file(f)
            if loaded is not None:
                dfs.append(loaded)

        return self._finalize_loaded_dfs(dfs, len(files))

    def load_csv_files_by_paths(self, file_paths: list) -> pd.DataFrame:
        """
        按指定文件路径加载 CSV 文件（仅加载目标品种）。

        Args:
            file_paths: CSV 文件路径列表
        """
        self.load_errors = []
        self.data_mode = None
        dfs = []

        for f in file_paths:
            loaded = self._try_load_csv_file(f)
            if loaded is not None:
                dfs.append(loaded)

        return self._finalize_loaded_dfs(dfs, len(file_paths))

    # ------------------------------------------------------------------
    # 统一加载方法
    # ------------------------------------------------------------------

    def load_data(self, file_pattern: str = "*.csv", show_progress: bool = True):
        """
        统一加载数据的方法。

        Args:
            file_pattern: 文件匹配模式（仅 CSV 模式）
            show_progress: 是否显示进度条（仅 TqSdk 模式）

        Returns:
            合并后的 DataFrame
        """
        if self.data_source == "tqsdk":
            return self.load_from_tqsdk(show_progress=show_progress)
        elif self.data_source == "csv":
            return self.load_csv_files(file_pattern=file_pattern)
        else:
            raise ValueError(f"不支持的数据源类型: {self.data_source}")

    # ------------------------------------------------------------------
    # 主力合约识别
    # ------------------------------------------------------------------

    def identify_dominant_contracts(self, method: str = "open_interest") -> pd.Series:
        """
        每日按品种识别主力合约。

        对每个品种的每个交易日，分别选出持仓量最大的合约作为主力合约，
        而非跨品种选出一个。这样每个品种每天都有自己的主力合约。
        """
        if self.all_contracts is None:
            raise RuntimeError("请先调用 load_data() 加载数据")

        if self.data_mode == "product":
            # 品种模式：每行就是一个品种的数据，全部是主力
            idx = self.all_contracts.groupby("date")["open_interest"].idxmax()
            dominant = self.all_contracts.loc[idx, ["date", "symbol"]].set_index(
                "date"
            )["symbol"]
        else:
            # 合约模式：按 (date, product) 分组，每个品种每天选一个主力
            df = self.all_contracts.sort_values(["date", "symbol"]).reset_index(drop=True)

            if "product" not in df.columns:
                # 无 product 列时回退到跨品种识别
                warnings.warn("数据缺少 product 列，回退到跨品种识别主力合约")
                idx = df.groupby("date")["open_interest"].idxmax()
                dominant = df.loc[idx, ["date", "symbol"]].set_index("date")["symbol"]
            elif method == "open_interest":
                # 按 (date, product) 分组，每组选持仓量最大的合约
                oi_idx = df.groupby(["date", "product"])["open_interest"].idxmax()
                oi_dominant = df.loc[oi_idx, ["date", "product", "symbol", "open_interest"]].copy()

                # 找出 open_interest 为 0 的记录 → 回退到 volume
                zero_oi_mask = oi_dominant["open_interest"] == 0
                if zero_oi_mask.any():
                    zero_oi_keys = oi_dominant[zero_oi_mask][["date", "product"]]
                    for _, row in zero_oi_keys.iterrows():
                        mask = (df["date"] == row["date"]) & (df["product"] == row["product"])
                        candidates = df[mask]
                        if len(candidates) > 0:
                            best = candidates.loc[candidates["volume"].idxmax()]
                            oi_dominant.loc[
                                (oi_dominant["date"] == row["date"]) & (oi_dominant["product"] == row["product"]),
                                "symbol"
                            ] = best["symbol"]

                dominant = oi_dominant.set_index("date")["symbol"]

            elif method == "volume":
                idx = df.groupby(["date", "product"])["volume"].idxmax()
                dominant = df.loc[idx, ["date", "symbol"]].set_index("date")["symbol"]
            else:
                raise ValueError(f"不支持的识别方法: {method}")

        self.dominant_map = dominant
        return dominant

    # ------------------------------------------------------------------
    # 连续序列构建（含展期标记）
    # ------------------------------------------------------------------

    def build_continuous_series(self) -> pd.DataFrame:
        """
        构建展期法连续主力合约序列。
        """
        if self.all_contracts is None:
            raise RuntimeError("请先调用 load_data() 加载数据")
        if self.dominant_map is None:
            self.identify_dominant_contracts()

        df = self.all_contracts.copy(deep=True)

        dominant_df = self.dominant_map.reset_index()
        dominant_df.columns = ["date", "dominant_symbol"]

        # 从 dominant_symbol 提取 product，用于按品种匹配
        if "product" in df.columns:
            dominant_df["product"] = dominant_df["dominant_symbol"].str.extract(
                r"\.([A-Za-z]+)\d", expand=False
            )
            # 按 (date, product) 匹配，避免笛卡尔积
            df = df.merge(dominant_df, on=["date", "product"], how="left")
        else:
            df = df.merge(dominant_df, on="date", how="left")

        # 按品种计算 prev_dominant_symbol
        if "product" in df.columns:
            df["prev_dominant_symbol"] = df.groupby("product")["dominant_symbol"].shift(1)
        else:
            df["prev_dominant_symbol"] = df["dominant_symbol"].shift(1)

        # 如果是品种模式，所有都是主力合约
        if self.data_mode == "product":
            df["is_dominant"] = True
        else:
            df["is_dominant"] = df["symbol"] == df["dominant_symbol"]

        df["rollover_flag"] = False
        if self.data_mode == "contract":
            dominant_mask = df["is_dominant"]
            rollover_condition = (
                df.loc[dominant_mask, "dominant_symbol"]
                != df.loc[dominant_mask, "prev_dominant_symbol"]
            ) & df.loc[dominant_mask, "prev_dominant_symbol"].notna()
            idx_to_set = df.index[dominant_mask][rollover_condition]
            df.loc[idx_to_set, "rollover_flag"] = True

        self.full_df = df
        self.continuous_df = df[df["is_dominant"]].copy().reset_index(drop=True)
        return self.full_df

    def build_spread_pairs(self) -> pd.DataFrame:
        """
        构建近远月合约对，将远月收盘价合并到主力合约行中。

        对每个品种的每个交易日：
          - 近月 = 当日主力合约
          - 远月 = 同品种中持仓量第二大的合约（OI 排名=2）

        在 full_df 中新增列：
          - far_symbol: 远月合约代码
          - far_close: 远月收盘价
          - spread: 近月收盘价 - 远月收盘价

        P1 整改（2026-06-10）：原实现 O(N_date × N_contracts) 嵌套循环，
        在 30+ 品种 × 170k+ 行下耗时长。改为 O(N) 向量化：
        groupby(['date', 'product']) + rank() 一次得到 OI 排名，
        提取 OI 排名=2 的行作为远月，再 merge 回全量数据。
        """
        if self.full_df is None:
            self.build_continuous_series()

        df = self.full_df.copy()

        if self.data_mode == "product":
            # 品种模式无法构建跨期价差，添加空列
            df["far_symbol"] = ""
            df["far_close"] = np.nan
            df["spread"] = np.nan
            self.full_df = df
            return df

        # 向量化：按 (date, product) 计算 OI 排名
        # OI 排名=1：主力；OI 排名=2：远月；其余：忽略
        df["_oi_rank_in_dp"] = df.groupby(["date", "product"])["open_interest"].rank(
            method="first", ascending=False
        )

        # 提取远月合约（OI 排名=2）信息
        far_rows = df[df["_oi_rank_in_dp"] == 2][["date", "product", "symbol", "close"]].rename(
            columns={"symbol": "far_symbol", "close": "far_close"}
        )

        # 仅在主力合约行追加远月列（其余行保持 NaN）
        dominant = df["is_dominant"].fillna(False)
        if far_rows.empty:
            # 无远月数据时直接补空列
            df["far_symbol"] = ""
            df["far_close"] = np.nan
            df["spread"] = np.nan
            df["spread"] = np.where(
                dominant, df["close"] - df["far_close"], np.nan
            )
            df = df.drop(columns=["_oi_rank_in_dp"])
            self.full_df = df
            return df

        # 远月信息按 (date, product) merge 到全量数据
        df = df.merge(far_rows, on=["date", "product"], how="left")
        # 非主力合约行的 far_close/far_symbol 强制 NaN（避免误用远月数据）
        df.loc[~dominant, "far_close"] = np.nan
        df.loc[~dominant, "far_symbol"] = ""
        # spread：仅主力合约行有效
        df["spread"] = np.where(
            dominant & df["far_close"].notna(),
            df["close"] - df["far_close"],
            np.nan,
        )

        df = df.drop(columns=["_oi_rank_in_dp"])
        self.full_df = df
        return df

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
        # 仅在合约模式（symbol 含合约号如 rb2401）时替换，品种模式已是品种级
        if "product" in dominant_df.columns and self.data_mode == "contract":
            # 构建品种级 symbol：交易所.品种（如 SHFE.RB），品种统一大写
            if "symbol" in dominant_df.columns:
                dominant_df["exchange"] = dominant_df["symbol"].str.split(".").str[0]
                dominant_df["symbol"] = dominant_df["exchange"] + "." + dominant_df["product"].str.upper()
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

    def get_product_symbols(self, product: Optional[str] = None) -> Dict[str, List[str]]:
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
        mask = (df["symbol"] == symbol) & (df["date"] >= start_date) & (df["date"] <= end_date)
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
