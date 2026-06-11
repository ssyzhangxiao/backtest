"""数据加载器 — TqSdk 数据源 Mixin。

提供 TqSdk 连接管理、缓存读写、品种加载等方法。
"""

import logging
import pickle
import re
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from core.data._constants import (
    CACHE_DIR,
    DAILY_SECONDS,
    MAX_CONTRACTS_PER_PRODUCT,
    PRODUCT_EXCHANGE_MAP,
)

_logger = logging.getLogger(__name__)


class TqsdkMixin:
    """TqSdk 数据源相关方法。"""

    # 以下属性由 DataLoader.__init__ 设置，此处声明类型供类型检查器使用
    _enable_cache: bool
    _cache_ttl_hours: int
    _data_length: int
    _max_reconnect_attempts: int
    _reconnect_delay_seconds: float
    _phone: Optional[str]
    _password: Optional[str]
    _api: object
    _symbols: list
    load_errors: list

    def _get_cache_path(self, symbol: str) -> Path:
        """获取品种数据的缓存路径

        Cache key 包含 data_length 和 MAX_CONTRACTS_PER_PRODUCT，
        调整任一参数都会导致旧缓存失效（这是预期行为）。
        """
        sanitized = symbol.replace(".", "_").replace("-", "_")
        return (
            CACHE_DIR
            / f"{sanitized}_{self._data_length}_{MAX_CONTRACTS_PER_PRODUCT}.pkl"
        )

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
                    attempt + 1,
                    self._max_reconnect_attempts + 1,
                )
                self._api = TqApi(auth=TqAuth(self._phone, self._password))
                _logger.info("TqSdk 连接成功")
                return
            except PermissionError as e:
                raise PermissionError(f"TqSdk 认证失败，请检查账号密码: {e}") from e
            except Exception as e:
                if attempt < self._max_reconnect_attempts:
                    delay = self._reconnect_delay_seconds * (2**attempt)
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
        m = re.search(r"(\d{2})(\d{2})$", symbol)
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

        # 按到期月份降序排序，取最近的 N 个合约
        sorted_contracts = sorted(quotes, key=self._parse_contract_month, reverse=True)[
            :MAX_CONTRACTS_PER_PRODUCT
        ]
        _logger.info(
            "%s: 找到 %d 个合约 (最旧=%s, 最新=%s)",
            product_symbol,
            len(sorted_contracts),
            sorted_contracts[-1],
            sorted_contracts[0],
        )

        contract_dfs = []
        for ins_id in sorted_contracts:
            try:
                klines = self._api.get_kline_serial(
                    ins_id, DAILY_SECONDS, data_length=self._data_length
                )
                self._api.wait_update(deadline=5)

                close_series = klines["close"]
                if len(close_series.dropna()) == 0:
                    continue

                df_contract = pd.DataFrame(
                    {
                        "date": pd.to_datetime(
                            klines["datetime"], unit="ns", errors="coerce"
                        ),
                        "symbol": ins_id,
                        "product": product_code,
                        "open": klines["open"].astype(float),
                        "high": klines["high"].astype(float),
                        "low": klines["low"].astype(float),
                        "close": klines["close"].astype(float),
                        "volume": klines["volume"].astype(float),
                        "open_interest": klines["open_oi"].astype(float),
                    }
                )
                df_contract = df_contract.dropna(subset=["date", "close"])
                if not df_contract.empty:
                    contract_dfs.append(df_contract)
            except Exception as e:
                self.load_errors.append(
                    {
                        "symbol": product_symbol,
                        "contract": ins_id,
                        "error": f"加载合约K线失败: {e}",
                    }
                )
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

        .. deprecated::
            将于 v0.2.0 删除（规则 22 迁移阶段 3）。
            新接口：``TqsdkAdapter.load()``。

        Returns:
            DataFrame
        """
        warnings.warn(
            "DataLoader.load_from_tqsdk() 已废弃（规则 22），"
            "请改用 TqsdkAdapter.load() —— create_data_source('tqsdk', ...)",
            DeprecationWarning,
            stacklevel=2,
        )
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
        self.all_contracts = self.all_contracts.sort_values(
            ["date", "symbol"]
        ).reset_index(drop=True)
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
