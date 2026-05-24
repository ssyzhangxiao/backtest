"""
TqSdk 数据源适配器 — 独立合约 + 展期模式。

从 TqSdk 直接获取真实交易所合约数据，支持每日主力合约识别和展期信号标记。
输出格式兼容 PyBroker 和展期模块。

输出列: date, symbol, open, high, low, close, volume, open_interest,
         is_dominant, dominant_symbol, prev_dominant_symbol, rollover_flag, product

使用方式:
    loader = TqSdkDataSource(phone="...", password="...",
                             symbols=["SHFE.RB", "DCE.M", "CZCE.TA"])
    loader.load_from_tqsdk()
    loader.identify_dominant_contracts()
    loader.build_continuous_series()
    df = loader.get_pybroker_df()
"""

import re
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# 独立合约模式：品种 → (TqSdk 交易所, TqSdk 品种代码) 映射
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

# 默认加载的核心品种
_DEFAULT_SYMBOLS = [
    "SHFE.RB", "SHFE.HC", "SHFE.AU", "SHFE.AG", "SHFE.CU",
    "DCE.M", "DCE.I", "DCE.J", "DCE.JM", "DCE.C",
    "DCE.P", "DCE.Y", "DCE.EG", "DCE.PP", "DCE.L",
    "CZCE.TA", "CZCE.MA", "CZCE.FG", "CZCE.SA", "CZCE.CF",
    "CZCE.OI", "CZCE.RM", "CZCE.SR", "CZCE.ZC",
    "CFFEX.IF", "CFFEX.IC", "CFFEX.IH",
    "INE.SC", "INE.NR",
]

_DAILY_SECONDS = 86400  # 86400 秒 = 1 天，用于 TqSdk K 线日线周期


class TqSdkDataSource:
    """
    TqSdk 独立合约数据源。

    对于每个品种，查询 TqSdk 获取所有上市合约 → 按到期月份排序取最近合约
    → 下载 K 线 → 识别每日主力合约 → 构建连续序列并标记展期信号。

    phone / password 是数据加载的必要参数，若不提供则无法调用 load_from_tqsdk()。

    Attributes:
        data_mode: 模式标识，固定为 'contract'
        all_contracts: 合并的原始数据
        dominant_map: 每日主力合约映射 {date: symbol}
        continuous_df: 主力合约连续序列
        full_df: 完整数据（含 is_dominant, rollover_flag 等辅助列）
        load_errors: 加载错误列表
    """

    PYBROKER_COLUMNS = [
        "date", "symbol", "open", "high", "low", "close",
        "volume", "open_interest",
        "is_dominant", "dominant_symbol", "prev_dominant_symbol",
        "rollover_flag", "product",
    ]

    # 每个品种最多下载的合约数（按到期月份排序后取最近的 N 个）
    MAX_CONTRACTS_PER_PRODUCT = 20

    def __init__(
        self,
        phone: Optional[str] = None,
        password: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        data_length: int = 2000,
    ):
        """
        Args:
            phone: 快期账号手机号。load_from_tqsdk() 的必要参数，不提供则无法加载数据。
            password: 快期账号密码。load_from_tqsdk() 的必要参数，不提供则无法加载数据。
            symbols: 品种代码列表，默认加载核心品种
            data_length: 每个合约下载的 K 线数量（日线），
                        2000 可覆盖约 6 年历史，5000 约 10 年
        """
        self._phone = phone
        self._password = password
        self._symbols = symbols or _DEFAULT_SYMBOLS
        self._data_length = data_length
        self._api = None

        self.all_contracts: Optional[pd.DataFrame] = None
        self.dominant_map: Optional[pd.Series] = None
        self.continuous_df: Optional[pd.DataFrame] = None
        self.full_df: Optional[pd.DataFrame] = None
        self.data_mode: str = "contract"
        self.load_errors: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    # 解析合约代码中的到期月份（如 "SHFE.rb2505" → (2025, 5)）
    @staticmethod
    def _parse_contract_month(symbol: str) -> Tuple[int, int]:
        """从合约代码中提取年份和月份。"""
        # 匹配末尾的数字部分，如 rb2505 → (2025, 5), TA505 → (2025, 5)
        m = re.search(r'(\d{2})(\d{2})$', symbol)
        if m:
            yy = int(m.group(1))
            mm = int(m.group(2))
            # 00-50 → 2000-2050, 50-99 → 1950-1999（已退市合约也可能存在）
            year = 2000 + yy if yy < 50 else 1900 + yy
            return year, mm
        return 2099, 13  # 无法解析的排在最后

    def load_from_tqsdk(
        self, show_progress: bool = True
    ) -> pd.DataFrame:
        """
        从 TqSdk 加载所有品种的独立合约 K 线。

        phone 和 password 是必要参数，若未提供则抛出 ValueError。

        Args:
            show_progress: 是否使用 tqdm 展示品种加载进度条（默认 True）

        Returns:
            合并后的 DataFrame
        """
        if not self._phone or not self._password:
            raise ValueError(
                "load_from_tqsdk() 需要提供 phone 和 password，"
                "请先设置 TqSdk 账号"
            )

        from tqsdk import TqApi, TqAuth

        self.load_errors = []
        api = TqApi(auth=TqAuth(self._phone, self._password))
        self._api = api

        dfs = []

        # 品种循环迭代器（如果启用进度条则包装 tqdm）
        symbols_iter = self._symbols
        if show_progress:
            try:
                from tqdm import tqdm
                symbols_iter = tqdm(self._symbols, desc="加载品种", unit="品种")
            except ImportError:
                pass

        for proj_sym in symbols_iter:
            pe_info = PRODUCT_EXCHANGE_MAP.get(proj_sym)
            if pe_info is None:
                self.load_errors.append({
                    "symbol": proj_sym,
                    "error": "未找到对应 PRODUCT_EXCHANGE_MAP 映射",
                })
                continue

            exchange_id, product_code = pe_info

            try:
                quotes = api.query_quotes(
                    ins_class="FUTURE",
                    product_id=product_code,
                    exchange_id=exchange_id,
                    expired=True,
                )
                api.wait_update(deadline=5)

                if not quotes:
                    self.load_errors.append({
                        "symbol": proj_sym,
                        "error": "未从 TqSdk 查到任何合约",
                    })
                    continue

                # 按到期月份排序，取最近 N 个合约（避免取到远月无量合约）
                sorted_contracts = sorted(
                    list(quotes), key=self._parse_contract_month
                )
                top_contracts = sorted_contracts[:self.MAX_CONTRACTS_PER_PRODUCT]

                contract_dfs = []
                for ins_id in top_contracts:
                    try:
                        klines = api.get_kline_serial(
                            ins_id, _DAILY_SECONDS, data_length=self._data_length
                        )

                        close_series = klines["close"]
                        if len(close_series.dropna()) == 0:
                            continue

                        # 直接使用纳秒级时间戳解析，丢弃解析失败的行
                        date_series = pd.to_datetime(
                            klines["datetime"], unit="ns", errors="coerce"
                        )

                        product = proj_sym.split(".")[1] if "." in proj_sym else proj_sym

                        df_contract = pd.DataFrame({
                            "date":          date_series,
                            "symbol":        ins_id,
                            "product":       product,
                            "open":          klines["open"].astype(float),
                            "high":          klines["high"].astype(float),
                            "low":           klines["low"].astype(float),
                            "close":         klines["close"].astype(float),
                            "volume":        klines["volume"].astype(float),
                            "open_interest": klines["open_oi"].astype(float),
                        })

                        # 丢弃日期或收盘价无效的行
                        df_contract = df_contract.dropna(
                            subset=["date", "close"]
                        ).reset_index(drop=True)

                        if not df_contract.empty:
                            contract_dfs.append(df_contract)
                    except Exception as e:
                        self.load_errors.append({
                            "symbol": proj_sym,
                            "contract": ins_id,
                            "error": f"加载合约K线失败: {e}",
                        })

                if contract_dfs:
                    dfs.append(pd.concat(contract_dfs, ignore_index=True))
                else:
                    self.load_errors.append({
                        "symbol": proj_sym,
                        "error": "所有合约K线为空",
                    })

            except Exception as e:
                self.load_errors.append({
                    "symbol": proj_sym,
                    "error": str(e),
                })

        if not dfs:
            error_detail = "\n".join(
                f"  - {e['symbol']}: {e['error']}" for e in self.load_errors
            )
            raise RuntimeError(f"TqSdk 未加载到任何有效数据:\n{error_detail}")

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)

        self.all_contracts = combined

        if self.load_errors:
            warnings.warn(
                f"部分品种/合约加载失败 ({len(self.load_errors)}/{len(self._symbols)}): "
                + "; ".join(
                    f"{e.get('symbol', '?')}"
                    f"{'/' + e['contract'] if 'contract' in e else ''}"
                    f": {e['error']}"
                    for e in self.load_errors
                )
            )

        return combined

    # ------------------------------------------------------------------
    # 主力合约识别
    # ------------------------------------------------------------------

    def identify_dominant_contracts(self, method: str = "open_interest") -> pd.Series:
        """
        每日识别主力合约。

        对于每个交易日：
        1. 优先使用 open_interest，选持仓量最大的合约
        2. 若该日所有合约 open_interest 均为 0，回退到 volume
        3. 若 volume 也全为 0，则选该日第一个合约（按 symbol 排序）并发出警告

        Args:
            method: 'open_interest' 或 'volume'

        Returns:
            主力合约映射 Series（索引=日期, 值=合约代码）
        """
        if self.all_contracts is None:
            raise RuntimeError("请先调用 load_from_tqsdk() 加载数据")

        # 按 date, symbol 排序，确保"第一个合约"有确定顺序
        df = self.all_contracts.sort_values(["date", "symbol"]).reset_index(drop=True)

        if method == "open_interest":
            # 先按持仓量识别主力
            oi_idx = df.groupby("date")["open_interest"].idxmax()
            oi_dominant = df.loc[oi_idx, ["date", "symbol", "open_interest"]].copy()

            # 找出 open_interest 为 0 的日期 → 回退到 volume
            zero_oi_dates = oi_dominant[oi_dominant["open_interest"] == 0]["date"].unique()

            if len(zero_oi_dates) > 0:
                # 对这些日期按 volume 重新识别
                vol_mask = df["date"].isin(zero_oi_dates)
                if vol_mask.any():
                    vol_idx = (
                        df[vol_mask].groupby("date")["volume"].idxmax()
                    )
                    vol_dominant = df.loc[vol_idx, ["date", "symbol", "volume"]].copy()

                    # 找出 volume 也为 0 的日期 → 回退到第一个合约
                    zero_vol_dates = vol_dominant[
                        vol_dominant["volume"] == 0
                    ]["date"].unique()

                    if len(zero_vol_dates) > 0:
                        # 按 symbol 排序后取每组第一个合约
                        first_mask = df["date"].isin(zero_vol_dates)
                        first_contracts = (
                            df[first_mask].groupby("date")["symbol"].first()
                        )
                        warnings.warn(
                            f"{len(zero_vol_dates)} 个交易日的持仓量和成交量均为零，"
                            f"已回退到按 symbol 排序的第一个合约作为主力"
                        )
                        # 更新 vol_dominant 中这些日期的 symbol
                        for dt, sym in first_contracts.items():
                            vol_dominant.loc[
                                vol_dominant["date"] == dt, "symbol"
                            ] = sym

                    # 将 volume 回退结果更新到 oi_dominant
                    for dt in zero_oi_dates:
                        if dt in vol_dominant["date"].values:
                            oi_dominant.loc[
                                oi_dominant["date"] == dt, "symbol"
                            ] = vol_dominant.loc[
                                vol_dominant["date"] == dt, "symbol"
                            ].values[0]

            dominant = oi_dominant.set_index("date")["symbol"]

        elif method == "volume":
            idx = df.groupby("date")["volume"].idxmax()
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

        添加 is_dominant、dominant_symbol、prev_dominant_symbol、rollover_flag 列。

        Returns:
            带辅助列的完整 DataFrame
        """
        if self.all_contracts is None:
            raise RuntimeError("请先调用 load_from_tqsdk() 加载数据")
        if self.dominant_map is None:
            self.identify_dominant_contracts()

        df = self.all_contracts.copy(deep=True)

        dominant_df = self.dominant_map.reset_index()
        dominant_df.columns = ["date", "dominant_symbol"]
        dominant_df["prev_dominant_symbol"] = dominant_df["dominant_symbol"].shift(1)

        df = df.merge(dominant_df, on="date", how="left")
        df["is_dominant"] = df["symbol"] == df["dominant_symbol"]

        df["rollover_flag"] = False
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

    # ------------------------------------------------------------------
    # 输出接口
    # ------------------------------------------------------------------

    def get_pybroker_df(self) -> pd.DataFrame:
        """
        获取 PyBroker 兼容格式的 DataFrame。

        必须包含 ['date', 'symbol', 'close'] 三列，否则抛出 ValueError。

        Returns:
            PyBroker 可直接使用的 DataFrame

        Raises:
            ValueError: 如果缺失必要的列
        """
        if self.full_df is None:
            self.build_continuous_series()

        required = ["date", "symbol", "close"]
        missing = [c for c in required if c not in self.full_df.columns]
        if missing:
            raise ValueError(
                f"PyBroker 输出缺少必要列: {missing}。"
                f"当前列: {list(self.full_df.columns)}"
            )

        available = [c for c in self.PYBROKER_COLUMNS if c in self.full_df.columns]
        result = self.full_df[available].copy()
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

        if "rollover_flag" not in self.full_df.columns:
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

    def get_data_summary(self) -> Dict:
        """获取数据摘要。"""
        if self.all_contracts is None:
            return {"status": "未加载数据"}

        products = self.product_symbols
        return {
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
            "load_errors": self.load_errors if self.load_errors else None,
        }

    @property
    def product_symbols(self) -> Dict[str, List[str]]:
        """获取品种列表 {品种: [合约代码列表]}。"""
        if self.all_contracts is None:
            raise RuntimeError("请先调用 load_from_tqsdk() 加载数据")
        return {
            p: sorted(g["symbol"].unique().tolist())
            for p, g in self.all_contracts.groupby("product")
        }

    def get_product_symbols(self, product: Optional[str] = None) -> Dict[str, List[str]]:
        """获取指定品种的合约代码列表。"""
        ps = self.product_symbols
        if product:
            return {product: ps.get(product, [])}
        return ps

    def close(self):
        """关闭 TqSdk 连接。"""
        if self._api is not None:
            try:
                self._api.close()
            except Exception:
                pass
            self._api = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()