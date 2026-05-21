"""
展期法数据加载模块。

支持两种数据格式：
1. 合约模式：按合约分文件（如 RB2310.csv），包含具体合约数据，支持展期
2. 品种模式：按品种分文件（如 SHFE.RB.csv），品种连续指数数据，无展期

输出格式兼容 PyBroker：包含 date, symbol, open, high, low, close, volume, open_interest,
以及辅助列 is_dominant, dominant_symbol, product。
"""

import os
import glob
import warnings
import pandas as pd
from typing import Optional, List, Dict


class DataLoader:
    """
    展期法数据加载器。

    负责读取本地CSV期货数据，识别主力合约，生成展期法连续序列，
    输出 PyBroker 兼容的 DataFrame。

    自动检测数据格式：
    - 如果CSV包含 symbol 列且 symbol 值为具体合约代码（如 RB2310），使用合约模式
    - 如果CSV包含 symbol 列且 symbol 值为品种代码（如 SHFE.RB），使用品种模式

    Attributes:
        data_dir: CSV文件所在目录路径
        all_contracts: 所有合约的原始数据（合并后）
        dominant_map: 每日主力合约映射 {date: symbol}
        continuous_df: 展期法连续主力合约数据
        full_df: 包含所有合约的完整数据（用于跨期策略）
        data_mode: 数据模式，'contract' 或 'product'
        load_errors: 加载失败的文件及错误信息列表
    """

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

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.all_contracts: Optional[pd.DataFrame] = None
        self.dominant_map: Optional[pd.Series] = None
        self.continuous_df: Optional[pd.DataFrame] = None
        self.full_df: Optional[pd.DataFrame] = None
        self.data_mode: Optional[str] = None
        self.load_errors: List[Dict[str, str]] = []
        self._product_symbols: Optional[Dict[str, List[str]]] = None

    def _detect_format_from_df(self, df: pd.DataFrame) -> str:
        """
        基于 DataFrame 列名检测数据格式。

        Args:
            df: 已读取的 DataFrame

        Returns:
            'contract' 或 'product'
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

    def load_csv_files(self, file_pattern: str = "*.csv") -> pd.DataFrame:
        """
        读取目录下所有匹配的CSV文件并合并。

        自动检测数据格式并使用对应的加载逻辑。

        Args:
            file_pattern: 文件匹配模式，默认 '*.csv'

        Returns:
            合并后的 DataFrame，包含所有合约数据

        Raises:
            FileNotFoundError: 当目录不存在或无匹配文件时
            ValueError: 当CSV缺少必要列时
        """
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        pattern = os.path.join(self.data_dir, file_pattern)
        files = sorted(glob.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"未找到匹配 '{file_pattern}' 的CSV文件: {self.data_dir}"
            )

        self.load_errors = []
        first_format = None
        dfs = []

        for f in files:
            try:
                df = pd.read_csv(f)
                if df.empty:
                    self.load_errors.append({"file": f, "error": "文件为空"})
                    continue

                fmt = self._detect_format_from_df(df)
                if first_format is None:
                    first_format = fmt
                    self.data_mode = first_format

                if fmt == "contract":
                    missing = set(self.CONTRACT_REQUIRED_COLUMNS) - set(df.columns)
                    if missing:
                        self.load_errors.append(
                            {"file": f, "error": f"缺少必要列: {missing}"}
                        )
                        continue
                    dfs.append(df)
                else:
                    loaded = self._load_product_csv(f, df)
                    if loaded is not None and not loaded.empty:
                        dfs.append(loaded)
                    else:
                        self.load_errors.append(
                            {"file": f, "error": "品种格式转换失败"}
                        )
            except Exception as e:
                self.load_errors.append({"file": f, "error": str(e)})
                continue

        if not dfs:
            error_summary = "\n".join(
                f"  - {e['file']}: {e['error']}" for e in self.load_errors
            )
            raise ValueError(f"没有成功加载任何CSV文件。加载错误:\n{error_summary}")

        if self.load_errors:
            warnings.warn(
                f"部分文件加载失败 ({len(self.load_errors)}/{len(files)}): "
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
            combined["product"] = combined["symbol"].str.extract(r"^([A-Za-z]+)")[0]

        self._product_symbols = None
        self.all_contracts = combined
        return combined

    def _load_product_csv(
        self, filepath: str, df: Optional[pd.DataFrame] = None
    ) -> Optional[pd.DataFrame]:
        """
        加载品种汇总格式的CSV文件。

        品种格式列：datetime, open, high, low, close, volume, amount, position, symbol
        转换为标准格式：date, symbol, open, high, low, close, volume, open_interest

        Args:
            filepath: CSV文件路径
            df: 可选，已读取的 DataFrame（避免二次读取）

        Returns:
            标准格式的 DataFrame，或 None
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
        except Exception:
            return None

    def identify_dominant_contracts(self, method: str = "open_interest") -> pd.Series:
        """
        每日识别主力合约。

        在品种模式下，每个品种只有一条记录，自动为主力。
        在合约模式下，以持仓量最大的合约作为主力合约；
        若持仓量全为零，回退到成交量（volume）识别。

        Args:
            method: 识别方法，'open_interest' 或 'volume'

        Returns:
            Series，索引为日期，值为主力合约代码
        """
        if self.all_contracts is None:
            raise RuntimeError("请先调用 load_csv_files() 加载数据")

        if self.data_mode == "product":
            idx = self.all_contracts.groupby("date")["open_interest"].idxmax()
            dominant = self.all_contracts.loc[idx, ["date", "symbol"]].set_index(
                "date"
            )["symbol"]
        else:
            actual_method = method
            if method == "open_interest":
                oi_sum = self.all_contracts.groupby("date")["open_interest"].sum()
                if (oi_sum == 0).all():
                    warnings.warn(
                        "所有日期的 open_interest 均为零，回退到 volume 识别主力合约"
                    )
                    actual_method = "volume"

            if actual_method == "open_interest":
                idx = self.all_contracts.groupby("date")["open_interest"].idxmax()
            elif actual_method == "volume":
                idx = self.all_contracts.groupby("date")["volume"].idxmax()
            else:
                raise ValueError(f"不支持的识别方法: {method}")

            dominant = self.all_contracts.loc[idx, ["date", "symbol"]].set_index(
                "date"
            )["symbol"]

        self.dominant_map = dominant
        return dominant

    def build_continuous_series(self) -> pd.DataFrame:
        """
        构建展期法连续主力合约序列。

        在每日主力合约数据上添加辅助列：
        - is_dominant: 是否为当日主力合约
        - dominant_symbol: 当日主力合约代码
        - prev_dominant_symbol: 前一日主力合约代码（用于检测展期）

        展期标志（rollover_flag）仅在主力合约行标记，
        非主力合约行始终为 False，避免误导依赖该标志的逻辑。

        Returns:
            包含所有合约数据的 DataFrame，带有展期辅助列
        """
        if self.all_contracts is None:
            raise RuntimeError("请先调用 load_csv_files() 加载数据")
        if self.dominant_map is None:
            self.identify_dominant_contracts()

        df = self.all_contracts.copy()

        dominant_df = self.dominant_map.reset_index()
        dominant_df.columns = ["date", "dominant_symbol"]
        dominant_df["prev_dominant_symbol"] = dominant_df["dominant_symbol"].shift(1)

        df = df.merge(dominant_df, on="date", how="left")
        df["is_dominant"] = df["symbol"] == df["dominant_symbol"]

        df["rollover_flag"] = False
        if self.data_mode == "contract":
            dominant_rows = df["is_dominant"]
            df.loc[dominant_rows, "rollover_flag"] = (
                df.loc[dominant_rows, "dominant_symbol"]
                != df.loc[dominant_rows, "prev_dominant_symbol"]
            ) & (df.loc[dominant_rows, "prev_dominant_symbol"].notna())

        self.full_df = df
        self.continuous_df = df[df["is_dominant"]].copy().reset_index(drop=True)
        return self.full_df

    def get_pybroker_df(self) -> pd.DataFrame:
        """
        获取 PyBroker 兼容格式的 DataFrame。

        PyBroker 要求的列：date, symbol, open, high, low, close, volume。
        额外注册的列：open_interest, is_dominant, dominant_symbol, prev_dominant_symbol,
                      rollover_flag, product。

        Returns:
            PyBroker 可直接使用的 DataFrame
        """
        if self.full_df is None:
            self.build_continuous_series()

        cols = [
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
        ]
        available_cols = [c for c in cols if c in self.full_df.columns]
        result = self.full_df[available_cols].copy()
        result = result.sort_values(["date", "symbol"]).reset_index(drop=True)
        return result

    def get_dominant_only_df(self) -> pd.DataFrame:
        """
        仅获取主力合约数据（展期法连续序列）。

        Returns:
            仅包含主力合约行的 DataFrame
        """
        if self.continuous_df is None:
            self.build_continuous_series()
        return self.continuous_df.copy()

    @property
    def product_symbols(self) -> Dict[str, List[str]]:
        """
        获取各品种的合约代码列表（带缓存）。

        Returns:
            字典 {品种代码: [合约代码列表]}
        """
        if self._product_symbols is None:
            if self.all_contracts is None:
                raise RuntimeError("请先调用 load_csv_files() 加载数据")
            self._product_symbols = {
                p: sorted(g["symbol"].unique().tolist())
                for p, g in self.all_contracts.groupby("product")
            }
        return self._product_symbols

    def get_product_symbols(
        self, product: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        获取各品种的合约代码列表。

        Args:
            product: 可选，指定品种代码，为 None 时返回所有品种

        Returns:
            字典 {品种代码: [合约代码列表]}
        """
        if product:
            syms = self.product_symbols.get(product, [])
            return {product: syms}
        return self.product_symbols

    def get_rollover_dates(self) -> pd.DataFrame:
        """
        获取所有展期日期及对应的合约切换信息。

        品种模式下始终返回空 DataFrame（无展期）。

        Returns:
            DataFrame，包含 date, prev_dominant_symbol, dominant_symbol
        """
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

    def get_data_summary(self) -> Dict:
        """
        获取数据摘要信息。

        Returns:
            包含数据概况的字典
        """
        if self.all_contracts is None:
            return {"status": "未加载数据"}

        products = self.product_symbols
        summary = {
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
            "rollover_count": len(self.get_rollover_dates())
            if self.full_df is not None
            else 0,
        }
        if self.load_errors:
            summary["load_errors"] = self.load_errors
        return summary
