"""数据加载器 — CSV 数据源 Mixin。

提供 CSV 文件加载、格式检测、后处理等方法。
"""

import glob
import logging
import os
import warnings
from typing import Optional

import pandas as pd

_logger = logging.getLogger(__name__)


class CsvMixin:
    """CSV 数据源相关方法。"""

    # 以下属性由 DataLoader.__init__ 设置，此处声明类型供类型检查器使用
    data_dir: Optional[str]
    data_mode: Optional[str]
    load_errors: list
    CONTRACT_REQUIRED_COLUMNS: list
    _product_symbols: object

    def _detect_format_from_df(self, df: pd.DataFrame) -> str:
        """基于 DataFrame 列名检测数据格式。"""
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
        """加载品种汇总格式的CSV文件。"""
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
                self.load_errors.append({"file": filepath, "error": "品种格式转换失败"})
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
        """读取目录下所有匹配的CSV文件并合并。"""
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

        combined = self._finalize_loaded_dfs(dfs, len(file_paths))
        # 2026-06-11 修复：把加载结果同步到 self.full_df，
        # 使 get_pybroker_df() 内部 build_continuous_series() 能在其上增强
        if self.full_df is None:
            self.full_df = combined
        return combined
