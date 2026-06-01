"""
PyBroker 数据源 — 将 DataFrame 转为 PyBroker 兼容数据源。

位置: core/engine/pybroker_data_source.py

提供:
  - PyBrokerDataSource: 数据源封装
  - create_hybrid_data_source: TqSdk 优先 + CSV fallback 工厂
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class PyBrokerDataSource:
    """
    PyBroker 兼容数据源。

    接受 pd.DataFrame（格式同 DataLoader.get_pybroker_df() 输出），
    提供 query 方法返回按日期/合约筛选的数据。

    使用方式：
        ds = PyBrokerDataSource(df)
        df_pybroker = ds.to_pybroker_df()
    """

    def __init__(self, df: pd.DataFrame):
        required_cols = {"date", "symbol", "open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"数据缺少必要列: {missing}")

        self._df = df.copy()
        self._df["date"] = pd.to_datetime(self._df["date"])
        for col in ["open", "high", "low", "close", "volume", "open_interest"]:
            if col in self._df.columns:
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce").astype(
                    float
                )
        self._df = self._df.sort_values(["symbol", "date"]).reset_index(drop=True)
        self._symbols = sorted(self._df["symbol"].unique())

    @property
    def symbols(self) -> List[str]:
        return self._symbols

    @property
    def date_range(self) -> Tuple[str, str]:
        return (
            str(self._df["date"].min().date()),
            str(self._df["date"].max().date()),
        )

    def query(
        self, start_date: str, end_date: str, symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """按日期和合约查询数据。"""
        mask = (self._df["date"] >= pd.Timestamp(start_date)) & (
            self._df["date"] <= pd.Timestamp(end_date)
        )
        result = self._df[mask].copy()
        if symbols:
            result = result[result["symbol"].isin(symbols)]
        return result

    def to_pybroker_df(self) -> pd.DataFrame:
        """返回 PyBroker 可直接使用的完整 DataFrame。"""
        return self._df.copy()

    def __len__(self) -> int:
        return len(self._df)


def create_hybrid_data_source(
    phone: Optional[str] = None,
    password: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
    data_length: int = 2000,
) -> PyBrokerDataSource:
    """
    混合数据源工厂：TqSdk 在线数据优先，本地 CSV 为 fallback。

    加载策略：
      1. 若提供 phone + password + symbols → 尝试从 TqSdk 加载实时数据
         → 成功则转为 PyBrokerDataSource 返回
      2. TqSdk 加载失败（未提供凭证/网络错误/账号过期等）→ 回退到 DataLoader
         → 从 data_dir 加载 CSV 数据
      3. 两者均失败 → 抛出 RuntimeError
    """
    phone = phone or os.environ.get("TQSDK_PHONE")
    password = password or os.environ.get("TQSDK_PASSWORD")
    if not phone or not password:
        try:
            import yaml as _yaml
            _cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
            if os.path.exists(_cfg_path):
                with open(_cfg_path, "r", encoding="utf-8") as _f:
                    _cfg = _yaml.safe_load(_f)
                _data_cfg = _cfg.get("data", {})
                phone = phone or _data_cfg.get("tqsdk_phone")
                password = password or _data_cfg.get("tqsdk_password")
        except Exception:
            pass

    if phone and password and symbols:
        try:
            from core.data_loader import DataLoader

            data_dir = data_dir or os.environ.get("DATA_DIR", "./data")
            csv_loader = DataLoader(data_source="csv", data_dir=data_dir)
            target_csv_files = []
            for sym in symbols:
                csv_path = os.path.join(data_dir, f"{sym}.csv")
                if os.path.exists(csv_path):
                    target_csv_files.append(csv_path)
            if not target_csv_files:
                raise RuntimeError(f"未找到目标品种的 CSV 文件: {symbols}")
            csv_loader.load_csv_files_by_paths(target_csv_files)
            csv_loader.build_continuous_series()
            csv_df = csv_loader.get_pybroker_df()

            if csv_df.empty:
                raise RuntimeError("CSV 品种级数据为空")

            logger.info("从 TqSdk 加载合约级数据（用于 spread 远月价差）...")
            tqsdk_loader = DataLoader(
                data_source="tqsdk",
                phone=phone,
                password=password,
                symbols=symbols,
                data_length=data_length,
            )
            tqsdk_loader.load_from_tqsdk(show_progress=True)
            tqsdk_loader.identify_dominant_contracts()
            tqsdk_loader.build_continuous_series()
            tqsdk_loader.build_spread_pairs()

            tqsdk_dom = tqsdk_loader.full_df[tqsdk_loader.full_df["is_dominant"]].copy()
            if "product" in tqsdk_dom.columns and "spread" in tqsdk_dom.columns:
                tqsdk_dom["exchange"] = tqsdk_dom["symbol"].str.split(".").str[0]
                tqsdk_dom["symbol"] = (
                    tqsdk_dom["exchange"] + "." + tqsdk_dom["product"].str.upper()
                )
                tqsdk_dom.drop(columns=["exchange"], inplace=True)

                spread_cols = ["date", "symbol", "far_symbol", "far_close", "spread"]
                available_cols = [c for c in spread_cols if c in tqsdk_dom.columns]
                if available_cols:
                    spread_df = tqsdk_dom[available_cols].copy()
                    spread_df["date"] = pd.to_datetime(spread_df["date"]).dt.normalize()
                    csv_df["date"] = pd.to_datetime(csv_df["date"]).dt.normalize()
                    for col in ["far_symbol", "far_close", "spread"]:
                        if col in csv_df.columns:
                            csv_df.drop(columns=[col], inplace=True)
                    csv_df = csv_df.merge(spread_df, on=["date", "symbol"], how="left")

            logger.info(
                "混合数据加载成功: CSV %d 行 + TqSdk spread, %d 品种",
                len(csv_df),
                csv_df["symbol"].nunique(),
            )
            return PyBrokerDataSource(csv_df)

        except Exception as e:
            logger.warning("TqSdk 混合模式失败 (%s)，回退到纯 CSV 数据源。", e)

    data_dir = data_dir or os.environ.get("DATA_DIR", "./data")
    logger.info("从本地 CSV 加载数据 (%s)...", data_dir)

    try:
        from core.data_loader import DataLoader

        loader = DataLoader(data_source="csv", data_dir=data_dir)
        if symbols:
            target_csv_files = []
            for sym in symbols:
                csv_path = os.path.join(data_dir, f"{sym}.csv")
                if os.path.exists(csv_path):
                    target_csv_files.append(csv_path)
            if target_csv_files:
                loader.load_csv_files_by_paths(target_csv_files)
            else:
                loader.load_all_csv()
        else:
            loader.load_all_csv()

        loader.build_continuous_series()
        df = loader.get_pybroker_df()

        if df.empty:
            raise RuntimeError("CSV 数据为空")

        logger.info("CSV 数据加载成功: %d 行, %d 品种", len(df), df["symbol"].nunique())
        return PyBrokerDataSource(df)

    except Exception as e:
        raise RuntimeError(f"CSV 数据加载失败: {e}") from e
