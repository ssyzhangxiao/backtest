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

    def for_symbol(self, symbol: str) -> "PyBrokerDataSource":
        """
        返回仅包含单个品种数据的子数据源。

        Args:
            symbol: 品种代码（如 'SHFE.RB'）

        Returns:
            仅含指定品种数据的新 PyBrokerDataSource 实例
        """
        filtered = self._df[self._df["symbol"] == symbol].copy()
        if filtered.empty:
            raise ValueError(f"数据源中无品种 {symbol}，可用品种: {self._symbols}")
        return PyBrokerDataSource(filtered)


def create_hybrid_data_source(
    phone: Optional[str] = None,
    password: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
    data_length: int = 2000,
) -> PyBrokerDataSource:
    """
    混合数据源工厂：TqSdk 在线数据为主，本地 CSV 仅用于 spread（远月价差）补充。

    加载策略：
      1. 必须提供 phone + password + symbols（凭证缺失则抛错，不静默回退）
      2. 从 TqSdk 加载所有合约级数据（连续主力合约识别 + spread 构建）
      3. CSV 仅在 TqSdk 缺失历史数据时做短时段补丁（仅 spread 字段），不做主回测源

    不允许回退到纯 CSV 数据源（避免使用降级方案）。
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

    if not phone or not password:
        raise RuntimeError(
            "TqSdk 凭证未配置（TQSDK_PHONE / TQSDK_PASSWORD）。"
            "规则1：禁止静默回退到本地 CSV。请配置天勤账号后重试。"
        )
    if not symbols:
        raise RuntimeError(
            "未指定品种列表（symbols）。规则1：禁止回退到全量本地 CSV。"
        )

    # ── 主路径：TqSdk 在线数据 ──
    logger.info("从 TqSdk 加载在线数据（%d 品种, data_length=%d）...", len(symbols), data_length)
    try:
        from core.data_loader import DataLoader

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

        tqsdk_df = tqsdk_loader.get_pybroker_df()
        if tqsdk_df.empty:
            raise RuntimeError("TqSdk 数据为空")

        # 仅当 TqSdk 缺少 spread 字段时，从 CSV 补 spread 列（不补主数据）
        if "spread" not in tqsdk_df.columns:
            data_dir = data_dir or os.environ.get("DATA_DIR", "./data")
            csv_loader = DataLoader(data_source="csv", data_dir=data_dir)
            target_csv_files = []
            for sym in symbols:
                csv_path = os.path.join(data_dir, f"{sym}.csv")
                if os.path.exists(csv_path):
                    target_csv_files.append(csv_path)
            if target_csv_files:
                csv_loader.load_csv_files_by_paths(target_csv_files)
                csv_loader.build_continuous_series()
                csv_df = csv_loader.get_pybroker_df()
                spread_cols = ["date", "symbol", "far_symbol", "far_close", "spread"]
                available_cols = [c for c in spread_cols if c in csv_df.columns]
                if available_cols:
                    spread_df = csv_df[available_cols].copy()
                    spread_df["date"] = pd.to_datetime(spread_df["date"]).dt.normalize()
                    tqsdk_df["date"] = pd.to_datetime(tqsdk_df["date"]).dt.normalize()
                    tqsdk_df = tqsdk_df.merge(spread_df, on=["date", "symbol"], how="left")
                    logger.info("已从 CSV 补充 spread 字段（%d 品种）", csv_df["symbol"].nunique())

        logger.info(
            "TqSdk 数据加载成功: %d 行, %d 品种",
            len(tqsdk_df),
            tqsdk_df["symbol"].nunique(),
        )
        return PyBrokerDataSource(tqsdk_df)

    except Exception as e:
        raise RuntimeError(
            f"TqSdk 数据加载失败: {e}。规则1：禁止回退到本地 CSV，请检查天勤账号/网络/品种代码。"
        ) from e
