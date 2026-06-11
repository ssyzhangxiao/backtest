"""数据加载器 — 主力合约识别、连续序列构建、价差对 Mixin。"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


class SeriesMixin:
    """主力合约识别、连续序列构建、价差对相关方法。"""

    # 以下属性由 DataLoader.__init__ 设置，此处声明类型供类型检查器使用
    all_contracts: Optional[pd.DataFrame]
    dominant_map: Optional[pd.Series]
    continuous_df: Optional[pd.DataFrame]
    full_df: Optional[pd.DataFrame]
    data_mode: Optional[str]

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
            df = self.all_contracts.sort_values(["date", "symbol"]).reset_index(
                drop=True
            )

            if "product" not in df.columns:
                # 无 product 列时回退到跨品种识别
                warnings.warn("数据缺少 product 列，回退到跨品种识别主力合约")
                idx = df.groupby("date")["open_interest"].idxmax()
                dominant = df.loc[idx, ["date", "symbol"]].set_index("date")["symbol"]
            elif method == "open_interest":
                # 按 (date, product) 分组，每组选持仓量最大的合约
                oi_idx = df.groupby(["date", "product"])["open_interest"].idxmax()
                oi_dominant = df.loc[
                    oi_idx, ["date", "product", "symbol", "open_interest"]
                ].copy()

                # 找出 open_interest 为 0 的记录 → 回退到 volume
                zero_oi_mask = oi_dominant["open_interest"] == 0
                if zero_oi_mask.any():
                    zero_oi_keys = oi_dominant[zero_oi_mask][["date", "product"]]
                    for _, row in zero_oi_keys.iterrows():
                        mask = (df["date"] == row["date"]) & (
                            df["product"] == row["product"]
                        )
                        candidates = df[mask]
                        if len(candidates) > 0:
                            best = candidates.loc[candidates["volume"].idxmax()]
                            oi_dominant.loc[
                                (oi_dominant["date"] == row["date"])
                                & (oi_dominant["product"] == row["product"]),
                                "symbol",
                            ] = best["symbol"]

                dominant = oi_dominant.set_index("date")["symbol"]

            elif method == "volume":
                idx = df.groupby(["date", "product"])["volume"].idxmax()
                dominant = df.loc[idx, ["date", "symbol"]].set_index("date")["symbol"]
            else:
                raise ValueError(f"不支持的识别方法: {method}")

        self.dominant_map = dominant
        return dominant

    def build_continuous_series(self) -> pd.DataFrame:
        """构建展期法连续主力合约序列。"""
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
            df["prev_dominant_symbol"] = df.groupby("product")["dominant_symbol"].shift(
                1
            )
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
        df["_oi_rank_in_dp"] = df.groupby(["date", "product"])["open_interest"].rank(
            method="first", ascending=False
        )

        # 提取远月合约（OI 排名=2）信息
        far_rows = df[df["_oi_rank_in_dp"] == 2][
            ["date", "product", "symbol", "close"]
        ].rename(columns={"symbol": "far_symbol", "close": "far_close"})

        # 仅在主力合约行追加远月列（其余行保持 NaN）
        dominant = df["is_dominant"].fillna(False)
        if far_rows.empty:
            # 无远月数据时直接补空列
            df["far_symbol"] = ""
            df["far_close"] = np.nan
            df["spread"] = np.nan
            df["spread"] = np.where(dominant, df["close"] - df["far_close"], np.nan)
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
