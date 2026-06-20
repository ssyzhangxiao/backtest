"""交易日历模块 — 从数据源自动推断交易日，整合节假日过滤。

数据源：PyBrokerDataSource（TqSdk 实际数据），非本地 CSV。
节假日：data/china_holidays.json（2025-2026）。

规则 17：不重复造轮子，复用 utils/date_utils.py 的 safe_to_timestamp。
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

from utils.date_utils import safe_to_timestamp

__all__ = [
    "TradingCalendar",
    "get_trading_calendar",
]

# 单例缓存
_calendar_instance: Optional[TradingCalendar] = None


class TradingCalendar:
    """交易日历：从数据源推断交易日 + 节假日过滤。

    用法：
        cal = TradingCalendar.from_data_source(ds)
        cal.is_trading_day("2026-06-19")
        cal.next_trading_day("2026-06-19")
        cal.trading_days_in_range("2026-06-01", "2026-06-30")
    """

    def __init__(self):
        self._dates: Optional[pd.DatetimeIndex] = None
        self._holidays: Set[date] = set()
        self._date_set: Optional[Set[pd.Timestamp]] = None

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_data_source(cls, data_source) -> TradingCalendar:
        """从 PyBrokerDataSource 或 pd.DataFrame 构建交易日历。

        Args:
            data_source: PyBrokerDataSource 实例 或 包含 'date' 列的 pd.DataFrame
        """
        cal = cls()
        cal._load_holidays()
        if hasattr(data_source, 'to_pybroker_df'):
            df = data_source.to_pybroker_df()
        elif isinstance(data_source, pd.DataFrame):
            df = data_source
        else:
            raise TypeError(f"不支持的数据源类型: {type(data_source)}，需要 PyBrokerDataSource 或 DataFrame")
        cal._build_from_df(df)
        return cal

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, date_col: str = "date") -> TradingCalendar:
        """从 DataFrame 构建交易日历。"""
        cal = cls()
        cal._load_holidays()
        cal._build_from_df(df, date_col)
        return cal

    # ------------------------------------------------------------------
    # 内部构建
    # ------------------------------------------------------------------

    def _build_from_df(self, df: pd.DataFrame, date_col: str = "date"):
        if date_col not in df.columns:
            raise KeyError(f"DataFrame 无 '{date_col}' 列，可用列: {list(df.columns)}")
        all_dates = pd.to_datetime(df[date_col]).dropna().unique()
        all_dates = pd.DatetimeIndex(all_dates).sort_values()
        # 归一化到日期（去掉时间部分）
        all_dates = all_dates.normalize().unique()
        all_dates = pd.DatetimeIndex(all_dates).sort_values()
        # 过滤节假日
        trading_dates = [d for d in all_dates if d.date() not in self._holidays]
        self._dates = pd.DatetimeIndex(trading_dates)
        self._date_set = set(self._dates)

    def _load_holidays(self, holiday_path: str = "data/china_holidays.json"):
        path = Path(holiday_path)
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for year_str, entries in data.items():
            for entry in entries:
                try:
                    self._holidays.add(date.fromisoformat(entry["date"]))
                except (ValueError, KeyError):
                    continue

    def _ensure_loaded(self):
        if self._dates is None:
            raise RuntimeError("交易日历未加载，请先调用 from_data_source() 或 from_dataframe()")

    # ------------------------------------------------------------------
    # 查询方法
    # ------------------------------------------------------------------

    def is_trading_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> bool:
        """判断是否为交易日。"""
        self._ensure_loaded()
        ts = safe_to_timestamp(d)
        if ts is None:
            return False
        return ts in self._date_set

    def last_trading_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> Optional[pd.Timestamp]:
        """返回 <= d 的最近交易日。"""
        self._ensure_loaded()
        ts = safe_to_timestamp(d)
        if ts is None:
            return None
        mask = self._dates <= ts
        if not mask.any():
            return None
        return self._dates[mask][-1]

    def next_trading_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> Optional[pd.Timestamp]:
        """返回 > d 的下一个交易日。"""
        self._ensure_loaded()
        ts = safe_to_timestamp(d)
        if ts is None:
            return None
        mask = self._dates > ts
        if not mask.any():
            return None
        return self._dates[mask][0]

    def previous_trading_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> Optional[pd.Timestamp]:
        """返回 < d 的上一个交易日。"""
        self._ensure_loaded()
        ts = safe_to_timestamp(d)
        if ts is None:
            return None
        mask = self._dates < ts
        if not mask.any():
            return None
        return self._dates[mask][-1]

    def trading_days_in_range(
        self, start: Union[str, date, datetime, pd.Timestamp],
        end: Union[str, date, datetime, pd.Timestamp],
    ) -> pd.DatetimeIndex:
        """返回 [start, end] 内的所有交易日。"""
        self._ensure_loaded()
        s = safe_to_timestamp(start)
        e = safe_to_timestamp(end)
        if s is None or e is None:
            return pd.DatetimeIndex([])
        return self._dates[(self._dates >= s) & (self._dates <= e)]

    def trading_days_count(self, start: str, end: str) -> int:
        """返回 [start, end] 内交易日数量。"""
        return len(self.trading_days_in_range(start, end))

    def shift_trading_day(
        self, d: Union[str, date, datetime, pd.Timestamp], n: int = 1,
    ) -> Optional[pd.Timestamp]:
        """平移 n 个交易日（n>0 向前，n<0 向后）。"""
        self._ensure_loaded()
        ts = safe_to_timestamp(d)
        if ts is None:
            return None
        positions = self._dates.get_indexer([ts], method="ffill")
        idx = positions[0]
        if idx == -1:
            return None
        target = idx + n
        if target < 0 or target >= len(self._dates):
            return None
        return self._dates[target]

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def all_dates(self) -> pd.DatetimeIndex:
        """所有交易日。"""
        self._ensure_loaded()
        return self._dates

    @property
    def first_date(self) -> pd.Timestamp:
        self._ensure_loaded()
        return self._dates[0]

    @property
    def last_date(self) -> pd.Timestamp:
        self._ensure_loaded()
        return self._dates[-1]

    @property
    def holiday_count(self) -> int:
        return len(self._holidays)

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._dates)

    def __repr__(self) -> str:
        if self._dates is None:
            return "TradingCalendar(未加载)"
        return f"TradingCalendar({len(self._dates)} 交易日, {self._dates[0].date()} ~ {self._dates[-1].date()})"


def get_trading_calendar(data_source=None) -> TradingCalendar:
    """获取全局交易日历（单例）。

    首次调用时需传入 data_source（PyBrokerDataSource 或 DataFrame），
    后续调用无需再传。
    """
    global _calendar_instance
    if _calendar_instance is None:
        if data_source is None:
            raise RuntimeError("首次调用 get_trading_calendar() 需传入 data_source 参数")
        _calendar_instance = TradingCalendar.from_data_source(data_source)
    return _calendar_instance