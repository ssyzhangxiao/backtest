"""交易数据完整性检查 — 验证每个品种的 OHLC/因子计算所需字段是否完整。

数据源：PyBrokerDataSource（TqSdk），非本地 CSV。
检查维度：字段完整性、日期连续性、最新数据时效、因子计算所需字段可用性。

规则 17：不重复造轮子，复用 calendar.py 的交易日历。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .daily_replay import DEFAULT_SYMBOLS
from .calendar import TradingCalendar

__all__ = [
    "check_data_completeness",
    "check_field_completeness",
    "DataCompletenessReport",
]

# 回测所需字段（按优先级）
REQUIRED_FIELDS = ["date", "open", "high", "low", "close", "volume"]
FACTOR_FIELDS = ["far_close"]  # carry 因子需要远月合约价格
OPTIONAL_FIELDS = ["open_interest", "turnover", "vwap"]


class DataCompletenessReport:
    """数据完整性检查报告。"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.status: str = "unknown"  # ok / warn / error / missing
        self.messages: List[str] = []
        self.details: Dict[str, Any] = {}

    def add_error(self, msg: str, **kwargs):
        self.status = "error"
        self.messages.append(f"[ERROR] {msg}")
        self.details.update(kwargs)

    def add_warning(self, msg: str, **kwargs):
        if self.status != "error":
            self.status = "warn"
        self.messages.append(f"[WARN] {msg}")
        self.details.update(kwargs)

    def add_info(self, msg: str, **kwargs):
        if self.status == "unknown":
            self.status = "ok"
        self.messages.append(f"[INFO] {msg}")
        self.details.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "status": self.status,
            "messages": self.messages,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return f"DataCompletenessReport({self.symbol}, {self.status}, {len(self.messages)} msgs)"


def check_field_completeness(
    df: pd.DataFrame, symbol: str, required: List[str] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """检查 DataFrame 字段完整度（非空比例）。

    Returns:
        (field_completeness_pct, missing_fields)
    """
    if required is None:
        required = REQUIRED_FIELDS
    completeness = {}
    missing = []
    for col in required:
        if col not in df.columns:
            completeness[col] = 0.0
            missing.append(col)
        else:
            non_null = df[col].notna().sum()
            total = len(df)
            completeness[col] = non_null / total * 100 if total > 0 else 0.0
            if completeness[col] < 99.9:
                missing.append(f"{col}({completeness[col]:.1f}%)")
    return completeness, missing


def check_data_completeness(
    data_source=None,
    symbols: Optional[List[str]] = None,
    lookback_days: int = 30,
    calendar: Optional[TradingCalendar] = None,
) -> pd.DataFrame:
    """检查每个品种的数据完整性，返回结构化报告。

    Args:
        data_source: PyBrokerDataSource 实例 或 包含 'symbol' 列的 pd.DataFrame
        symbols: 品种列表，默认 DEFAULT_SYMBOLS
        lookback_days: 最新数据时效阈值（天数）
        calendar: 交易日历（用于计算交易日间隔）

    Returns:
        DataFrame 每行一个品种，列: symbol, status, messages, last_date, days_gap,
              trading_gap, field_issues, field_count, row_count
    """
    symbols = symbols or DEFAULT_SYMBOLS
    today = pd.Timestamp.now().normalize()

    # 获取 DataFrame
    if data_source is None:
        raise RuntimeError("data_source 不能为空，请传入 PyBrokerDataSource 或 DataFrame")
    if hasattr(data_source, 'to_pybroker_df'):
        df = data_source.to_pybroker_df()
    elif isinstance(data_source, pd.DataFrame):
        df = data_source
    else:
        raise TypeError(f"不支持的数据源类型: {type(data_source)}")

    if 'symbol' not in df.columns:
        raise KeyError("DataFrame 缺少 'symbol' 列，无法按品种检查")

    reports = []
    for sym in symbols:
        report = DataCompletenessReport(sym)
        sub = df[df['symbol'] == sym].copy()

        if sub.empty:
            report.add_error("品种数据为空", row_count=0)
            reports.append(report.to_dict())
            continue

        # 1. 检查必需字段
        field_pct, field_missing = check_field_completeness(sub, sym)
        report.add_info(
            f"字段完整度: {len(field_pct)}/{len(field_pct)} ",
            field_completeness=field_pct,
            field_issues=field_missing,
            field_count=len(field_pct),
        )
        if field_missing:
            report.add_warning(f"字段缺失: {field_missing}")

        # 2. 检查因子计算所需字段
        for col in FACTOR_FIELDS:
            if col not in sub.columns:
                report.add_warning(f"因子计算字段缺失: {col}（carry/basis_momentum 可能无效）")
            elif sub[col].notna().sum() / len(sub) < 0.5:
                report.add_warning(f"因子计算字段可用率低: {col}={sub[col].notna().sum()/len(sub)*100:.1f}%")

        # 3. 检查日期连续性
        sub = sub.sort_values('date').reset_index(drop=True)
        dates = pd.to_datetime(sub['date'])
        last_date = dates.max()
        first_date = dates.min()

        # 日期去重检查
        dup_count = dates.duplicated().sum()
        if dup_count > 0:
            report.add_warning(f"存在 {dup_count} 行重复日期")

        # 日期间隙检查（相邻日期间隔 > 7 天）
        gaps = dates.diff().dropna()
        big_gaps = gaps[gaps > timedelta(days=7)]
        if len(big_gaps) > 0:
            gap_dates = [dates.iloc[i] for i in big_gaps.index]
            report.add_warning(
                f"发现 {len(big_gaps)} 处日期间隙 > 7 天: {[str(d.date()) for d in gap_dates[:5]]}",
                gap_count=len(big_gaps),
            )

        # 4. 检查最新数据时效
        days_gap = (today - last_date).days
        trading_gap = None
        if calendar is not None:
            try:
                td = calendar.trading_days_in_range(last_date, today)
                trading_gap = len(td) - 1  # 不含 last_date 本身
            except Exception:
                trading_gap = None

        report.add_info(
            f"最新数据: {last_date.date()}, 距今 {days_gap} 天",
            last_date=last_date,
            first_date=first_date,
            days_gap=days_gap,
            trading_gap=trading_gap,
            row_count=len(sub),
        )

        if days_gap > lookback_days:
            report.add_warning(f"最新数据距今 {days_gap} 天，超过阈值 {lookback_days} 天")

        # 5. 检查极端值
        if 'close' in sub.columns and 'high' in sub.columns and 'low' in sub.columns:
            # close 应在 [low, high] 范围内
            close_out_of_range = (
                (sub['close'] < sub['low']) | (sub['close'] > sub['high'])
            ).sum()
            if close_out_of_range > 0:
                report.add_warning(f"close 超出 [low, high] 范围: {close_out_of_range} 行")

        reports.append(report.to_dict())

    return pd.DataFrame(reports)


def check_data_completeness_summary(
    data_source=None,
    symbols: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """检查数据完整性并返回简版报告（仅 status + 关键指标）。"""
    full = check_data_completeness(data_source, symbols, **kwargs)
    if full.empty:
        return full
    summary = full[['symbol', 'status']].copy()
    # 从 details 字典中提取关键字段
    for col in ['last_date', 'days_gap', 'trading_gap', 'row_count']:
        summary[col] = full['details'].apply(lambda d: d.get(col) if isinstance(d, dict) else None)
    return summary