"""跨品种价差因子辅助函数。"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.ext.factors.alpha_futures.cross_spread import (
    STRONG_IC_PAIRS,
    compute_pair_spread_factor,
)


def compute_pair_signal(
    data_source,
    a_sym: str,
    b_sym: str,
    start: str,
    end: str,
    spread_window: int = 60,
    smoothing_window: int = 3,
) -> Optional[pd.DataFrame]:
    """计算一对跨品种价差因子信号（按日期对齐 A 和 B）。"""
    a_data = data_source.query(start, end, symbols=[a_sym])
    b_data = data_source.query(start, end, symbols=[b_sym])
    if a_data is None or b_data is None:
        return None
    if len(a_data) < spread_window or len(b_data) < spread_window:
        return None
    merged = (
        a_data[["date", "close"]]
        .merge(b_data[["date", "close"]], on="date", suffixes=("_a", "_b"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(merged) < spread_window:
        return None
    close_a = merged["close_a"].values.astype(float)
    close_b = merged["close_b"].values.astype(float)
    signal = compute_pair_spread_factor(
        close_a,
        close_b,
        spread_window=spread_window,
        smoothing_window=smoothing_window,
    )
    return pd.DataFrame({"date": pd.to_datetime(merged["date"]), "value": signal})


def build_cross_spread_panel(
    data_source,
    symbols: List[str],
    start: str,
    end: str,
    pair_names: List[str],
) -> pd.DataFrame:
    """预计算所有强 IC 配对的价差信号并转换为面板。"""
    rows = []
    for pair_name in pair_names:
        if pair_name not in STRONG_IC_PAIRS:
            continue
        a_sym, b_sym = STRONG_IC_PAIRS[pair_name]
        sig = compute_pair_signal(data_source, a_sym, b_sym, start, end)
        if sig is None or sig.empty:
            continue
        for sym in symbols:
            rows.append(
                {
                    "date": sig["date"].values,
                    "symbol": sym,
                    "factor": f"XSPR_{pair_name}",
                    "value": sig["value"].values,
                }
            )
    if not rows:
        return pd.DataFrame()
    # 展开为长格式
    all_rows = []
    for r in rows:
        dates = r["date"]
        values = r["value"]
        for i in range(len(dates)):
            if np.isfinite(values[i]):
                all_rows.append(
                    {
                        "date": pd.to_datetime(dates[i]),
                        "symbol": r["symbol"],
                        "factor": r["factor"],
                        "value": float(values[i]),
                    }
                )
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
