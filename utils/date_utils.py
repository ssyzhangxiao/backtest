import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("backtest_app")


def safe_to_timestamp(value, label: str = "") -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            logger.warning("%s 日期转换结果为 NaT: %s", label, value)
            return None
        logger.debug("%s 日期转换成功: %s -> %s", label, value, ts)
        return ts
    except Exception as e:
        logger.error("%s 日期转换失败: %s, 错误: %s", label, value, e)
        return None


def apply_date_filter(
    df: pd.DataFrame,
    bt_start,
    bt_end,
    date_col: str = "date",
) -> pd.DataFrame:
    if bt_start is None and bt_end is None:
        logger.info("未设置日期筛选，使用全量数据 (%d 行)", len(df))
        return df

    start_ts = safe_to_timestamp(bt_start, label="bt_start")
    end_ts = safe_to_timestamp(bt_end, label="bt_end")

    if start_ts is None and end_ts is None:
        logger.warning("两个日期均转换失败，跳过日期筛选")
        return df

    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        logger.error("开始日期(%s)大于结束日期(%s)，交换两者", start_ts, end_ts)
        start_ts, end_ts = end_ts, start_ts

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    original_len = len(df)

    mask = pd.Series(True, index=df.index)
    if start_ts is not None:
        mask &= df[date_col] >= start_ts
    if end_ts is not None:
        mask &= df[date_col] <= end_ts

    df = df[mask].copy()
    logger.info(
        "日期筛选: [%s ~ %s], 原始 %d 行 -> 筛选后 %d 行",
        start_ts, end_ts, original_len, len(df),
    )
    return df