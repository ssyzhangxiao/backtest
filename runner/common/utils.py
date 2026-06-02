"""
通用工具模块。

提取自 run_full_backtest.py 和 core/report_builder.py 中的重复工具函数，
统一调用入口，消除重复实现（重复#1）。
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_EPSILON = 1e-10


def safe_float(val: Any) -> float:
    """
    安全转 float，异常时返回 0.0。

    统一替代 run_full_backtest.py:81 和 core/report_builder.py:67 中的重复实现。

    Args:
        val: 待转换的值

    Returns:
        float 值，转换失败返回 0.0
    """
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def is_valid_number(val: Any) -> bool:
    """
    检查值是否为有效的有限数值。

    Args:
        val: 待检查的值

    Returns:
        True 表示有效有限数值
    """
    try:
        v = float(val)
        return not (np.isnan(v) or np.isinf(v))
    except (ValueError, TypeError):
        return False


def safe_div(a: float, b: float) -> float:
    """
    安全除法，除零返回 0.0。

    Args:
        a: 被除数
        b: 除数

    Returns:
        除法结果，b 接近零时返回 0.0
    """
    return a / b if abs(b) > _EPSILON else 0.0


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """
    安全保存 DataFrame 到 CSV。

    Args:
        df: 待保存的数据
        path: 保存路径
    """
    try:
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            logger.debug(f"跳过空数据保存: {path}")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"已保存: {path}")
    except Exception as e:
        logger.error(f"保存CSV失败 {path}: {e}")


def format_metrics(m: dict) -> dict:
    """
    格式化绩效指标：四舍五入 float，NaN/Inf 改为 'N/A'。

    委托 utils/metrics.MetricsCalculator，此处保留轻量版本用于简单场景。

    Args:
        m: 原始指标字典

    Returns:
        格式化后的指标字典
    """
    result: dict = {}
    for k, v in m.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            result[k] = "N/A"
        elif isinstance(v, float):
            result[k] = round(v, 4)
        else:
            result[k] = v
    return result


def sanitize_filename(name: str) -> str:
    """
    清理文件名，移除非法字符，防止路径遍历。

    Args:
        name: 原始文件名

    Returns:
        安全的文件名
    """
    import re

    # 移除或替换非法字符
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    # 移除控制字符
    sanitized = re.sub(r"[\x00-\x1f\x7f]", "", sanitized)
    # 防止路径遍历
    sanitized = sanitized.replace("..", "_")
    # 去除首尾空格
    sanitized = sanitized.strip()
    return sanitized if sanitized else "unnamed"
