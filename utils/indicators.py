"""
技术指标公共工具函数。

提取项目内多处重复实现的技术指标计算逻辑为统一公共函数，
确保计算结果一致、接口清晰、异常处理完善。
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    计算真实波幅（True Range）。

    TR = max(H-L, |H-prevC|, |L-prevC|)

    Args:
        high: 最高价序列（pd.Series，带索引）
        low: 最低价序列（pd.Series，带索引）
        close: 收盘价序列（pd.Series，带索引）

    Returns:
        TR序列（pd.Series）

    Raises:
        ValueError: 输入长度不足
    """
    if len(high) < 2:
        raise ValueError("计算TR至少需要2个数据点")

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # 首行用H-L填充（无前一日收盘价）
    if len(tr) > 0:
        tr.iloc[0] = tr1.iloc[0]
    return tr


def _adx_core(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    ADX核心计算逻辑（内部函数，返回完整Series）。

    Args:
        high: 最高价（pd.Series）
        low: 最低价（pd.Series）
        close: 收盘价（pd.Series）
        period: ADX周期

    Returns:
        (adx, plus_di, minus_di) 三个pd.Series
    """
    tr = compute_true_range(high, low, close)

    # +DM / -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # Wilder平滑：rolling(period).mean()
    atr = tr.rolling(window=period, min_periods=period).mean()
    atr_safe = atr.replace(0, np.nan)

    plus_di = 100 * (
        plus_dm.rolling(window=period, min_periods=period).mean() / atr_safe
    )
    minus_di = 100 * (
        minus_dm.rolling(window=period, min_periods=period).mean() / atr_safe
    )

    # DX → ADX
    dx_denom = (plus_di + minus_di).abs()
    dx = np.where(dx_denom > 0, 100 * (plus_di - minus_di).abs() / dx_denom, 0.0)
    dx = pd.Series(dx, index=high.index)
    adx = dx.rolling(window=period, min_periods=period).mean()

    return adx, plus_di, minus_di


def compute_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> Tuple[float, float, float]:
    """
    计算ADX（平均趋向指数）及其组件的最新值。

    ADX > 25 表示趋势存在，< 20 表示无趋势。
    使用Wilder平滑法（与标准实现一致）。

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: ADX计算周期（默认14）

    Returns:
        (adx_value, plus_di, minus_di) 元组
        - adx_value: ADX值（float），数据不足返回0.0
        - plus_di: +DI值（float）
        - minus_di: -DI值（float）

    Raises:
        ValueError: period <= 0
    """
    if period <= 0:
        raise ValueError(f"ADX周期必须为正整数，当前值: {period}")

    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)

    if len(c) < period * 2:
        logger.debug(f"数据长度{len(c)}不足(需{period * 2})，ADX返回0")
        return 0.0, 0.0, 0.0

    adx_s, plus_di_s, minus_di_s = _adx_core(
        pd.Series(h), pd.Series(l), pd.Series(c), period
    )

    # 取最后一个有效值
    def _last_valid(s: pd.Series) -> float:
        if len(s) > 0 and not np.isnan(s.iloc[-1]):
            return float(s.iloc[-1])
        return 0.0

    return _last_valid(adx_s), _last_valid(plus_di_s), _last_valid(minus_di_s)


def compute_adx_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> pd.Series:
    """
    计算ADX完整序列（用于市场环境检测等需要历史序列的场景）。

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: ADX计算周期（默认14）

    Returns:
        ADX序列（pd.Series），与输入等长，前段为NaN

    Raises:
        ValueError: period <= 0
    """
    if period <= 0:
        raise ValueError(f"ADX周期必须为正整数，当前值: {period}")

    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)

    if len(c) < period * 2:
        return pd.Series(np.full(len(c), np.nan))

    adx_s, _, _ = _adx_core(pd.Series(h), pd.Series(l), pd.Series(c), period)
    return adx_s


def compute_adx_components(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算ADX完整组件序列（供市场环境检测等需要+DI/-DI序列的场景）。

    与 compute_adx 不同，本函数接受并返回 pd.Series，
    保留原始索引，适合在 DataFrame 管道中使用。

    Args:
        high: 最高价序列（pd.Series，带索引）
        low: 最低价序列（pd.Series，带索引）
        close: 收盘价序列（pd.Series，带索引）
        period: ADX计算周期（默认14）

    Returns:
        (adx, plus_di, minus_di) 三个pd.Series，保留原始索引

    Raises:
        ValueError: period <= 0
    """
    if period <= 0:
        raise ValueError(f"ADX周期必须为正整数，当前值: {period}")

    return _adx_core(high, low, close, period)


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    method: str = "simple",
) -> pd.Series:
    """
    计算ATR（平均真实波幅）。

    统一替代 run_full_backtest.py:1194 和 backtest_runner.py:165 中的重复实现。
    基于 compute_true_range() 构建，确保 TR 计算一致。

    Args:
        high: 最高价序列（pd.Series，带索引）
        low: 最低价序列（pd.Series，带索引）
        close: 收盘价序列（pd.Series，带索引）
        period: ATR计算周期（默认14）
        method: 平滑方法，"simple"=简单移动平均，"wilder"=Wilder指数平滑

    Returns:
        ATR序列（pd.Series），与输入等长

    Raises:
        ValueError: period <= 0
    """
    if period <= 0:
        raise ValueError(f"ATR周期必须为正整数，当前值: {period}")

    tr = compute_true_range(high, low, close)

    if method == "wilder":
        # Wilder指数平滑：与 backtest_runner.py 原实现一致
        atr = pd.Series(np.nan, index=high.index, dtype=float)
        # 首个ATR = 前 period 个 TR 的简单平均
        if len(tr) >= period:
            atr.iloc[period - 1] = tr.iloc[:period].mean()
            # 后续用递推：ATR = (prev_ATR * (period-1) + TR) / period
            for i in range(period, len(tr)):
                atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period
        return atr
    else:
        # 简单移动平均：与 run_full_backtest.py 原实现一致
        return tr.rolling(period, min_periods=1).mean()
