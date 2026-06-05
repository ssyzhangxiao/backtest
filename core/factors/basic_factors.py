"""
基础因子模块。

从 OHLCV 数据计算 ts_momentum / roll_yield / alpha019 / alpha032 四个因子。
这些因子是简化版实现，用于回测快速验证和因子稳定性分析。

正式因子体系见 core/factors/alpha_futures_*.py。
"""

import numpy as np
import pandas as pd

from utils.indicators import compute_atr


def compute_ts_momentum(close: pd.Series, window: int = 20) -> pd.Series:
    """
    计算时间序列动量因子：过去N日累计收益率。

    Args:
        close: 收盘价序列
        window: 动量窗口（默认20天）

    Returns:
        因子值序列，NaN表示数据不足
    """
    return close.pct_change(window)


def compute_roll_yield(close: pd.Series, lookback: int = 20) -> pd.Series:
    """
    计算展期收益率因子：价格偏离N日均线的百分比。

    Args:
        close: 收盘价序列
        lookback: 均线窗口（默认20天）

    Returns:
        因子值序列（百分比），NaN表示数据不足
    """
    sma = close.rolling(window=lookback, min_periods=lookback).mean()
    return (close - sma) / sma * 100


def compute_alpha019(
    close: pd.Series, short_window: int = 7, long_window: int = 250
) -> pd.Series:
    """
    计算 Alpha#019 因子：短期反转信号 × 长期动量排名。

    定义：(-1 * sign(((close - delay(close,7)) + delta(close,7)))) * (1 + rank((1 + sum(returns,250))))

    Args:
        close: 收盘价序列
        short_window: 短期窗口（默认7天）
        long_window: 长期窗口（默认250天）

    Returns:
        因子值序列，NaN表示数据不足
    """
    # 短期反转：7日价格变化方向
    close_7d_ago = close.shift(short_window)
    delta_7d = close.diff(short_window)
    short_term = close - close_7d_ago + delta_7d
    sign_component = -np.sign(short_term.values)

    # 长期动量：过去250日累计收益
    returns = close.pct_change()
    cum_returns = pd.Series(np.nan, index=close.index)
    for i in range(long_window, len(close) + 1):
        window_returns = returns.iloc[i - long_window : i]
        cum_returns.iloc[i - 1] = np.prod(1 + window_returns) - 1

    cum_rank = cum_returns.rank(pct=True)
    factor = sign_component * (1 + cum_rank.values)
    return factor


def compute_alpha032(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    ma_window: int = 7,
    corr_window: int = 230,
) -> pd.Series:
    """
    计算 Alpha#032 因子：价格偏离均线 + VWAP相关性。

    定义：scale(((sum(close,7)/7)-close)) + (20*scale(correlation(vwap, delay(close,5),230)))

    Args:
        close: 收盘价序列
        high: 最高价序列
        low: 最低价序列
        volume: 成交量序列
        ma_window: 均线窗口（默认7天）
        corr_window: 相关性窗口（默认230天）

    Returns:
        因子值序列，NaN表示数据不足
    """
    # 第一部分：价格偏离7日均线
    ma_7 = close.rolling(window=ma_window, min_periods=ma_window).mean()
    price_deviation = ma_7 - close

    # 第二部分：VWAP与5日前收盘价的230天滚动相关性
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).rolling(10, min_periods=1).sum() / volume.rolling(
        10, min_periods=1
    ).sum().replace(0, 1)
    close_5d_ago = close.shift(5)
    rolling_corr = vwap.rolling(window=corr_window, min_periods=corr_window // 2).corr(
        close_5d_ago
    )

    factor = price_deviation.values + 20 * rolling_corr.values
    return factor


def compute_factor_scores_from_ohlcv(
    ohlcv: pd.DataFrame,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    从 OHLCV 数据计算各因子得分（不依赖 PyBroker）。

    委托 utils/indicators.compute_atr() 计算 ATR，消除重复实现。

    因子：
      - ts_momentum: 20日收益率，归一化到 [-1, 1]
      - roll_yield: 价格偏离20日均线的百分比，归一化到 [-1, 1]
      - alpha019: 简化版短期反转 × 长期动量排名
      - alpha032: 简化版 收盘价与VWAP的相关系数

    Args:
        ohlcv: 含 close, high, low, volume 列的 DataFrame，按日期排序
        atr_period: ATR 计算周期

    Returns:
        含各因子得分和前瞻收益的 DataFrame
    """
    df = ohlcv.copy().sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # ── ts_momentum: 20日收益率归一化 ──
    ret_20 = compute_ts_momentum(close, 20)
    # 委托公共 compute_atr
    atr = compute_atr(high, low, close, atr_period)
    mom_norm = np.clip(ret_20 / (atr / close + 1e-8) * 0.1, -1.0, 1.0)
    df["ts_momentum"] = mom_norm.fillna(0.0)

    # ── roll_yield: 价格偏离20日均线 ──
    spread_pct = compute_roll_yield(close, 20)
    df["roll_yield"] = np.clip(spread_pct / 5.0, -1.0, 1.0).fillna(0.0)

    # ── alpha019: 简化版短期反转 × 长期动量排名 ──
    alpha019 = compute_alpha019(close, 7, 250)
    df["alpha019"] = np.clip(alpha019 * 0.3, -1.0, 1.0).fillna(0.0)

    # ── alpha032: 简化版 收盘价与VWAP的相关系数 ──
    alpha032 = compute_alpha032(close, high, low, volume, 7, 230)
    df["alpha032"] = np.clip(alpha032.fillna(0.0), -1.0, 1.0)

    # ── 前瞻收益（5日） ──
    df["forward_return"] = close.shift(-5) / close - 1.0

    return df
