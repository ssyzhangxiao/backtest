"""
波动率类因子（V_01~V_04）。

波动率因子捕捉持仓量异动与价格波动的关联信号，
持仓量异动是波动率扩张的先兆。

因子列表：
  - V_01: 5日持仓量变化率
  - V_02: 平滑日内波动率与5日增仓乘积
  - V_03: 日内振幅与增仓幅度双滚动标准化乘积
  - V_04: 持仓量均线差率（OI-MACD柱）
"""

from typing import Dict

import numpy as np

from .operators import delay, delta, mean, safe_div, std, zscore


def compute_volatility_factors(
    high: np.ndarray,
    low: np.ndarray,
    oi: np.ndarray,
    intraday_ret: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    计算所有波动率类因子（V_01~V_04）。

    Args:
        high: 最高价序列
        low: 最低价序列
        oi: 安全清洗后的持仓量序列
        intraday_ret: 平滑日内涨幅序列

    Returns:
        {因子编号: 因子值序列}
    """
    return {
        "V_01": _compute_V_01(oi),
        "V_02": _compute_V_02(intraday_ret, oi),
        "V_03": _compute_V_03(high, low, oi),
        "V_04": _compute_V_04(oi),
    }


def _compute_V_01(oi: np.ndarray) -> np.ndarray:
    """
    V_01: 5日持仓量变化率。

    公式: (OI-DELAY(OI,5))/DELAY(OI,5)*100
    改造: 持仓量异动是波动率扩张的先兆
    """
    return safe_div(oi - delay(oi, 5), delay(oi, 5)) * 100


def _compute_V_02(
    intraday_ret: np.ndarray,
    oi: np.ndarray,
) -> np.ndarray:
    """
    V_02: 平滑日内波动率与5日增仓乘积。

    公式: STD(INTRADAY_RET, 20) * DELTA(OI, 5)
    改造: 高波动+持续增仓=确认趋势行情
    """
    return std(intraday_ret, 20) * delta(oi, 5)


def _compute_V_03(
    high: np.ndarray,
    low: np.ndarray,
    oi: np.ndarray,
) -> np.ndarray:
    """
    V_03: 日内振幅与增仓幅度双滚动标准化乘积。

    公式: ZSCORE(HIGH-LOW, 20) * ZSCORE(DELTA(OI,1), 20)
    改造: 使用滚动标准化（窗口20），避免全序列前瞻性偏差
    """
    range_val = high - low
    delta_oi = delta(oi, 1)
    return zscore(range_val, window=20) * zscore(delta_oi, window=20)


def _compute_V_04(oi: np.ndarray) -> np.ndarray:
    """
    V_04: 持仓量均线差率（OI-MACD柱）。

    公式: (MEAN(OI,9)-MEAN(OI,26))/MEAN(OI,12)*100
    改造: 持仓量长短均线发散→波动率将上升
    """
    oi_9 = mean(oi, 9)
    oi_26 = mean(oi, 26)
    oi_12 = mean(oi, 12)
    return safe_div(oi_9 - oi_26, oi_12) * 100
