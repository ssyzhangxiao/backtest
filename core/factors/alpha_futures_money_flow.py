"""
资金流类因子（M_01~M_05）。

资金流因子量化多空哪方在主动加仓，
基于日内价格位置和持仓量变化方向判断资金流向。

因子列表：
  - M_01: 6日日内多空力量与增仓累积
  - M_02: 20日日内多空力量与增仓累积
  - M_03: 20日条件增仓累积
  - M_04: 期限结构驱动的资金流
  - M_05: 持仓量MACD指标
"""

from typing import Callable, Dict

import numpy as np

from .operators import delay, delta, safe_div, sma_ema, sum_rolling


def compute_money_flow_factors(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    oi: np.ndarray,
    carry: np.ndarray,
    cache_get: Callable[[str, Callable[[], np.ndarray]], np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    计算所有资金流类因子（M_01~M_05）。

    Args:
        close: 收盘价序列
        high: 最高价序列
        low: 最低价序列
        oi: 安全清洗后的持仓量序列
        carry: Carry因子序列（已正交化）
        cache_get: 缓存获取函数，避免重复计算

    Returns:
        {因子编号: 因子值序列}
    """
    return {
        "M_01": _compute_M_01(close, high, low, oi),
        "M_02": _compute_M_02(close, high, low, oi),
        "M_03": _compute_M_03(close, oi),
        "M_04": _compute_M_04(carry, oi),
        "M_05": _compute_M_05(oi, cache_get),
    }


def _compute_M_01(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    oi: np.ndarray,
) -> np.ndarray:
    """
    M_01: 6日日内多空力量与增仓累积。

    公式: SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*DELTA(OI,1), 6)
    改造: 日内价格位置乘以增仓量，量化多空哪方在主动加仓
    """
    delta_oi = delta(oi, 1)
    safe_range = high - low
    power = safe_div((close - low) - (high - close), safe_range)
    return sum_rolling(power * delta_oi, 6)


def _compute_M_02(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    oi: np.ndarray,
) -> np.ndarray:
    """
    M_02: 20日日内多空力量与增仓累积。

    公式: SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*DELTA(OI,1), 20)
    改造: M_01的中期波段版本，过滤单日噪声
    """
    delta_oi = delta(oi, 1)
    safe_range = high - low
    power = safe_div((close - low) - (high - close), safe_range)
    return sum_rolling(power * delta_oi, 20)


def _compute_M_03(close: np.ndarray, oi: np.ndarray) -> np.ndarray:
    """
    M_03: 20日条件增仓累积。

    公式: SUM(CLOSE>DELAY(CLOSE,1)?DELTA(OI,1):(CLOSE<DELAY(CLOSE,1)?-DELTA(OI,1):0),20)
    改造: 中期资金流向能量潮指标
    """
    delta_oi = delta(oi, 1)
    prev_close = delay(close, 1)
    conditional_oi = np.where(
        close > prev_close,
        delta_oi,
        np.where(close < prev_close, -delta_oi, 0.0),
    )
    return sum_rolling(conditional_oi, 20)


def _compute_M_04(carry: np.ndarray, oi: np.ndarray) -> np.ndarray:
    """
    M_04: 期限结构驱动的资金流。

    公式: SUM(CARRY>0?DELTA(OI,1):-DELTA(OI,1), 10)
    改造: 近月升水时增仓=多头力量；远月升水时增仓=空头力量
    适用性: 更贴近商品产业链逻辑
    """
    delta_oi = delta(oi, 1)
    directional_oi = np.where(carry > 0, delta_oi, -delta_oi)
    return sum_rolling(directional_oi, 10)


def _compute_M_05(
    oi: np.ndarray,
    cache_get: Callable[[str, Callable[[], np.ndarray]], np.ndarray],
) -> np.ndarray:
    """
    M_05: 持仓量MACD指标。

    公式: SMA(OI,13,2)-SMA(OI,27,2)-SMA(SMA(OI,13,2)-SMA(OI,27,2),10,2)
    改造: 经典量能指标的OI版，金叉/死叉提示资金面拐点
    """
    sma_13 = cache_get("oi_sma13", lambda: sma_ema(oi, 13, 2))
    sma_27 = cache_get("oi_sma27", lambda: sma_ema(oi, 27, 2))
    dif = sma_13 - sma_27
    dea = sma_ema(dif, 10, 2)
    return dif - dea
