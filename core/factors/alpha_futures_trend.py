"""
趋势类因子（T_01~T_05）。

趋势因子捕捉价格动量与持仓量变化的共振信号，
OI替代VOLUME，增仓确认趋势方向。

因子列表：
  - T_01: 6日动量与日增仓乘积
  - T_02: 12日动量与总持仓乘积
  - T_03: 日度收益率与日增仓乘积
  - T_04: 期限结构与增仓共振（顶级Alpha因子）
  - T_05: 6日条件增仓累积（OBV-OI变形）
"""

from typing import Dict

import numpy as np

from .operators import delay, delta, safe_div, sum_rolling


def compute_trend_factors(
    close: np.ndarray,
    oi: np.ndarray,
    carry: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    计算所有趋势类因子（T_01~T_05）。

    Args:
        close: 收盘价序列（需向后复权）
        oi: 安全清洗后的持仓量序列
        carry: Carry因子序列（已正交化）

    Returns:
        {因子编号: 因子值序列}
    """
    return {
        "T_01": _compute_T_01(close, oi),
        "T_02": _compute_T_02(close, oi),
        "T_03": _compute_T_03(close, oi),
        "T_04": _compute_T_04(carry, oi),
        "T_05": _compute_T_05(close, oi),
    }


def _compute_T_01(close: np.ndarray, oi: np.ndarray) -> np.ndarray:
    """
    T_01: 6日动量与日增仓乘积。

    公式: (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6) * DELTA(OI,1)
    改造: OI替代VOLUME，增仓确认趋势
    """
    momentum = safe_div(delta(close, 6), delay(close, 6))
    return momentum * delta(oi, 1)


def _compute_T_02(close: np.ndarray, oi: np.ndarray) -> np.ndarray:
    """
    T_02: 12日动量与总持仓乘积。

    公式: (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12) * OI
    改造: 过滤低持仓伪突破
    """
    momentum = safe_div(delta(close, 12), delay(close, 12))
    return momentum * oi


def _compute_T_03(close: np.ndarray, oi: np.ndarray) -> np.ndarray:
    """
    T_03: 日度收益率与日增仓乘积。

    公式: (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1) * DELTA(OI,1)
    改造: 极短期资金入场方向确认
    """
    daily_ret = safe_div(delta(close, 1), delay(close, 1))
    return daily_ret * delta(oi, 1)


def _compute_T_04(carry: np.ndarray, oi: np.ndarray) -> np.ndarray:
    """
    T_04: 期限结构与增仓共振（顶级Alpha因子）。

    公式: CARRY * DELTA(OI,1)
    改造: Back+增仓=做多，Contango+增仓=做空
    适用性: Carry已做流动性过滤和动量正交化，OI已做换月/交割月清洗
    """
    return carry * delta(oi, 1)


def _compute_T_05(close: np.ndarray, oi: np.ndarray) -> np.ndarray:
    """
    T_05: 6日条件增仓累积（OBV-OI变形）。

    公式: SUM(CLOSE>DELAY(CLOSE,1)?DELTA(OI,1):(CLOSE<DELAY(CLOSE,1)?-DELTA(OI,1):0),6)
    改造: 上涨计正增仓，下跌计负增仓
    """
    delta_oi = delta(oi, 1)
    prev_close = delay(close, 1)
    conditional_oi = np.where(
        close > prev_close,
        delta_oi,
        np.where(close < prev_close, -delta_oi, 0.0),
    )
    return sum_rolling(conditional_oi, 6)
