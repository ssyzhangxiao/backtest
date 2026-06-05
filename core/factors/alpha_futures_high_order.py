"""
高阶复合类因子（H_01~H_05）。

高阶复合因子结合趋势、回归、波动率、资金流等多维信号，
通过条件逻辑和标准化乘积捕捉非线性交易机会。

因子列表：
  - H_01: 条件性结构动量
  - H_02: 7日价格变化与持仓衰减线性排名复合因子
  - H_03: 相对持仓时序排名与反转时序排名乘积
  - H_04: 价格加速度与相对持仓排名复合
  - H_05: 三重共振因子
"""

from typing import Callable, Dict, Optional

import numpy as np

from .operators import (
    abs_,
    decay_linear,
    delta,
    delay,
    mean,
    safe_div,
    sign,
    sum_rolling,
    tsrank,
    zscore,
)


def compute_high_order_factors(
    close: np.ndarray,
    oi: np.ndarray,
    carry: np.ndarray,
    zscore_window: Optional[int] = None,
    cache_get: Callable[[str, Callable[[], np.ndarray]], np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    计算所有高阶复合类因子（H_01~H_05）。

    Args:
        close: 收盘价序列
        oi: 安全清洗后的持仓量序列
        carry: Carry因子序列（已正交化）
        zscore_window: zscore窗口（None=扩张窗口）
        cache_get: 缓存获取函数，避免重复计算

    Returns:
        {因子编号: 因子值序列}
    """
    if cache_get is None:
        # 无缓存时直接计算
        _cache: Dict[str, np.ndarray] = {}

        def cache_get(key: str, fn: Callable[[], np.ndarray]) -> np.ndarray:
            if key not in _cache:
                _cache[key] = fn()
            return _cache[key]

    return {
        "H_01": _compute_H_01(close, oi, carry, cache_get),
        "H_02": _compute_H_02(close, oi, zscore_window, cache_get),
        "H_03": _compute_H_03(close, oi, cache_get),
        "H_04": _compute_H_04(close, oi, zscore_window, cache_get),
        "H_05": _compute_H_05(close, oi, carry, zscore_window),
    }


def _compute_H_01(
    close: np.ndarray,
    oi: np.ndarray,
    carry: np.ndarray,
    cache_get: Callable[[str, Callable[[], np.ndarray]], np.ndarray],
) -> np.ndarray:
    """
    H_01: 条件性结构动量。

    公式: (MEAN(OI,20)<OI) ? (CARRY*TSRANK(ABS(DELTA(CLOSE,7)),60)) : (-1*OI)
    改造: 仅在增仓环境下交易期限结构与价格动量共振；缩仓退守负持仓因子
    适用性: 高度契合商品非线性特征
    """
    oi_mean_20 = cache_get("oi_mean20", lambda: mean(oi, 20))
    is_accumulating = oi > oi_mean_20
    price_momentum = tsrank(abs_(delta(close, 7)), 60)
    return np.where(is_accumulating, carry * price_momentum, -1 * oi)


def _compute_H_02(
    close: np.ndarray,
    oi: np.ndarray,
    zscore_window: Optional[int],
    cache_get: Callable[[str, Callable[[], np.ndarray]], np.ndarray],
) -> np.ndarray:
    """
    H_02: 7日价格变化与持仓衰减线性排名复合因子。

    公式: -1*ZSCORE(DELTA(CLOSE,7)*(1-ZSCORE(DECAYLINEAR(OI/MEAN(OI,20),9),w)),w)
          * (1+ZSCORE(SUM(RET,250),w))
    改造: 衰减加权强调近期持仓异动，结合长期收益排名
    """
    oi_mean_20 = cache_get("oi_mean20", lambda: mean(oi, 20))
    rel_oi = safe_div(oi, oi_mean_20)
    decay_oi = decay_linear(rel_oi, 9)
    z_decay = zscore(decay_oi, window=zscore_window)
    delta_close_7 = delta(close, 7)
    inner = delta_close_7 * (1 - z_decay)
    ret = safe_div(delta(close, 1), delay(close, 1))
    sum_ret_250 = sum_rolling(ret, 250)
    z_long_ret = zscore(sum_ret_250, window=zscore_window)
    return -1 * zscore(inner, window=zscore_window) * (1 + z_long_ret)


def _compute_H_03(
    close: np.ndarray,
    oi: np.ndarray,
    cache_get: Callable[[str, Callable[[], np.ndarray]], np.ndarray],
) -> np.ndarray:
    """
    H_03: 相对持仓时序排名与反转时序排名的乘积。

    公式: TSRANK(OI/MEAN(OI,20), 20) * TSRANK(-1*DELTA(CLOSE,7), 8)
    改造: 持仓异常放大+价格短期超跌→反弹拐点
    """
    oi_mean_20 = cache_get("oi_mean20", lambda: mean(oi, 20))
    rel_oi = safe_div(oi, oi_mean_20)
    return tsrank(rel_oi, 20) * tsrank(-1 * delta(close, 7), 8)


def _compute_H_04(
    close: np.ndarray,
    oi: np.ndarray,
    zscore_window: Optional[int],
    cache_get: Callable[[str, Callable[[], np.ndarray]], np.ndarray],
) -> np.ndarray:
    """
    H_04: 价格加速度与相对持仓排名复合。

    公式: (-1*ZSCORE(TSRANK(CLOSE,10),w)) * ZSCORE(DELTA(DELTA(CLOSE,1),1),w)
          * ZSCORE(TSRANK(OI/MEAN(OI,20),5),w)
    改造: 二阶价格变化(加速度)+短期资金面爆发→趋势启动极初期
    """
    z_ts_rank = zscore(tsrank(close, 10), window=zscore_window)
    delta_close_1 = delta(close, 1)
    z_accel = zscore(delta(delta_close_1, 1), window=zscore_window)
    oi_mean_20 = cache_get("oi_mean20", lambda: mean(oi, 20))
    rel_oi = safe_div(oi, oi_mean_20)
    z_oi_rank = zscore(tsrank(rel_oi, 5), window=zscore_window)
    return (-1 * z_ts_rank) * z_accel * z_oi_rank


def _compute_H_05(
    close: np.ndarray,
    oi: np.ndarray,
    carry: np.ndarray,
    zscore_window: Optional[int],
) -> np.ndarray:
    """
    H_05: 三重共振因子。

    公式: ZSCORE(CARRY,w) * ZSCORE(DELTA(OI,5),w) * SIGN(DELTA(CLOSE,5))
    改造: 期限结构陡峭化+持仓量趋势流入+价格突破方向三者同向共振
    适用性: 高阶统计标准化后乘积，极高胜率信号
    """
    z_carry = zscore(carry, window=zscore_window)
    z_delta_oi = zscore(delta(oi, 5), window=zscore_window)
    price_dir = sign(delta(close, 5))
    return z_carry * z_delta_oi * price_dir
