"""
因子基础算子库。

提供因子计算所需的通用数学算子，所有因子模块共享。
包括：延迟、差分、移动平均、标准化、排名、衰减、安全除法等。

设计原则：
  - 纯函数，无副作用，无状态
  - 所有函数接受 np.ndarray 输入，返回 np.ndarray
  - NaN 安全：输入含 NaN 时结果合理传播
"""

from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 安全除法
# ──────────────────────────────────────────────


def safe_div(
    numer: np.ndarray,
    denom: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    安全除法：分母为零或NaN时返回fill_value，避免产生inf。

    统一替换所有手动 np.where(np.abs(denom)<eps, nan, numer/denom) 逻辑。

    Args:
        numer: 分子
        denom: 分母
        fill_value: 分母无效时的填充值

    Returns:
        安全除法结果
    """
    denom = np.asarray(denom, dtype=float)
    numer = np.asarray(numer, dtype=float)
    # 分母为0或NaN的位置
    invalid = (np.abs(denom) < 1e-10) | np.isnan(denom)
    result = np.where(invalid, fill_value, numer / np.where(invalid, 1.0, denom))
    return result


# ──────────────────────────────────────────────
# 时序算子
# ──────────────────────────────────────────────


def delay(arr: np.ndarray, n: int) -> np.ndarray:
    """延迟n期：DELAY(x, n) = x_{t-n}"""
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) > n:
        result[n:] = arr[:-n]
    return result


def delta(arr: np.ndarray, n: int) -> np.ndarray:
    """差分：DELTA(x, n) = x_t - x_{t-n}"""
    return arr - delay(arr, n)


# ──────────────────────────────────────────────
# 滚动统计算子
# ──────────────────────────────────────────────


def sma(arr: np.ndarray, window: int) -> np.ndarray:
    """简单移动平均：SMA(x, n)"""
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=window).mean().values


def std(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动标准差：STD(x, n)"""
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=window).std().values


def sum_rolling(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动求和：SUM(x, n)"""
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=window).sum().values


def mean(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动均值：MEAN(x, n)，等价于sma"""
    return sma(arr, window)


def corr(arr1: np.ndarray, arr2: np.ndarray, window: int) -> np.ndarray:
    """滚动相关系数：CORR(x, y, n)"""
    s1 = pd.Series(arr1)
    s2 = pd.Series(arr2)
    return s1.rolling(window=window, min_periods=window).corr(s2).values


# ──────────────────────────────────────────────
# 标准化与排名算子
# ──────────────────────────────────────────────


def zscore(arr: np.ndarray, window: Optional[int] = None) -> np.ndarray:
    """
    标准化：ZSCORE(x, window)。

    - window=None（默认）：扩张窗口标准化，使用截止当前时刻的全部历史数据，
      无前瞻性偏差。
    - window>0：滚动窗口标准化，仅使用最近window期数据。
    - 禁止全序列标准化（使用未来数据），避免前瞻性偏差。

    注意：扩张窗口在序列前期（前几期）标准化结果波动较大，
    属于正常现象，随着数据积累会趋于稳定。
    """
    s = pd.Series(arr, dtype=float)
    if window is None:
        # 扩张窗口：仅使用截止当前时刻的历史数据，无前瞻性
        expanding_mean = s.expanding(min_periods=2).mean()
        expanding_std = s.expanding(min_periods=2).std()
        result = (s - expanding_mean) / expanding_std.replace(0, np.nan)
    elif window > 0:
        # 滚动窗口
        rolling_mean = s.rolling(window=window, min_periods=window).mean()
        rolling_std = s.rolling(window=window, min_periods=window).std()
        result = (s - rolling_mean) / rolling_std.replace(0, np.nan)
    else:
        raise ValueError(
            f"zscore 不允许 window={window}，全序列标准化存在前瞻性偏差。"
            "请使用 window=None（扩张窗口）或 window>0（滚动窗口）。"
        )
    return result.values


def tsrank(arr: np.ndarray, window: int) -> np.ndarray:
    """
    时序排名：TSRANK(x, n)，当前值在最近n期中的排名百分位。

    使用 pct=True（平均排名），平局值取中间排名。
    这在因子库中是标准做法，确保相同值获得相同排名。
    """
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=window).rank(pct=True).values


# ──────────────────────────────────────────────
# 数学变换算子
# ──────────────────────────────────────────────


def sign(arr: np.ndarray) -> np.ndarray:
    """符号函数"""
    return np.sign(arr)


def abs_(arr: np.ndarray) -> np.ndarray:
    """绝对值（避免与内置abs冲突）"""
    return np.abs(arr)


def log(arr: np.ndarray) -> np.ndarray:
    """自然对数（安全处理，避免log(0)）"""
    return np.log(np.abs(arr) + 1e-10)


def decay_linear(arr: np.ndarray, window: int) -> np.ndarray:
    """
    衰减线性加权：DECAYLINEAR(x, n)。
    权重从1到n线性递增，最近期权重最大。
    """
    weights = np.arange(1, window + 1, dtype=float)
    weights = weights / weights.sum()
    s = pd.Series(arr)
    return (
        s.rolling(window=window, min_periods=window)
        .apply(lambda x: np.dot(x, weights), raw=True)
        .values
    )


def winsorize(
    arr: np.ndarray,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> np.ndarray:
    """
    缩尾处理：将极端值钳制到指定百分位。

    规则9要求：每个因子计算完成后，建议进行1%和99%缩尾去除极端值。

    Args:
        arr: 输入数组
        lower_pct: 下界百分位（默认0.01=1%）
        upper_pct: 上界百分位（默认0.99=99%）

    Returns:
        缩尾后的数组（NaN保持不变）
    """
    s = pd.Series(arr, dtype=float)
    lower = s.quantile(lower_pct)
    upper = s.quantile(upper_pct)
    return s.clip(lower=lower, upper=upper).values


def clipping(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    截断处理：将值钳制到 [lower, upper] 区间。

    用于V_01等因子防止异常放大。

    Args:
        arr: 输入数组
        lower: 下界
        upper: 上界

    Returns:
        截断后的数组
    """
    return np.clip(np.asarray(arr, dtype=float), lower, upper)


def sma_ema(arr: np.ndarray, n: int, m: int) -> np.ndarray:
    """
    EMA变体：SMA(x, n, m) = (x*m + prev_SMA*(n-m)) / n。
    同花顺SMA函数，m为平滑系数。

    NaN处理：遇到NaN输入时保持NaN并停止递归计算，
    下一个非NaN值作为新的初始化起点重新开始递归。
    """
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) == 0:
        return result

    # 寻找第一个非NaN值作为初始化起点
    initialized = False
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            # NaN输入保持NaN，不递归
            result[i] = np.nan
            continue

        if not initialized:
            # 第一个有效值用x本身初始化
            result[i] = arr[i]
            initialized = True
        else:
            # 需要找到上一个非NaN的result值
            prev = np.nan
            for j in range(i - 1, -1, -1):
                if not np.isnan(result[j]):
                    prev = result[j]
                    break
            if np.isnan(prev):
                # 没有可用的前值，重新初始化
                result[i] = arr[i]
            else:
                result[i] = (arr[i] * m + prev * (n - m)) / n

    return result
