"""
回归类因子（R_01~R_05）。

回归因子捕捉价格与持仓量的背离信号，
识别"增仓滞涨"等顶部反转和底部反转机会。

因子列表：
  - R_01: 平滑日内涨跌与增仓率背离
  - R_02: 最高价与增仓率滚动标准化5日相关性
  - R_03: 收益率变化与平滑开盘增仓相关性乘积
  - R_04: 期限结构均值回复
  - R_05: 负相对持仓量
"""

from typing import Dict, Optional

import numpy as np

from .operators import corr, delay, delta, log, mean, safe_div, zscore


def compute_reversal_factors(
    close: np.ndarray,
    high: np.ndarray,
    oi: np.ndarray,
    intraday_ret: np.ndarray,
    open_adj: np.ndarray,
    carry: np.ndarray,
    zscore_window: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    计算所有回归类因子（R_01~R_05）。

    Args:
        close: 收盘价序列
        high: 最高价序列
        oi: 安全清洗后的持仓量序列
        intraday_ret: 平滑日内涨幅序列
        open_adj: 平滑开盘价序列
        carry: Carry因子序列（已正交化）
        zscore_window: zscore窗口（None=扩张窗口）

    Returns:
        {因子编号: 因子值序列}
    """
    return {
        "R_01": _compute_R_01(oi, intraday_ret, zscore_window),
        "R_02": _compute_R_02(high, oi),
        "R_03": _compute_R_03(close, open_adj, oi, zscore_window),
        "R_04": _compute_R_04(carry, zscore_window),
        "R_05": _compute_R_05(oi),
    }


def _compute_R_01(
    oi: np.ndarray,
    intraday_ret: np.ndarray,
    zscore_window: Optional[int] = None,
) -> np.ndarray:
    """
    R_01: 平滑日内涨跌与增仓率的背离。

    公式: -1 * CORR(ZSCORE(DELTA(LOG(OI),1),w), ZSCORE(INTRADAY_RET,w), 6)
    改造: INTRADAY_RET替换原公式消除跳空缺口干扰；ZSCORE使用滚动/扩张窗口
    适用性: 捕捉"增仓滞涨"的顶部反转
    """
    log_oi = log(oi)
    z_delta_oi = zscore(delta(log_oi, 1), window=zscore_window)
    z_intraday = zscore(intraday_ret, window=zscore_window)
    return -1 * corr(z_delta_oi, z_intraday, 6)


def _compute_R_02(high: np.ndarray, oi: np.ndarray) -> np.ndarray:
    """
    R_02: 最高价与增仓率滚动标准化的5日相关性。

    公式: -1 * CORR(HIGH, ZSCORE(DELTA(OI,1),20), 5)
    改造: delta_oi使用滚动标准化（窗口20），避免全序列前瞻性偏差
    """
    delta_oi = delta(oi, 1)
    z_delta_oi = zscore(delta_oi, window=20)
    return -1 * corr(high, z_delta_oi, 5)


def _compute_R_03(
    close: np.ndarray,
    open_adj: np.ndarray,
    oi: np.ndarray,
    zscore_window: Optional[int] = None,
) -> np.ndarray:
    """
    R_03: 收益率变化与平滑开盘增仓相关性的乘积。

    公式: (-1*ZSCORE(DELTA(RET,3),w)) * CORR(OPEN_ADJ, DELTA(OI,1), 10)
    改造: 动量反转+资金流向判断多空翻转点
    """
    ret = safe_div(delta(close, 1), delay(close, 1))
    z_delta_ret = zscore(delta(ret, 3), window=zscore_window)
    corr_val = corr(open_adj, delta(oi, 1), 10)
    return (-1 * z_delta_ret) * corr_val


def _compute_R_04(
    carry: np.ndarray,
    zscore_window: Optional[int] = None,
) -> np.ndarray:
    """
    R_04: 期限结构均值回复。

    公式: -1 * ZSCORE(CARRY, w)
    改造: 极端Back/Contango结构不可持续，均值回复
    适用性: Carry已做流动性过滤和动量正交化
    """
    return -1 * zscore(carry, window=zscore_window)


def _compute_R_05(oi: np.ndarray) -> np.ndarray:
    """
    R_05: 负相对持仓量。

    公式: -1 * OI / MEAN(OI, 20)
    改造: 相对持仓量极度萎缩→蓄势反转节点
    """
    return -1 * safe_div(oi, mean(oi, 20))
