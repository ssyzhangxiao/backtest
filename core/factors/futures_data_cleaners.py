"""
商品期货适用性工程改造 — 数据清洗算子。

三大核心改造的数据预处理层：
  1. 跳空缺口修复（compute_open_adj / compute_intraday_ret）
  2. Carry期限结构（compute_carry / _orthogonalize_carry）
  3. OI安全清洗（compute_oi_safe / generate_delivery_exclude / adjust_price_for_roll）

这些清洗算子被 AlphaFutures24.compute_all() 在因子计算前统一调用，
确保输入数据无换月陷阱、无交割月干扰、无跳空污染。
"""

from typing import Optional

import numpy as np
import pandas as pd

from .futures_config import OIThresholdType
from .operators import delay, safe_div


# ──────────────────────────────────────────────
# 跳空缺口修复
# ──────────────────────────────────────────────


def compute_open_adj(
    open_price: np.ndarray,
    close_price: np.ndarray,
    gap_weight: Optional[np.ndarray] = None,
    default_weight: float = 0.5,
) -> np.ndarray:
    """
    平滑开盘价（跳空缺口修复）。

    原始逻辑：OPEN_ADJ = OPEN*0.5 + DELAY(CLOSE,1)*0.5
    适用性改造：权重按品种历史跳空延续率自适应分配，非固定0.5。

    Args:
        open_price: 开盘价序列（需向后复权或比例复权）
        close_price: 收盘价序列（需向后复权或比例复权）
        gap_weight: 品种自适应权重序列（None则使用default_weight）
        default_weight: 默认权重（0.5）

    Returns:
        平滑开盘价序列
    """
    prev_close = delay(close_price, 1)
    w = gap_weight if gap_weight is not None else default_weight
    return open_price * w + prev_close * (1 - w)


def compute_intraday_ret(
    close_price: np.ndarray,
    open_adj: np.ndarray,
    limit_move_threshold: float = 0.06,
    open_price: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    平滑日内涨幅（跳空缺口修复后）。

    INTRADAY_RET = (CLOSE - OPEN_ADJ) / OPEN_ADJ
    适用性改造：开盘触及涨跌停（涨停+跌停）时INTRADAY_RET置零。

    Args:
        close_price: 收盘价序列
        open_adj: 平滑开盘价序列
        limit_move_threshold: 涨跌停阈值（绝对值超过此值视为涨跌停）
        open_price: 原始开盘价（用于涨跌停检测）

    Returns:
        平滑日内涨幅序列
    """
    ret = safe_div(close_price - open_adj, open_adj)

    # 涨跌停过滤：开盘涨跌幅绝对值超阈值时置零（同时检测涨停和跌停）
    if open_price is not None:
        prev_close = delay(close_price, 1)
        open_change = safe_div(open_price - prev_close, prev_close)
        limit_hit = np.abs(open_change) > limit_move_threshold
        ret[limit_hit] = 0.0

    return ret


# ──────────────────────────────────────────────
# Carry 期限结构
# ──────────────────────────────────────────────


def _resolve_oi_threshold(
    threshold: OIThresholdType,
    symbol: str,
) -> int:
    """解析流动性阈值：支持int/Dict/Callable三种类型。"""
    if isinstance(threshold, int):
        return threshold
    elif isinstance(threshold, dict):
        return threshold.get(symbol, 10000)
    elif callable(threshold):
        return threshold(symbol)
    else:
        return 10000


def compute_carry(
    near_price: np.ndarray,
    far_price: np.ndarray,
    far_oi: Optional[np.ndarray] = None,
    oi_threshold: OIThresholdType = 10000,
    symbol: str = "",
    momentum_window: int = 0,
    close_price: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    期限结构因子（Carry），含流动性过滤和动量正交化。

    CARRY = (近月收盘 - 远月收盘) / 远月收盘

    适用性改造：
      1. 远月持仓量低于阈值时Carry置零（流动性枯竭保护）
      2. 可选动量正交化：滚动回归剥离动量效应，保留纯Carry Alpha

    Args:
        near_price: 近月合约价格序列
        far_price: 远月合约价格序列
        far_oi: 远月合约持仓量序列（流动性过滤）
        oi_threshold: 远月持仓量阈值（支持int/Dict/Callable）
        symbol: 品种代码（用于按品种动态阈值）
        momentum_window: 动量正交化窗口（0=不进行正交化）
        close_price: 收盘价序列（动量正交化需要）

    Returns:
        Carry因子序列（已正交化或原始）
    """
    carry = safe_div(near_price - far_price, far_price)

    # 流动性过滤：远月持仓量不足时置零
    if far_oi is not None:
        resolved_threshold = _resolve_oi_threshold(oi_threshold, symbol)
        low_liquidity = far_oi < resolved_threshold
        carry[low_liquidity] = 0.0

    # 动量正交化：滚动回归剥离动量效应
    if momentum_window > 0 and close_price is not None:
        carry = _orthogonalize_carry(carry, close_price, momentum_window)

    return carry


def _orthogonalize_carry(
    carry: np.ndarray,
    close_price: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Carry因子动量正交化。

    方法：滚动窗口内用动量（N日收益率）对Carry做OLS回归，
    取残差作为正交化后的Carry因子。这确保Carry的Alpha
    不是动量效应的"马甲"。

    Args:
        carry: 原始Carry因子序列
        close_price: 收盘价序列（用于计算动量）
        window: 滚动回归窗口

    Returns:
        正交化后的Carry因子序列
    """
    momentum = safe_div(
        close_price - delay(close_price, window),
        delay(close_price, window),
    )
    # 滚动回归：carry = alpha + beta * momentum + epsilon
    # 取残差 epsilon 作为正交化Carry
    s_carry = pd.Series(carry)
    s_mom = pd.Series(momentum)

    def _rolling_resid(df: pd.DataFrame) -> float:
        """滚动回归取残差"""
        if len(df) < 3:
            return np.nan
        y = df.iloc[:, 0].values
        x = df.iloc[:, 1].values
        # 剔除NaN
        valid = ~(np.isnan(y) | np.isnan(x))
        if valid.sum() < 3:
            return np.nan
        y_v = y[valid]
        x_v = x[valid]
        # OLS: y = a + b*x
        x_with_const = np.column_stack([np.ones(len(x_v)), x_v])
        try:
            beta = np.linalg.lstsq(x_with_const, y_v, rcond=None)[0]
            resid = y_v[-1] - beta[0] - beta[1] * x_v[-1]
            return resid
        except np.linalg.LinAlgError:
            return np.nan

    combined = pd.DataFrame({"carry": s_carry, "momentum": s_mom})
    result = combined.rolling(window=window, min_periods=3).apply(
        lambda df: _rolling_resid(df),
        raw=False,
    )
    return result.iloc[:, 0].values


# ──────────────────────────────────────────────
# OI 安全清洗
# ──────────────────────────────────────────────


def compute_oi_safe(
    open_interest: np.ndarray,
    is_dominant: Optional[np.ndarray] = None,
    delivery_exclude: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    安全持仓量（适用性改造核心）。

    1. 主力合约换月陷阱：换月日前后OI数据置NaN
    2. 交割月强制平仓：进入交割月前N天OI数据置NaN

    Args:
        open_interest: 原始持仓量序列
        is_dominant: 是否主力合约标记序列（True=主力）
        delivery_exclude: 交割月剔除标记序列（True=需剔除）

    Returns:
        清洗后的持仓量序列
    """
    oi = open_interest.copy().astype(float)

    # 换月日检测：is_dominant从True变False或反之
    # 适用性改造：换月日前后3天置NaN，避免DELTA(OI,1)等产生虚假信号
    if is_dominant is not None:
        is_dom = np.asarray(is_dominant, dtype=bool)
        for i in range(1, len(is_dom)):
            if is_dom[i] != is_dom[i - 1]:
                # 换月日及前后3天置NaN
                for j in range(max(0, i - 3), min(len(oi), i + 4)):
                    oi[j] = np.nan

    # 交割月剔除
    if delivery_exclude is not None:
        oi[delivery_exclude] = np.nan

    return oi


def generate_delivery_exclude(
    dates: np.ndarray,
    contract_month: np.ndarray,
    exclude_days: int = 5,
) -> np.ndarray:
    """
    自动生成交割月剔除标记。

    根据合约月份和日期，在进入交割月前N个交易日标记为需剔除。
    交割月定义为合约月份当月，剔除范围从交割月第一个交易日
    往前推 exclude_days 个交易日。

    Args:
        dates: 交易日期序列（datetime64 或可转换为 pd.Timestamp）
        contract_month: 合约月份序列（格式：'YYYYMM' 或 int）
        exclude_days: 交割月前剔除天数

    Returns:
        bool 数组，True 表示该交易日需剔除
    """
    dates = pd.to_datetime(dates)
    n = len(dates)
    exclude = np.zeros(n, dtype=bool)

    # 获取所有不同的合约月份
    unique_months = np.unique(contract_month)

    for cm in unique_months:
        # 解析合约月份
        cm_str = str(cm)
        if len(cm_str) == 6:
            year, month = int(cm_str[:4]), int(cm_str[4:6])
        else:
            continue

        # 交割月第一天
        delivery_start = pd.Timestamp(year=year, month=month, day=1)

        # 找到交割月第一个交易日的索引
        delivery_idx = None
        for i in range(n):
            if dates[i] >= delivery_start:
                delivery_idx = i
                break

        if delivery_idx is None:
            continue

        # 从交割月第一个交易日往前推 exclude_days 天
        start_idx = max(0, delivery_idx - exclude_days)
        exclude[start_idx : delivery_idx + 1] = True

    return exclude


def adjust_price_for_roll(
    close: np.ndarray,
    roll_map: np.ndarray,
) -> np.ndarray:
    """
    根据换月映射表对价格序列进行向后复权调整。

    当主力合约切换时，新合约价格需要乘以复权因子，
    以消除换月带来的价格跳空。

    Args:
        close: 原始收盘价序列
        roll_map: 换月映射表，1=无换月，-1=换月日（需调整后续价格）

    Returns:
        向后复权后的价格序列

    注意：
        更完整的复权逻辑应使用 core/engine/pybroker_data_source.py
        中的 create_hybrid_data_source() 获取已复权数据。
        此函数仅提供轻量级备用方案。
    """
    adjusted = close.copy().astype(float)
    cum_factor = 1.0

    for i in range(len(roll_map)):
        if roll_map[i] == -1 and i > 0:
            # 换月日：计算复权因子
            if abs(adjusted[i - 1]) > 1e-10 and abs(close[i]) > 1e-10:
                cum_factor *= adjusted[i - 1] / close[i]
        adjusted[i] = close[i] * cum_factor

    return adjusted
