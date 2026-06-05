"""
基础因子模块。

从 OHLCV 数据计算 ts_momentum / roll_yield / alpha019 / alpha032 四个因子。
这些因子是简化版实现，用于回测快速验证和因子稳定性分析。

正式因子体系见 core/factors/alpha_futures_*.py。
"""

import numpy as np
import pandas as pd

from utils.indicators import compute_atr


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

    # ── ts_momentum: 20日收益率归一化 ──
    ret_20 = close.pct_change(20)
    # 委托公共 compute_atr
    atr = compute_atr(df["high"], df["low"], close, atr_period)
    mom_norm = np.clip(ret_20 / (atr / close + 1e-8) * 0.1, -1.0, 1.0)
    df["ts_momentum"] = mom_norm.fillna(0.0)

    # ── roll_yield: 价格偏离20日均线 ──
    sma_20 = close.rolling(20, min_periods=1).mean()
    spread_pct = (close - sma_20) / (sma_20 + 1e-8) * 100
    df["roll_yield"] = np.clip(spread_pct / 5.0, -1.0, 1.0).fillna(0.0)

    # ── alpha019: 简化版短期反转 × 长期动量排名 ──
    short_term = close - close.shift(7) + (close - close.shift(7)).shift(7)
    sign_component = -np.sign(short_term.fillna(0.0))
    returns = close.pct_change()
    cum_ret_250 = returns.rolling(250, min_periods=1).apply(
        lambda x: np.prod(1 + x) - 1, raw=False
    )
    cum_rank = cum_ret_250.rank(pct=True).fillna(0.5)
    df["alpha019"] = np.clip(sign_component * (1 + cum_rank) * 0.3, -1.0, 1.0).fillna(0.0)

    # ── alpha032: 简化版 收盘价与VWAP的相关系数 ──
    typical_price = (df["high"] + df["low"] + close) / 3
    vwap = (typical_price * df["volume"]).rolling(10, min_periods=1).sum() / \
           df["volume"].rolling(10, min_periods=1).sum().replace(0, 1)
    corr_vwap = close.rolling(10, min_periods=1).corr(vwap)
    df["alpha032"] = np.clip(corr_vwap.fillna(0.0), -1.0, 1.0)

    # ── 前瞻收益（5日） ──
    df["forward_return"] = close.shift(-5) / close - 1.0

    return df
