"""
单品种单策略回测辅助函数。

替代旧的 scripts/run_cta_batch._run_single，
基于 UnifiedFactorPool 统一计算信号。

用于 walk_forward.py、parameter_plateau.py 等验证模块。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger


def _run_single_backtest(
    symbol: str,
    strategy_name: str,
    strategy_params: Dict[str, Any],
    entry_threshold: float = 0.05,
    initial_cash: float = 1_000_000,
    full_start: str = "2020-01-01",
    test_start: str = "2023-01-01",
) -> Optional[Dict[str, Any]]:
    """运行单个品种×单个策略×单组参数的回测。

    用 UnifiedFactorPool 计算信号，模拟简单 CTA 执行（每日调仓），
    替代旧的 scripts/run_cta_batch._run_single + CTAExecutorBuilder。

    Args:
        symbol: 品种代码（如 "SHFE.RB"）
        strategy_name: CTA 策略名（如 "carry", "donchian_breakout"）
        strategy_params: 策略参数 dict
        entry_threshold: 入场信号阈值
        initial_cash: 初始资金
        full_start: 全量数据起始
        test_start: 回测起始

    Returns:
        {total_return_pct, sharpe_ratio, max_drawdown_pct, win_rate_pct, total_trades} 或 None
    """
    try:
        from core.execution.factor_pool import UnifiedFactorPool
        from core.execution.signal_abstraction import DEFAULT_CTA_WEIGHTS
        from core.data import DataLoader
    except ImportError:
        return None

    # 1) 加载数据
    try:
        loader = DataLoader(data_source="tqsdk", symbols=[symbol])
        loader.load_data(show_progress=False)
        loader.identify_dominant_contracts()
        loader.build_continuous_series()
        df = loader.get_pybroker_df()
    except Exception:
        try:
            import os
            loader = DataLoader(data_source="csv", data_dir=os.environ.get("DATA_DIR", "data"))
            loader.load_data(show_progress=False)
            df = loader.get_pybroker_df()
        except Exception:
            return None

    if df is None or len(df) < 100:
        return None

    # 2) 计算信号
    pool = UnifiedFactorPool()
    try:
        strat_params = {strategy_name: strategy_params}
        signal_df = pool.compute_all(df, symbol, strategy_params=strat_params)
    except Exception:
        return None

    if strategy_name not in signal_df.columns:
        logger.warning(f"{symbol} {strategy_name}: 信号列不存在")
        return None

    # 3) 模拟简单执行
    signal = signal_df[strategy_name].values
    close = df["close"].values if "close" in df.columns else None
    if close is None or len(close) < 30:
        return None

    # 对齐信号和数据
    n = min(len(close), len(signal))
    close = close[-n:]
    signal = signal[-n:]

    ret = np.diff(close) / close[:-1]
    pos = np.where(signal[1:] > entry_threshold, 1,
                   np.where(signal[1:] < -entry_threshold, -1, 0))
    strat_ret = pos * ret

    # 4) 计算指标
    mask = ~np.isnan(strat_ret) & ~np.isnan(ret)
    strat_ret_clean = strat_ret[mask]

    total_return = float(np.prod(1 + strat_ret_clean) - 1) if len(strat_ret_clean) > 0 else 0.0
    sharpe = float(np.mean(strat_ret_clean) / np.std(strat_ret_clean) * np.sqrt(252)) \
        if len(strat_ret_clean) > 5 and np.std(strat_ret_clean) > 1e-10 else 0.0

    # 最大回撤
    cum = np.cumprod(1 + strat_ret_clean) if len(strat_ret_clean) > 0 else np.array([1.0])
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    # 胜率
    wins = strat_ret_clean[strat_ret_clean > 0]
    total_trades = int(np.sum(np.abs(np.diff(pos.astype(float))) > 0.5))
    win_rate = float(len(wins) / max(len(strat_ret_clean), 1))

    return {
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(abs(max_dd) * 100, 2),
        "win_rate_pct": round(win_rate * 100, 1),
        "total_trades": total_trades,
    }
