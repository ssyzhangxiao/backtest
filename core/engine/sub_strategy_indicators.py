"""
子策略 PyBroker 指标构建器。

⚠️ **路径 C→A 合并（2026-06-07）**：
  原 5 个 `build_xxx_indicators` 函数各自实现了一套**裸价算法**（pct_change / tanh / ADX / rolling corr），
  与 `core/factors/alpha_futures/sub_strategy_aggregator.compute_sub_strategy_scores_from_ohlcv`
  路径 A 的因子库算法**完全不同**。这导致：
    - 主回测（路径 C）与因子验证（路径 A）得到不同的 5 子策略得分
    - 同一 close 序列的 trend_signal 在两条路径下数值不一致
    - 违反规则17（不重复造轮子）

  整改后：所有 5 个 build 函数**统一调用** `compute_sub_strategy_scores_from_ohlcv`（路径 A），
  从结果 DataFrame 提取对应列作为 PyBroker 指标输出。

P1-任务6 整改：从 strategy_indicators.py 中移出，每个子策略的指标构建函数按子策略独立管理。
P1-任务7 整改：所有注册动作必须通过显式调用 register_default_indicators() 完成，禁止模块加载时自动调用。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.engine.strategy_indicators import (
    StrategyIndicatorRegistry,
    StrategyExitHookRegistry,
)
from core.factors.alpha_futures.sub_strategy_aggregator import (
    compute_sub_strategy_scores_from_ohlcv,
)
from core.factors.alpha_futures.config import AlphaFuturesConfig


# ---------------------------------------------------------------------------
# 内部辅助：从 PyBroker bar_data 提取 OHLCV DataFrame
# ---------------------------------------------------------------------------


def _ohlcv_from_bar(bar_data) -> Optional[pd.DataFrame]:
    """
    从 PyBroker bar_data 提取 OHLCV DataFrame（缺失字段返回 None）。

    bar_data 通常具有 high/low/open/close/volume/open_interest 等 numpy 数组属性。
    """
    try:
        close = getattr(bar_data, "close", None)
        high = getattr(bar_data, "high", None)
        low = getattr(bar_data, "low", None)
        if close is None or high is None or low is None:
            return None
        n = len(close)
        open_ = getattr(bar_data, "open", None)
        if open_ is None:
            open_ = close
        volume = getattr(bar_data, "volume", None)
        if volume is None:
            volume = np.zeros(n)
        oi = getattr(bar_data, "open_interest", None)
        if oi is None:
            oi = np.zeros(n)
        # 期限结构因子需要：远月合约收盘价（far_close 已在
        # backtest_runner.py 中通过 StaticScope.register_custom_cols 注册）
        fc = getattr(bar_data, "far_close", None)
        if fc is None:
            fc = np.full(n, np.nan, dtype=float)
        else:
            fc = np.asarray(fc, dtype=float)
        dates = getattr(bar_data, "date", None)
        if dates is None:
            dates = pd.date_range("2025-01-01", periods=n, freq="D")
        else:
            dates = pd.to_datetime(dates)
        return pd.DataFrame(
            {
                "date": dates,
                "open": np.asarray(open_, dtype=float),
                "high": np.asarray(high, dtype=float),
                "low": np.asarray(low, dtype=float),
                "close": np.asarray(close, dtype=float),
                "volume": np.asarray(volume, dtype=float),
                "open_interest": np.asarray(oi, dtype=float),
                "far_close": fc,
            }
        )
    except Exception:
        return None


# 共享配置（避免每次创建新实例）
_DEFAULT_CONFIG = AlphaFuturesConfig()


def _signal_from_factor_column(
    bar_data, column: str, strategy_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> np.ndarray:
    """
    通用：调路径 A 的因子聚合器，提取指定列作为 PyBroker 指标输出。

    路径 C→A 合并后，所有 5 个 build_xxx_indicators 内部走此函数，
    保证主回测与因子验证的算法一致性。

    strategy_params 透传 best_params（trend.window 等），让参数化窗口动量通道
    真正生效。
    """
    df = _ohlcv_from_bar(bar_data)
    if df is None or len(df) < 30:
        # 数据不足时返回全零（不污染信号）
        close_arr = getattr(bar_data, "close", None)
        n = len(close_arr) if close_arr is not None else 0
        return np.zeros(n, dtype=float)
    try:
        scored = compute_sub_strategy_scores_from_ohlcv(
            df,
            config=_DEFAULT_CONFIG,
            strategy_params=strategy_params,
        )
        if column not in scored.columns:
            return np.zeros(len(df), dtype=float)
        return scored[column].fillna(0.0).to_numpy()
    except Exception:
        # 因子计算失败时回退到 0（不中断主回测）
        return np.zeros(len(df), dtype=float)


# ---------------------------------------------------------------------------
# 子策略指标构建器（统一走路径 A）
# ---------------------------------------------------------------------------


def build_trend_indicators(params: Dict[str, Any]) -> List[tuple]:
    """构建趋势策略 PyBroker 指标。params 透传 best_params（含 trend.window）。"""
    captured_params = dict(params or {})

    def trend_signal(bar_data):
        return _signal_from_factor_column(
            bar_data, "trend", strategy_params={"trend": captured_params}
        )

    return [("trend_signal", trend_signal)]


def build_term_structure_indicators(params: Dict[str, Any]) -> List[tuple]:
    """
    构建期限结构策略 PyBroker 指标（**走路径 A**）。

    整改记录（2026-06-07）：
      原实现用单合约收盘价相对均线的偏离**模拟**期限结构，与因子库的 TS_01/TS_02/TS_03
      算法不一致。现在统一调 `compute_sub_strategy_scores_from_ohlcv` 提取 `term_structure` 列。
    """
    captured_params = dict(params or {})

    def term_structure_signal(bar_data):
        return _signal_from_factor_column(
            bar_data,
            "term_structure",
            strategy_params={"term_structure": captured_params},
        )

    return [("term_structure_signal", term_structure_signal)]


def build_mean_reversion_indicators(params: Dict[str, Any]) -> List[tuple]:
    """构建均值回归策略 PyBroker 指标（走路径 A）。params 透传 best_params。"""
    captured_params = dict(params or {})

    def mean_reversion_signal(bar_data):
        return _signal_from_factor_column(
            bar_data,
            "mean_reversion",
            strategy_params={"mean_reversion": captured_params},
        )

    return [("mean_reversion_signal", mean_reversion_signal)]


def build_vol_breakout_indicators(params: Dict[str, Any]) -> List[tuple]:
    """构建波动率突破策略 PyBroker 指标（走路径 A）。params 透传 best_params。"""
    captured_params = dict(params or {})

    def vol_breakout_signal(bar_data):
        return _signal_from_factor_column(
            bar_data,
            "vol_breakout",
            strategy_params={"vol_breakout": captured_params},
        )

    return [("vol_breakout_signal", vol_breakout_signal)]


def build_composite_indicators(params: Dict[str, Any]) -> List[tuple]:
    """构建复合共振策略 PyBroker 指标（走路径 A）。params 透传 best_params。"""
    captured_params = dict(params or {})

    def composite_signal(bar_data):
        return _signal_from_factor_column(
            bar_data,
            "composite_resonance",
            strategy_params={"composite_resonance": captured_params},
        )

    return [("composite_signal", composite_signal)]


# ---------------------------------------------------------------------------
# 退出钩子
# ---------------------------------------------------------------------------


def _term_structure_exit_checker(ctx, indicator_values, strategy_params):
    """期限结构策略退出：价差收敛时平仓（示例退出规则）。"""
    del ctx  # 接口保留，示例实现未使用
    ts_val = indicator_values.get("term_structure_signal")
    if ts_val is None:
        return False
    exit_thr = strategy_params.get("term_structure", {}).get("exit_threshold", 0.2)
    return abs(float(ts_val)) < exit_thr


# ---------------------------------------------------------------------------
# 显式注册函数（入口）
# ---------------------------------------------------------------------------

_REGISTERED = False


def register_default_indicators() -> None:
    """
    显式注册5子策略的指标构建函数。

    P1-任务7整改：必须由调用方显式触发，
    禁止在 import 阶段自动执行，避免隐式副作用。
    重复调用幂等。
    """
    global _REGISTERED
    if _REGISTERED:
        return

    StrategyIndicatorRegistry.register(
        "trend",
        build_trend_indicators,
        indicator_names=["trend_signal"],
        indicator_to_factor={"trend_signal": "trend"},
    )
    StrategyIndicatorRegistry.register(
        "term_structure",
        build_term_structure_indicators,
        indicator_names=["term_structure_signal"],
        indicator_to_factor={"term_structure_signal": "term_structure"},
    )
    StrategyIndicatorRegistry.register(
        "mean_reversion",
        build_mean_reversion_indicators,
        indicator_names=["mean_reversion_signal"],
        indicator_to_factor={"mean_reversion_signal": "mean_reversion"},
    )
    StrategyIndicatorRegistry.register(
        "vol_breakout",
        build_vol_breakout_indicators,
        indicator_names=["vol_breakout_signal"],
        indicator_to_factor={"vol_breakout_signal": "vol_breakout"},
    )
    StrategyIndicatorRegistry.register(
        "composite_resonance",
        build_composite_indicators,
        indicator_names=["composite_signal"],
        indicator_to_factor={"composite_signal": "composite_resonance"},
    )

    StrategyExitHookRegistry.register(
        "term_structure",
        _term_structure_exit_checker,
        reason="价差收敛平仓",
    )

    _REGISTERED = True


def unregister_default_indicators() -> None:
    """
    显式注销5子策略的指标构建函数（主要用于测试）。
    """
    global _REGISTERED
    StrategyIndicatorRegistry.clear()
    StrategyExitHookRegistry.clear()
    _REGISTERED = False
