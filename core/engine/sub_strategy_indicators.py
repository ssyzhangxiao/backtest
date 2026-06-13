"""
子策略 PyBroker 指标构建器。

⚠️ **路径 C→A 合并（2026-06-07 → 2026-06-13）**：
  所有 build 函数走 `signal_from_factor_column`（由 sub_strategy_adapter.py 导出），
  保证主回测与因子验证的算法一致性。

  2026-06-13：辅助函数 _ohlcv_from_bar / _signal_from_factor_column 已迁入
  sub_strategy_adapter.py，本文件仅保留指标构建 + 注册逻辑。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.engine.strategy_indicators import (
    StrategyIndicatorRegistry,
    StrategyExitHookRegistry,
)
from core.engine.sub_strategy_adapter import signal_from_factor_column


# ---------------------------------------------------------------------------
# 子策略指标构建器（统一走 signal_from_factor_column）
# ---------------------------------------------------------------------------


def build_trend_indicators(params: Dict[str, Any]) -> List[tuple]:
    """构建趋势策略 PyBroker 指标。"""
    captured_params = dict(params or {})
    def trend_signal(bar_data):
        return signal_from_factor_column(
            bar_data, "trend", strategy_params={"trend": captured_params},
        )
    return [("trend_signal", trend_signal)]


def build_term_structure_indicators(params: Dict[str, Any]) -> List[tuple]:
    """构建期限结构策略 PyBroker 指标。"""
    captured_params = dict(params or {})
    def term_structure_signal(bar_data):
        return signal_from_factor_column(
            bar_data, "term_structure",
            strategy_params={"term_structure": captured_params},
        )
    return [("term_structure_signal", term_structure_signal)]


def build_mean_reversion_indicators(params: Dict[str, Any]) -> List[tuple]:
    """构建均值回归策略 PyBroker 指标。"""
    captured_params = dict(params or {})
    def mean_reversion_signal(bar_data):
        return signal_from_factor_column(
            bar_data, "mean_reversion",
            strategy_params={"mean_reversion": captured_params},
        )
    return [("mean_reversion_signal", mean_reversion_signal)]


def build_vol_breakout_indicators(params: Dict[str, Any]) -> List[tuple]:
    """构建波动率突破策略 PyBroker 指标。"""
    captured_params = dict(params or {})
    def vol_breakout_signal(bar_data):
        return signal_from_factor_column(
            bar_data, "vol_breakout",
            strategy_params={"vol_breakout": captured_params},
        )
    return [("vol_breakout_signal", vol_breakout_signal)]


def build_composite_indicators(params: Dict[str, Any]) -> List[tuple]:
    """构建复合共振策略 PyBroker 指标。"""
    captured_params = dict(params or {})
    def composite_signal(bar_data):
        return signal_from_factor_column(
            bar_data, "composite_resonance",
            strategy_params={"composite_resonance": captured_params},
        )
    return [("composite_signal", composite_signal)]


# ---------------------------------------------------------------------------
# 退出钩子
# ---------------------------------------------------------------------------


def _term_structure_exit_checker(ctx, indicator_values, strategy_params):
    """期限结构策略退出：价差收敛时平仓。"""
    del ctx
    ts_val = indicator_values.get("term_structure_signal")
    if ts_val is None:
        return False
    exit_thr = strategy_params.get("term_structure", {}).get("exit_threshold", 0.2)
    return abs(float(ts_val)) < exit_thr


# ---------------------------------------------------------------------------
# 显式注册函数
# ---------------------------------------------------------------------------

_REGISTERED = False


def register_default_indicators() -> None:
    """显式注册 5 子策略的指标构建函数和退出钩子。重复调用幂等。"""
    global _REGISTERED
    if _REGISTERED:
        return

    for name, builder, names, mapping in [
        ("trend", build_trend_indicators, ["trend_signal"], {"trend_signal": "trend"}),
        ("term_structure", build_term_structure_indicators, ["term_structure_signal"],
         {"term_structure_signal": "term_structure"}),
        ("mean_reversion", build_mean_reversion_indicators, ["mean_reversion_signal"],
         {"mean_reversion_signal": "mean_reversion"}),
        ("vol_breakout", build_vol_breakout_indicators, ["vol_breakout_signal"],
         {"vol_breakout_signal": "vol_breakout"}),
        ("composite_resonance", build_composite_indicators, ["composite_signal"],
         {"composite_signal": "composite_resonance"}),
    ]:
        StrategyIndicatorRegistry.register(name, builder, indicator_names=names,
                                            indicator_to_factor=mapping)

    StrategyExitHookRegistry.register(
        "term_structure", _term_structure_exit_checker, reason="价差收敛平仓",
    )
    _REGISTERED = True


def unregister_default_indicators() -> None:
    """显式注销 5 子策略的指标构建函数（主要用于测试）。"""
    global _REGISTERED
    StrategyIndicatorRegistry.clear()
    StrategyExitHookRegistry.clear()
    _REGISTERED = False
