"""
四因子 CTA 指标注册表（2026-06-19）。

将 SignalAbstractionLayer.get_four_factor_signal 暴露为 PyBroker 指标，
供 backtest_runner / e12_four_factor 实验调用。

四因子 = 动量(donchian_breakout) + 期限结构(carry) + 基差动量 + 仓单变化率

注册模式（与 sub_strategy_indicators.py 一致）：
  from core.engine.strategy_indicators import StrategyIndicatorRegistry
  from core.execution.four_factor_indicators import register_four_factor_indicators

  register_four_factor_indicators()
  spec = StrategyIndicatorRegistry.get("four_factor")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.engine.strategy_indicators import StrategyIndicatorRegistry
from core.engine.sub_strategy_adapter import ohlcv_from_bar

logger = logging.getLogger(__name__)

# 四因子策略的默认权重（必须与 config.yaml 的 four_factor.weights 保持一致）
DEFAULT_FOUR_FACTOR_WEIGHTS: Dict[str, float] = {
    "donchian_breakout": 0.30,
    "carry": 0.25,
    "basis_momentum": 0.25,
    "receipt_change": 0.20,
}


# ═══════════════════════════════════════════════════════════════
# 向量化四因子信号计算（用于 PyBroker 指标）
# ═══════════════════════════════════════════════════════════════


def compute_four_factor_signals_vectorized(
    ohlcv: pd.DataFrame,
    symbol: str,
    weights: Optional[Dict[str, float]] = None,
    basis_window: int = 20,
    receipt_window: int = 20,
    receipt_data: Optional[pd.Series] = None,
) -> np.ndarray:
    """
    向量化计算每个 bar 的四因子融合信号（无 look-ahead）。

    算法（参考 SignalAbstractionLayer.get_four_factor_signal）：
      1. 一次性计算 4 个原始信号序列（donchian_breakout / carry / basis_momentum / receipt_change）
      2. 每个 bar 截取到当前为止的历史，避免未来信息
      3. 用当前 bar 的 4 个信号值按权重融合 → 输出单个标量（已 clip 到 [-1, 1]）

    Args:
        ohlcv: 含 date / close / high / low / volume / (far_close?) 列的 DataFrame，按日期升序
        symbol: 品种代码
        weights: 4 因子权重，None 时用 DEFAULT_FOUR_FACTOR_WEIGHTS
        basis_window: 基差动量窗口
        receipt_window: 仓单变化率窗口
        receipt_data: 仓单数据（pd.Series，按日期索引），None 时 receipt_change = 0

    Returns:
        np.ndarray，长度 = len(ohlcv)，前 30 个 warmup bar = 0
    """
    if ohlcv is None or ohlcv.empty:
        return np.zeros(0, dtype=float)

    w = dict(weights or DEFAULT_FOUR_FACTOR_WEIGHTS)
    n = len(ohlcv)
    out = np.zeros(n, dtype=float)

    close = ohlcv["close"].to_numpy(dtype=float)
    high = (
        ohlcv["high"].to_numpy(dtype=float)
        if "high" in ohlcv.columns
        else close.copy()
    )
    low = (
        ohlcv["low"].to_numpy(dtype=float)
        if "low" in ohlcv.columns
        else close.copy()
    )

    # 1) donchian_breakout：N 日最高/最低通道突破
    donchian_arr = _compute_donchian_signal(close, high, low, window=20)

    # 2) carry：期限结构（近月-远月）信号
    carry_arr = _compute_carry_signal(ohlcv, window=20)

    # 3) basis_momentum：基差动量（依赖 far_close）
    if "far_close" in ohlcv.columns and ohlcv["far_close"].notna().any():
        from core.factors.basis_momentum import compute_basis_momentum

        basis_arr = compute_basis_momentum(
            close,
            far_close=ohlcv["far_close"].to_numpy(dtype=float),
            basis_window=basis_window,
        )
    else:
        basis_arr = np.zeros(n, dtype=float)

    # 4) receipt_change：仓单变化率
    if receipt_data is not None and not receipt_data.empty:
        try:
            from core.factors.receipt_factor import compute_receipt_factor_signal

            df_index = (
                pd.to_datetime(ohlcv["date"])
                if "date" in ohlcv.columns
                else pd.to_datetime(ohlcv.index)
            )
            receipt_aligned = receipt_data.reindex(df_index)
            receipt_series = compute_receipt_factor_signal(
                receipt_aligned, window=receipt_window,
            )
            receipt_arr = receipt_series.to_numpy(dtype=float)
            receipt_arr = np.where(np.isfinite(receipt_arr), receipt_arr, 0.0)
        except Exception as e:  # noqa: BLE001
            logger.warning("receipt_change 计算失败（%s），使用 0", e)
            receipt_arr = np.zeros(n, dtype=float)
    else:
        receipt_arr = np.zeros(n, dtype=float)

    # 5) 逐 bar 融合（仅看历史，warmup = 30 bar）
    w_d = float(w.get("donchian_breakout", 0.0))
    w_c = float(w.get("carry", 0.0))
    w_b = float(w.get("basis_momentum", 0.0))
    w_r = float(w.get("receipt_change", 0.0))

    for i in range(30, n):
        d = float(np.clip(donchian_arr[i], -1.0, 1.0)) if np.isfinite(donchian_arr[i]) else 0.0
        c = float(np.clip(carry_arr[i], -1.0, 1.0)) if np.isfinite(carry_arr[i]) else 0.0
        b = float(np.clip(basis_arr[i], -1.0, 1.0)) if np.isfinite(basis_arr[i]) else 0.0
        r = float(np.clip(receipt_arr[i], -1.0, 1.0)) if np.isfinite(receipt_arr[i]) else 0.0
        raw = d * w_d + c * w_c + b * w_b + r * w_r
        out[i] = float(np.clip(raw, -1.0, 1.0))

    return out


def _compute_donchian_signal(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, window: int = 20,
) -> np.ndarray:
    """Donchian 通道突破信号：close 在 N 日 high 通道上方 → +1，下方 → -1。"""
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    for i in range(window, n):
        hh = np.nanmax(high[i - window: i])
        ll = np.nanmin(low[i - window: i])
        rng = hh - ll
        if not np.isfinite(rng) or rng < 1e-8:
            out[i] = 0.0
            continue
        # 标准化到 [-1, 1]：收盘在通道中位附近 → 0
        pos = (close[i] - (hh + ll) / 2.0) / (rng / 2.0)
        out[i] = float(np.clip(pos, -1.0, 1.0))
    return out


def _compute_carry_signal(ohlcv: pd.DataFrame, window: int = 20) -> np.ndarray:
    """期限结构信号：远月-近月 基差率（无 far_close 时回退到 0）。"""
    n = len(ohlcv)
    out = np.zeros(n, dtype=float)
    if "far_close" not in ohlcv.columns or not ohlcv["far_close"].notna().any():
        return out
    close = ohlcv["close"].to_numpy(dtype=float)
    far = ohlcv["far_close"].to_numpy(dtype=float)
    for i in range(window, n):
        c = close[i]
        f = far[i]
        if not (np.isfinite(c) and np.isfinite(f)) or c < 1e-8:
            out[i] = 0.0
            continue
        # 基差率：(far - near) / near
        basis_rate = (f - c) / c
        out[i] = float(np.clip(basis_rate * 10.0, -1.0, 1.0))  # 放大 10 倍
    return out


# ═══════════════════════════════════════════════════════════════
# PyBroker 指标构建器
# ═══════════════════════════════════════════════════════════════


def build_four_factor_indicators(params: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """
    构建四因子 CTA PyBroker 指标。

    Args:
        params: 字典，含以下键：
          - weights: 4 因子权重（可选）
          - basis_window: 基差动量窗口
          - receipt_window: 仓单变化率窗口
          - receipt_data: 仓单数据 Series 字典 {symbol: Series}（可选）

    Returns:
        [("four_factor_signal", fn)] 列表
    """
    weights = params.get("weights", DEFAULT_FOUR_FACTOR_WEIGHTS)
    basis_window = int(params.get("basis_window", 20))
    receipt_window = int(params.get("receipt_window", 20))
    receipt_data_map: Dict[str, pd.Series] = params.get("receipt_data", {}) or {}
    captured_symbol = str(params.get("symbol", ""))

    def four_factor_signal(bar_data) -> np.ndarray:
        ohlcv = ohlcv_from_bar(bar_data)
        if ohlcv is None or len(ohlcv) < 30:
            close = getattr(bar_data, "close", None)
            return np.zeros(
                len(close) if close is not None else 0, dtype=float,
            )
        # 尝试从 bar_data 推断 symbol（PyBroker 不会直接传，由 data 列推断）
        sym = captured_symbol
        if not sym and "symbol" in ohlcv.columns:
            sym = str(ohlcv["symbol"].iloc[-1])
        receipt_series = receipt_data_map.get(sym) if sym else None
        return compute_four_factor_signals_vectorized(
            ohlcv,
            symbol=sym or "UNKNOWN",
            weights=weights,
            basis_window=basis_window,
            receipt_window=receipt_window,
            receipt_data=receipt_series,
        )

    return [("four_factor_signal", four_factor_signal)]


# ═══════════════════════════════════════════════════════════════
# 注册函数
# ═══════════════════════════════════════════════════════════════

_FOUR_FACTOR_REGISTERED = False


def register_four_factor_indicators() -> None:
    """显式注册四因子指标构建函数。重复调用幂等。"""
    global _FOUR_FACTOR_REGISTERED
    if _FOUR_FACTOR_REGISTERED:
        return
    StrategyIndicatorRegistry.register(
        "four_factor",
        build_four_factor_indicators,
        indicator_names=["four_factor_signal"],
        indicator_to_factor={"four_factor_signal": "four_factor"},
    )
    _FOUR_FACTOR_REGISTERED = True


def unregister_four_factor_indicators() -> None:
    """注销四因子指标（测试用）。"""
    global _FOUR_FACTOR_REGISTERED
    if not _FOUR_FACTOR_REGISTERED:
        return
    spec = StrategyIndicatorRegistry._specs.pop("four_factor", None)
    StrategyIndicatorRegistry._builders.pop("four_factor", None)
    _FOUR_FACTOR_REGISTERED = False
    del spec
