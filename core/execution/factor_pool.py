"""
统一因子池 — 单入口计算所有信号源。

架构位置：core/execution/factor_pool.py

设计目标：
  1. 统一 24 Alpha 因子 + 6 CTA 策略的信号计算入口
  2. 输出格式统一的 DataFrame，供 SignalAbstractionLayer 按模式提取
  3. 新增因子/策略只需在注册表中添加，无需修改执行器

三层架构：
  OHLCV → UnifiedFactorPool.compute_all()
           ├── compute_sub_strategy_scores_from_ohlcv()  ← 5 子策略（24 因子聚合）
           └── _CTABatchWrapper.compute_all()             ← 6 个 CTA 策略
         → DataFrame(11 列 + forward_return)

CTA 策略状态管理（_CTABatchWrapper）：
  - carry / pair_trading：需要 spread / far_close 列，batch 计算前预先注入
  - tsi_garch：维护 GARCH sigma 缓存（_call_counter + _model_cache）
  - 其他策略：无状态依赖，逐 bar 计算即可

向后兼容：
  - 不修改任何现有模块
  - 下游仍可通过旧的 compute_sub_strategy_scores_from_ohlcv() / get_cta_strategy() 调用
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.ext.factors.alpha_futures.config import AlphaFuturesConfig
from core.ext.factors.alpha_futures.sub_strategy_aggregator import (
    DEFAULT_FACTOR_NAMES,
    compute_sub_strategy_scores_from_ohlcv,
)
from core.strategies.cta.registry import CTA_STRATEGY_REGISTRY

__all__ = ["UnifiedFactorPool", "ALL_SIGNAL_NAMES", "CTA_SIGNAL_NAMES"]

# ── 7 个 CTA 策略的规范名（去重后取唯一策略名） ──
# CTA_STRATEGY_REGISTRY 中 alias 不重复计算：
#   carry / carry_zscore → carry (取第一个)
#   tsi_garch / state_aware_trend → tsi_garch
#   momentum_ma / simple_trend → momentum_ma
#   oi_signal → 持仓量衍生信号（需 oi 列）
_CTA_PRIMARY_NAMES: List[str] = [
    "carry",
    "vol_mean_reversion",
    "donchian_breakout",
    "momentum_ma",
    "tsi_garch",
    "pair_trading",
    "oi_signal",
]

# 全部信号列名（11 列）
CTA_SIGNAL_NAMES: List[str] = list(_CTA_PRIMARY_NAMES)
ALL_SIGNAL_NAMES: List[str] = DEFAULT_FACTOR_NAMES + CTA_SIGNAL_NAMES

# ── spread 依赖的策略 ──
_SPREAD_DEPENDENT = {"carry", "pair_trading"}


class _CTABatchWrapper:
    """CTA 策略 batch 包装器 — 一次性算完所有 bar。

    内部维护策略实例状态（GARCH 缓存、趋势方向、ADF 检验计数器等）。
    每个品种调用一次 compute_all()，返回 {策略名: signal_array}。
    """

    def __init__(self) -> None:
        # 按规范名缓存策略实例（每个实例跨 bar 保持状态）
        self._strategies: Dict[str, Any] = {}
        for name in _CTA_PRIMARY_NAMES:
            cls = CTA_STRATEGY_REGISTRY.get(name)
            if cls is not None:
                self._strategies[name] = cls()

    def ready(self) -> List[str]:
        """返回已就绪的策略名列表。"""
        return list(self._strategies.keys())

    def compute_all(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> Dict[str, np.ndarray]:
        """计算所有 CTA 策略在全部 bar 上的信号序列。

        输入 DataFrame 必须含 close / high / low / volume 列。
        对于 spread 依赖策略，需含 spread 或 far_close 列（可选）。

        Returns:
            {策略名: np.ndarray(signal)}，长度 = len(df)，
            前 30 个 warmup bar 为 NaN。
        """
        n = len(df)
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float) if "high" in df.columns else close
        low = df["low"].to_numpy(dtype=float) if "low" in df.columns else close
        volume = (
            df["volume"].to_numpy(dtype=float)
            if "volume" in df.columns
            else np.zeros(n)
        )

        # 预提取 spread / far_close（供 spread 依赖策略使用）
        spread_arr: Optional[np.ndarray] = None
        far_arr: Optional[np.ndarray] = None
        if "spread" in df.columns:
            spread_arr = df["spread"].to_numpy(dtype=float)
        if "far_close" in df.columns:
            far_arr = df["far_close"].to_numpy(dtype=float)

        # 预提取 oi（供 oi_signal 策略使用，方向四 P1）
        oi_arr: Optional[np.ndarray] = None
        if "open_interest" in df.columns:
            oi_arr = df["open_interest"].to_numpy(dtype=float)
            # 计算 oi_signal 全量序列
            from core.factors.oi_signal import compute_oi_signal  # noqa: WPS433

            oi_signal_arr = compute_oi_signal(close, oi_arr, window=20)
        else:
            oi_signal_arr = None

        results: Dict[str, np.ndarray] = {}
        for name, strat in self._strategies.items():
            arr = np.full(n, np.nan, dtype=float)

            # spread 依赖策略：注入序列到策略状态
            if name in _SPREAD_DEPENDENT:
                if spread_arr is not None:
                    strat.set_state(symbol, "_spread", spread_arr)
                if far_arr is not None and name == "pair_trading":
                    strat.set_state(symbol, "_far_price", far_arr)

            # oi_signal 策略：注入预计算的 oi_signal 序列
            if name == "oi_signal" and oi_signal_arr is not None:
                strat.set_state(symbol, "_oi_signal", oi_signal_arr)

            # 逐 bar 计算（warmup 30 bar）
            for i in range(30, n):
                try:
                    arr[i] = strat.compute_signal(
                        symbol=symbol,
                        close=close[: i + 1],
                        high=high[: i + 1],
                        low=low[: i + 1],
                        volume=volume[: i + 1] if n > 0 else None,
                    )
                except Exception:
                    arr[i] = 0.0

            results[name] = arr

        return results


class UnifiedFactorPool:
    """统一因子池 — 单入口计算所有信号源。

    用法::

        pool = UnifiedFactorPool()
        signals = pool.compute_all(ohlcv, symbol="SHFE.RB")
        # signals 是含 11 列 + forward_return 的 DataFrame

    设计原则：
      - 同一品种只需调用一次 compute_all()，结果缓存到 _cache 字典
      - compute_signals_for_bar() 从缓存读取最新 bar，避免重复计算
      - 新增因子/策略只需在 CTA_STRATEGY_REGISTRY 或 factor_registry 注册，
        无需修改本类
    """

    def __init__(
        self,
        alpha_config: Optional[AlphaFuturesConfig] = None,
    ) -> None:
        self._alpha_config = alpha_config or AlphaFuturesConfig()
        # CTA batch 包装器（延迟初始化：首次 compute_all 时创建）
        self._cta_wrapper: Optional[_CTABatchWrapper] = None
        # 缓存：{symbol: DataFrame}
        self._cache: Dict[str, pd.DataFrame] = {}

    # ── 公共接口 ──

    def compute_all(
        self,
        ohlcv: pd.DataFrame,
        symbol: str,
        strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """计算所有信号，返回含 11 列 + forward_return 的 DataFrame。

        Args:
            ohlcv: 含 date, open, high, low, close, volume(可选) 的 DataFrame
            symbol: 品种代码（用于 CTA 策略状态管理）
            strategy_params: 子策略参数（透传给 sub_strategy_aggregator）

        Returns:
            DataFrame，列 = ALL_SIGNAL_NAMES + ["forward_return"]
        """
        if symbol in self._cache:
            return self._cache[symbol]

        # 1) 5 子策略（24 Alpha 因子聚合）
        alpha_df = compute_sub_strategy_scores_from_ohlcv(
            ohlcv,
            config=self._alpha_config,
            strategy_params=strategy_params,
        )

        # 2) 6 个 CTA 信号
        if self._cta_wrapper is None:
            self._cta_wrapper = _CTABatchWrapper()
        cta_results = self._cta_wrapper.compute_all(ohlcv, symbol)

        # 3) 合并
        result = alpha_df.copy()
        for name in CTA_SIGNAL_NAMES:
            arr = cta_results.get(name)
            if arr is not None:
                result[name] = arr
            else:
                result[name] = np.nan

        self._cache[symbol] = result
        return result

    def compute_signals_for_bar(
        self,
        ohlcv: pd.DataFrame,
        symbol: str,
        bar_idx: int,
        strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """仅返回最新 bar 的所有信号（性能优化：首次调用缓存全量结果）。"""
        df = self.compute_all(ohlcv, symbol, strategy_params)
        if bar_idx >= len(df):
            return {name: 0.0 for name in ALL_SIGNAL_NAMES}
        return {
            name: float(df[name].iloc[bar_idx])
            for name in ALL_SIGNAL_NAMES
            if name in df.columns
        }

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """清除缓存（品种级或全部）。"""
        if symbol:
            self._cache.pop(symbol, None)
        else:
            self._cache.clear()

    @property
    def cached_symbols(self) -> List[str]:
        """返回已缓存的品种列表。"""
        return list(self._cache.keys())
