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

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.ext.factors.alpha_futures.config import AlphaFuturesConfig
from core.ext.factors.alpha_futures.sub_strategy_aggregator import (
    DEFAULT_FACTOR_NAMES,
    compute_sub_strategy_scores_from_ohlcv,
)
from core.strategies.cta.registry import CTA_STRATEGY_REGISTRY

_logger = logging.getLogger(__name__)

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

# ── 四因子 CTA 升级（2026-06-19）新增的独立信号 ──
# - basis_momentum：基差动量（依赖 far_close 列，无数据返回 0）
# - receipt_change：仓单变化率（依赖外部 receipt_data 输入）
_FACTOR_PRIMARY_NAMES: List[str] = [
    "basis_momentum",
    "receipt_change",
]

# 全部信号列名（11 个 CTA 策略 + 2 个四因子信号 = 13 列）
CTA_SIGNAL_NAMES: List[str] = list(_CTA_PRIMARY_NAMES) + list(_FACTOR_PRIMARY_NAMES)
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
        receipt_data: Optional[pd.Series] = None,
        receipt_window: Optional[int] = None,
        basis_window: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """计算所有 CTA 策略在全部 bar 上的信号序列。

        输入 DataFrame 必须含 close / high / low / volume 列。
        对于 spread 依赖策略，需含 spread 或 far_close 列（可选）。

        四因子新增（2026-06-19）：
          - basis_momentum：依赖 far_close 列（无则全 0）
          - receipt_change：依赖外部传入的 receipt_data（pd.Series，按日期索引）

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

        # ── 四因子新增：基差动量 + 仓单变化率（独立计算，无需策略实例） ──
        from core.factors.basis_momentum import compute_basis_momentum
        from core.factors.receipt_factor import compute_receipt_factor_signal

        # basis_momentum：依赖 far_close 列
        if "far_close" in df.columns:
            basis_arr = compute_basis_momentum(
                close,
                far_close=far_arr if far_arr is not None else df["far_close"].to_numpy(dtype=float),
                basis_window=basis_window,
            )
        else:
            basis_arr = np.zeros(n, dtype=float)
        results["basis_momentum"] = basis_arr

        # receipt_change：依赖外部 receipt_data
        if receipt_data is not None and not receipt_data.empty:
            try:
                # 将 receipt_data 对齐到 df 的索引
                # 2026-06-19：去除 time 成分（OHLCV 16:00 vs receipt 00:00 不匹配）
                if "date" in df.columns:
                    df_index = pd.to_datetime(df["date"]).dt.normalize()
                else:
                    df_index = pd.to_datetime(df.index).normalize()
                # 同步去除 receipt index 的 time
                receipt_norm_index = (
                    pd.to_datetime(receipt_data.index).normalize()
                )
                receipt_data_aligned = pd.Series(
                    receipt_data.values, index=receipt_norm_index,
                )
                # 去重（同一日期多条取最后）
                # ⚠️ 2026-06-20 文档：若同一日有多个仓单快照（如盘前/盘后两次更新），
                # `keep="last"` 取最后值，在回测当日可能引入轻微未来数据偏倚。
                # 实际影响低：① 仓单日级数据 1 天 1 值为常态；
                # ② df_index 已 normalize 到日级，reindex 不会跨日错配。
                # 上游若需严格无偏，请保证 `receipt_data` 已是 T-1 收盘后快照。
                receipt_data_aligned = receipt_data_aligned[
                    ~receipt_data_aligned.index.duplicated(keep="last")
                ]
                receipt_aligned = receipt_data_aligned.reindex(df_index)
                receipt_signal_series = compute_receipt_factor_signal(
                    receipt_aligned, window=receipt_window,
                )
                receipt_arr = receipt_signal_series.to_numpy(dtype=float)
                # 缺失日期（NaN）填 0
                receipt_arr = np.where(np.isfinite(receipt_arr), receipt_arr, 0.0)
            except Exception as e:  # noqa: BLE001
                _logger.warning("receipt_change 计算失败（%s），使用 0 信号", e)
                receipt_arr = np.zeros(n, dtype=float)
        else:
            receipt_arr = np.zeros(n, dtype=float)
        results["receipt_change"] = receipt_arr

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
        # 仓单数据缓存：{symbol: pd.Series}（2026-06-19 四因子新增）
        self._receipt_cache: Dict[str, pd.Series] = {}
        # 默认窗口（可通过 setter 调整）
        self._default_receipt_window: int = 20
        self._default_basis_window: int = 20

    # ── 公共接口 ──

    def preload_receipt_data(
        self,
        receipt_data: Dict[str, pd.Series],
        receipt_window: int = 20,
        basis_window: int = 20,
    ) -> None:
        """预加载所有品种的仓单数据。

        调用后 compute_all() / compute_signals_for_bar() 内部自动应用。
        Args:
            receipt_data: {symbol: pd.Series(date-indexed)}
            receipt_window: 仓单因子窗口
            basis_window: 基差动量窗口
        """
        self._receipt_cache = dict(receipt_data or {})
        self._default_receipt_window = int(receipt_window)
        self._default_basis_window = int(basis_window)
        # 清空信号缓存以触发重算
        self._cache.clear()
        _logger.info(
            "[FactorPool] preload receipt_data: %d symbols, window=%d/%d",
            len(self._receipt_cache), self._default_receipt_window, self._default_basis_window,
        )

    def get_receipt_data(self, symbol: str) -> Optional[pd.Series]:
        """获取指定品种的仓单数据（外部读取）。"""
        return self._receipt_cache.get(symbol)

    def compute_all(
        self,
        ohlcv: pd.DataFrame,
        symbol: str,
        strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
        receipt_data: Optional[pd.Series] = None,
        receipt_window: Optional[int] = None,
        basis_window: Optional[int] = None,
    ) -> pd.DataFrame:
        """计算所有信号，返回含 11 列 + forward_return 的 DataFrame。

        Args:
            ohlcv: 含 date, open, high, low, close, volume(可选) 的 DataFrame
            symbol: 品种代码（用于 CTA 策略状态管理）
            strategy_params: 子策略参数（透传给 sub_strategy_aggregator）
            receipt_data: 仓单日度序列（pd.Series，按日期索引），用于仓单因子
            receipt_window: 仓单因子窗口（默认 20）
            basis_window: 基差动量窗口（默认 20）

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

        # 2) 7 个 CTA 策略 + 2 个四因子信号
        if self._cta_wrapper is None:
            self._cta_wrapper = _CTABatchWrapper()
        # 优先使用显式传入的 receipt_data，否则从预加载缓存中取
        if receipt_data is None and self._receipt_cache:
            receipt_data = self._receipt_cache.get(symbol)
            # 2026-06-19：兼容 PyBroker 大写短码（如 "rb"）与 cache 中完整代码（如 "SHFE.RB"）
            if receipt_data is None:
                short = symbol.split(".")[-1].lower() if "." in symbol else symbol.lower()
                for cache_key, cache_val in self._receipt_cache.items():
                    if cache_key.split(".")[-1].lower() == short:
                        receipt_data = cache_val
                        break
        # 调试日志：2026-06-20 从 sys.stderr.write 改 logger.debug（避免污染 stderr）
        if "rb" in symbol.lower() or "FG" in symbol:
            _logger.debug(
                f"[FP_COMPUTE] {symbol}: receipt_data={'YES' if receipt_data is not None else 'NO'}, "
                f"cache_keys={list(self._receipt_cache.keys())}"
            )
        # 优先使用显式窗口，否则用预加载默认窗口
        if receipt_window is None:
            receipt_window = self._default_receipt_window
        if basis_window is None:
            basis_window = self._default_basis_window
        cta_results = self._cta_wrapper.compute_all(
            ohlcv,
            symbol,
            receipt_data=receipt_data,
            receipt_window=receipt_window,
            basis_window=basis_window,
        )

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
