"""
配对选择器 — 维护动态配对池 + 计算每品种配对信号。

职责：
  1. 月度重筛配对（每 N bar 一次协整检验）
  2. 维护缓存：当前有效的 (hedge_ratio, pvalue)
  3. 提供 compute_symbol_scores(close_matrix, current_bar) → {symbol: z}

设计原则：
  - 缓存粒度：以 bar_idx 为单位，已计算过的不再重算
  - 重筛频率：默认每 20 bar 一次（约 1 个月，按日频 bar）
  - 配对数：所有可能的 C(N,2) 组合
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cointegration import (
    CointegrationResult,
    batch_rolling_cointegration,
    rolling_ols_hedge_ratio,
)
from .spread_signal import (
    aggregate_pair_zscores_to_symbols,
    rolling_pair_zscore_matrix,
)


@dataclass
class PairSelectorParams:
    """PairSelector 默认参数。"""
    ols_window: int = 60           # OLS/hedge ratio 窗口
    adf_window: int = 60           # ADF 协整检验窗口
    pvalue_threshold: float = 0.05 # ADF p-value 阈值
    rebalance_interval: int = 20   # 配对重筛间隔（≈ 1 月，按日频 bar）
    zscore_lookback: int = 60      # z-score 计算窗口
    clip_abs: float = 3.0          # 横截面 z-score 裁剪阈值
    min_pairs: int = 2             # 至少有效配对数（达不到则全 0 信号）


DEFAULT_PARAMS = PairSelectorParams()


class PairSelector:
    """配对选择器 — 维护动态配对池 + 输出每品种 z-score。

    用法::

        close_df = pd.DataFrame(...)  # 列=品种，索引=bar
        selector = PairSelector(close_df.columns.tolist(), **DEFAULT_PARAMS)
        for bar_idx in range(close_df.shape[0]):
            scores = selector.compute_symbol_scores(close_df, bar_idx)
            # scores = {"SHFE.AL": 0.8, "SHFE.CU": -0.8, ...}
    """

    def __init__(
        self,
        symbols: List[str],
        params: PairSelectorParams = DEFAULT_PARAMS,
    ) -> None:
        self.symbols = list(symbols)
        self.params = params
        # 全 C(N,2) 配对
        self.all_pairs: List[Tuple[str, str]] = list(combinations(self.symbols, 2))
        # 当前有效配对 + 协整结果（rebalance 时刷新）
        self._valid_pairs: Dict[Tuple[str, str], CointegrationResult] = {}
        self._last_rebalance_bar: int = -10**9
        # 历史 z-score 缓存 {(A, B): np.ndarray} 长度 = N
        self._pair_zscore_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._initialized: bool = False

    def __repr__(self) -> str:
        return (
            f"<PairSelector symbols={len(self.symbols)} "
            f"all_pairs={len(self.all_pairs)} "
            f"valid={len(self._valid_pairs)}>"
        )

    # ────────────────────────────────────────────────────────────
    # 内部：重筛协整
    # ────────────────────────────────────────────────────────────

    def _maybe_rebalance(self, close_df: pd.DataFrame, bar_idx: int) -> None:
        """每 rebalance_interval 重新筛选有效配对。"""
        if bar_idx - self._last_rebalance_bar < self.params.rebalance_interval:
            return
        if bar_idx < self.params.adf_window:
            return
        self._valid_pairs = batch_rolling_cointegration(
            close_df,
            self.all_pairs,
            bar_idx=bar_idx,
            window=self.params.adf_window,
            pvalue_threshold=self.params.pvalue_threshold,
        )
        self._last_rebalance_bar = bar_idx
        # 重筛后清空 z-score 缓存（hedge ratio 变了）
        self._pair_zscore_cache.clear()

    # ────────────────────────────────────────────────────────────
    # 公共：每品种横截面 z-score
    # ────────────────────────────────────────────────────────────

    def compute_symbol_scores(
        self,
        close_df: pd.DataFrame,
        bar_idx: int,
    ) -> Dict[str, float]:
        """计算当前 bar 时刻所有品种的配对横截面 z-score。

        Returns:
            {symbol: net_z_score}，无信号或配对不足时全部为 0.0
        """
        self._maybe_rebalance(close_df, bar_idx)
        if len(self._valid_pairs) < self.params.min_pairs:
            return {s: 0.0 for s in self.symbols}
        # 计算所有有效配对的 z-score
        pair_z = rolling_pair_zscore_matrix(
            close_df, self._valid_pairs,
            lookback=self.params.zscore_lookback,
            end_bar=bar_idx,
        )
        # 聚合到品种
        return aggregate_pair_zscores_to_symbols(
            pair_z, self.symbols, clip_abs=self.params.clip_abs,
        )

    # ────────────────────────────────────────────────────────────
    # 诊断
    # ────────────────────────────────────────────────────────────

    def get_valid_pair_info(self) -> List[Dict[str, object]]:
        """返回当前有效配对的诊断信息（用于日志/分析）。"""
        return [
            {
                "pair": f"{a}/{b}",
                "hedge_ratio": res.hedge_ratio,
                "pvalue": res.pvalue,
                "n_obs": res.n_obs,
            }
            for (a, b), res in self._valid_pairs.items()
        ]
