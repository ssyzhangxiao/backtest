"""
信号抽象层 — 从统一因子池提取信号，按模式组装。

架构位置：core/execution/signal_abstraction.py

职责：
  1. 屏蔽 UnifiedFactorPool 的内部细节，为上层提供模式化的信号接口
  2. 支持三种模式：CROSS_SECTIONAL / CTA / HYBRID
  3. 标准化输出（所有信号 clip 到 [-1, 1]）
  4. 预留未来冗余：新增信号模式只需在 SignalMode 添加枚举值 + 实现对应方法

三种模式：
  - CROSS_SECTIONAL: 提取 5 子策略得分 → 喂给 FactorScoringEngine 做横截面排名
  - CTA: 提取 6 策略信号 → 支持单信号或加权合成
  - HYBRID: 横截面排名 × CTA 时序信号的互补组合

用法::

    pool = UnifiedFactorPool()
    layer = SignalAbstractionLayer(pool)

    # 横截面模式
    cs = layer.get_cross_sectional_signals(symbol, ohlcv, bar_idx)

    # CTA 模式（合成单信号）
    cta = layer.get_cta_composite_signal(symbol, ohlcv, bar_idx)

    # 混合模式
    hybrid = layer.get_hybrid_signal(symbol, ohlcv, bar_idx, cross_section_z=0.5)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

import logging
import numpy as np
import pandas as pd

from core.execution.factor_pool import (
    CTA_SIGNAL_NAMES,
    DEFAULT_FACTOR_NAMES,
    UnifiedFactorPool,
)
from core.factors.pair_trading import PairSelector, PairSelectorParams

__all__ = [
    "SignalMode",
    "SignalAbstractionLayer",
    "DEFAULT_CTA_WEIGHTS",
]

_logger = logging.getLogger(__name__)

# ── CTA 6 策略固定权重（来自 signal_fusion.py 验证过的权重） ──
DEFAULT_CTA_WEIGHTS: Dict[str, float] = {
    "carry": 0.30,
    "vol_mean_reversion": 0.30,
    "donchian_breakout": 0.20,
    "momentum_ma": 0.10,
    "tsi_garch": 0.05,
    "pair_trading": 0.05,
}


class SignalMode(Enum):
    """信号提取模式枚举。

    新增模式只需在此添加枚举值 + 在 SignalAbstractionLayer 添加对应方法。
    """

    CROSS_SECTIONAL = "cross_sectional"
    """横截面模式：提取 5 子策略信号，不做品种间比较"""

    CTA = "cta"
    """单品种 CTA 模式：提取 6 策略信号"""

    HYBRID = "hybrid"
    """混合模式：横截面排名 × CTA 时序互补（线性加权）"""

    HYBRID_DYNAMIC = "hybrid_dynamic"
    """动态混合模式：横截面作为 CTA 的仓位缩放因子（不减仓不减收益，
    异号时减仓，方向由 CTA 决定）。"""


class SignalAbstractionLayer:
    """信号抽象层 — 从统一因子池到各模式信号的标准化提取。

    所有 public 方法输出均为 clip 到 [-1, 1] 的信号值。
    """

    def __init__(
        self,
        factor_pool: UnifiedFactorPool,
        default_mode: str = "cross_sectional",
        cta_weight: float = 0.5,
        xs_position_base: float = 0.5,
        xs_position_ceiling: float = 1.0,
        xs_opposite_penalty: float = 0.5,
        # ── 方向三：配对交易横截面（2026-06-17） ──
        cross_section_source: str = "default",
        pair_params: Optional[PairSelectorParams] = None,
    ) -> None:
        self._pool = factor_pool
        self.mode = default_mode
        """当前信号模式（可运行时切换）。"""
        self.cta_weight = cta_weight
        """混合模式下 CTA 信号权重（0~1）。"""
        # 动态混合模式参数（方向二：横截面作为仓位缩放因子）
        self.xs_position_base = xs_position_base
        """横截面信号弱时 CTA 仓位缩放下限（0~1）。"""
        self.xs_position_ceiling = xs_position_ceiling
        """横截面信号强时 CTA 仓位缩放上限（0~1）。"""
        self.xs_opposite_penalty = xs_opposite_penalty
        """CTA 与横截面异号时的额外仓位惩罚系数（0~1）。"""
        # 方向三：配对交易横截面信号
        self.cross_section_source = cross_section_source
        """横截面信号来源：default（5 子策略）/ pair_trading（配对 z-score）。"""
        self.pair_params = pair_params or PairSelectorParams()
        self._pair_selector: Optional[PairSelector] = None
        self._pair_score_cache: Dict[int, Dict[str, float]] = {}
        """{bar_idx: {symbol: pair_zscore}} — 在 precompute 时填充。"""
        # 方向四 P1：CTA 合成权重（默认 None → 使用 DEFAULT_CTA_WEIGHTS）
        self.cta_composite_weights: Optional[Dict[str, float]] = None

    # ────────────────────────────────────────────────────────────
    # 模式 A：横截面信号
    # ────────────────────────────────────────────────────────────

    def get_cross_sectional_signals(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        bar_idx: int,
        strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """返回 {子策略名: 得分}，供 FactorScoringEngine 横截面排名。

        输出 5 个信号：trend, term_structure, mean_reversion,
                     vol_breakout, composite_resonance

        Args:
            symbol: 品种代码
            ohlcv: OHLCV DataFrame
            bar_idx: 当前 bar 索引
            strategy_params: 子策略参数（透传给因子聚合器）

        Returns:
            {子策略名: 得分 [-1, 1]}
        """
        signals = self._pool.compute_signals_for_bar(
            ohlcv,
            symbol,
            bar_idx,
            strategy_params,
        )
        return {
            name: float(np.clip(signals.get(name, 0.0), -1.0, 1.0))
            for name in DEFAULT_FACTOR_NAMES
        }

    # ────────────────────────────────────────────────────────────
    # 模式 B：CTA 信号
    # ────────────────────────────────────────────────────────────

    def get_cta_signals(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        bar_idx: int,
    ) -> Dict[str, float]:
        """返回 {策略名: 信号值}，共 6 个。

        Args:
            symbol: 品种代码
            ohlcv: OHLCV DataFrame
            bar_idx: 当前 bar 索引

        Returns:
            {策略名: 信号值 [-1, 1]}
        """
        signals = self._pool.compute_signals_for_bar(ohlcv, symbol, bar_idx)
        return {
            name: float(np.clip(signals.get(name, 0.0), -1.0, 1.0))
            for name in CTA_SIGNAL_NAMES
        }

    def get_cta_composite_signal(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        bar_idx: int,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """6 个 CTA 信号加权合成单值 [-1, 1]。

        Args:
            symbol: 品种代码
            ohlcv: OHLCV DataFrame
            bar_idx: 当前 bar 索引
            weights: 权重字典，默认使用 DEFAULT_CTA_WEIGHTS

        Returns:
            合成信号 [-1, 1]
        """
        signals = self.get_cta_signals(symbol, ohlcv, bar_idx)
        w = (
            self.cta_composite_weights
            if self.cta_composite_weights is not None
            else weights or DEFAULT_CTA_WEIGHTS
        )
        weighted = sum(signals.get(k, 0.0) * w.get(k, 0.0) for k in w)
        return float(np.clip(weighted, -1.0, 1.0))

    # ────────────────────────────────────────────────────────────
    # 模式 C：混合信号（横截面排名 × CTA 时序）
    # ────────────────────────────────────────────────────────────

    def get_hybrid_signal(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        bar_idx: int,
        cross_section_z: float,
        cta_weight: Optional[float] = None,
    ) -> float:
        """混合信号：横截面排名与 CTA 时序信号的互补组合（线性加权）。

        逻辑：
          - cross_section_z > 0 → 品种相对偏强，cross_section_z < 0 → 偏弱
          - cta_composite > 0   → 自身趋势向上，< 0 → 向下
          - 混合 = (1 - cta_weight) * cross_section_z
                + cta_weight     * cta_composite

        Args:
            symbol: 品种代码
            ohlcv: OHLCV DataFrame
            bar_idx: 当前 bar 索引
            cross_section_z: 来自 FactorScoringEngine 的品种横截面 z-score
            cta_weight: CTA 信号权重（0~1），0 = 纯横截面，1 = 纯 CTA

        Returns:
            混合信号 [-1, 1]
        """
        cw = self.cta_weight if cta_weight is None else cta_weight
        cta_sig = self.get_cta_composite_signal(symbol, ohlcv, bar_idx)
        blended = (1.0 - cw) * cross_section_z + cw * cta_sig
        return float(np.clip(blended, -1.0, 1.0))

    def get_hybrid_signal_dynamic(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        bar_idx: int,
        cross_section_z: float,
    ) -> float:
        """动态混合信号：横截面作为 CTA 的仓位缩放因子（方向二实现）。

        设计思想（2026-06-15）：
          - 方向完全由 CTA 信号决定（保留趋势跟随能力）
          - 横截面信号只影响仓位大小：信号强→满仓，信号弱→半仓
          - CTA 与横截面异号时（趋势末期 / 反向市）→ 额外减仓，充当过滤器
          - 不要求横截面本身盈利，只需其绝对值能反映市场分歧/状态

        公式：
          cross_strength = clip(|z|, 0, 1)          # 横截面强度（0~1）
          position_scale = base + (ceiling-base)*cross_strength
          if cta_sig * z < 0:                       # 异号
              position_scale *= opposite_penalty
          final_signal = cta_sig * position_scale

        默认参数下：
          - 横截面无信息时，CTA 仓位 = 0.5（半仓）
          - 横截面强且与 CTA 同向时，CTA 仓位 = 1.0（满仓）
          - 异号时，CTA 仓位 = 0.25（半仓的半仓）

        Args:
            symbol: 品种代码
            ohlcv: OHLCV DataFrame
            bar_idx: 当前 bar 索引
            cross_section_z: 来自 FactorScoringEngine 的品种横截面 z-score

        Returns:
            仓位缩放后的信号，符号同 cta_sig，幅度 ≤ |cta_sig|
        """
        cta_sig = self.get_cta_composite_signal(symbol, ohlcv, bar_idx)
        # 横截面强度（裁剪到 [0, 1]）
        cross_strength = float(np.clip(abs(cross_section_z), 0.0, 1.0))
        # 基础仓位缩放
        position_scale = (
            self.xs_position_base
            + (self.xs_position_ceiling - self.xs_position_base) * cross_strength
        )
        # CTA 与横截面异号 → 额外减仓
        if float(cta_sig) * float(cross_section_z) < 0.0:
            position_scale *= self.xs_opposite_penalty
        # 缩放后的信号：方向由 CTA 决定
        scaled = float(cta_sig) * float(np.clip(position_scale, 0.0, 1.0))
        return float(np.clip(scaled, -1.0, 1.0))

    # ────────────────────────────────────────────────────────────
    # 方向三：配对交易横截面信号（2026-06-17）
    # ────────────────────────────────────────────────────────────

    def precompute_pair_signals(self, close_df: pd.DataFrame) -> None:
        """预计算所有 bar 的配对横截面 z-score，存入缓存。

        Args:
            close_df: 列=品种，索引=bar 的 close 矩阵
        """
        symbols = list(close_df.columns)
        if self._pair_selector is None or self._pair_selector.symbols != symbols:
            self._pair_selector = PairSelector(symbols, self.pair_params)
        self._pair_score_cache.clear()
        n = len(close_df)
        for i in range(n):
            scores = self._pair_selector.compute_symbol_scores(close_df, i)
            self._pair_score_cache[i] = dict(scores)
        _logger.info(
            "PairSelector precomputed: %d bars × %d symbols, valid_pairs=%d",
            n,
            len(symbols),
            len(self._pair_selector._valid_pairs),
        )

    def get_pair_cross_section_scores(
        self,
        symbol: str,
        bar_idx: int,
    ) -> Optional[float]:
        """获取某品种某 bar 的配对横截面 z-score（来自预计算缓存）。

        Returns:
            pair z-score（float），或 None（未启用/未预计算）
        """
        if self._pair_selector is None:
            return None
        row = self._pair_score_cache.get(bar_idx)
        if row is None:
            return None
        return float(row.get(symbol, 0.0))

    def is_pair_trading_source(self) -> bool:
        """当前 cross_section_source 是否为配对交易。"""
        return self.cross_section_source == "pair_trading"

    # ────────────────────────────────────────────────────────────
    # 方向四 P2：四因子 CTA 融合（2026-06-19）
    # ────────────────────────────────────────────────────────────

    # 四因子默认权重（动量 / 期限结构 / 基差动量 / 仓单）
    # 等权近似：动量 0.30、期限 0.25、基差 0.25、仓单 0.20
    DEFAULT_FOUR_FACTOR_WEIGHTS: Dict[str, float] = {
        "donchian_breakout": 0.30,  # 动量
        "carry": 0.25,              # 期限结构
        "basis_momentum": 0.25,     # 基差动量
        "receipt_change": 0.20,     # 仓单变化率
    }

    def get_four_factor_signal(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        bar_idx: int,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """四因子融合信号（动量 + 期限 + 基差动量 + 仓单）。

        信号组成：
          - donchian_breakout：来自 CTA 策略（趋势动量）
          - carry：来自 CTA 策略（期限结构）
          - basis_momentum：来自 factor_pool 独立计算
          - receipt_change：来自 factor_pool 独立计算（无仓单数据时为 0）

        缺失数据回退：
          - 调用方应在调用前通过 has_basis / has_receipt 决定权重
          - 工具方法 `compute_four_factor_weights()` 自动按可用因子分配权重

        Args:
            symbol: 品种代码
            ohlcv: OHLCV DataFrame（含 date, open, high, low, close, volume）
            bar_idx: 当前 bar 索引
            weights: 自定义权重 {因子名: 权重}，默认使用 DEFAULT_FOUR_FACTOR_WEIGHTS

        Returns:
            融合信号 [-1, 1]
        """
        signals = self._pool.compute_signals_for_bar(ohlcv, symbol, bar_idx)
        donchian = float(np.clip(signals.get("donchian_breakout", 0.0), -1.0, 1.0))
        carry = float(np.clip(signals.get("carry", 0.0), -1.0, 1.0))
        basis_mom = float(np.clip(signals.get("basis_momentum", 0.0), -1.0, 1.0))
        receipt = float(np.clip(signals.get("receipt_change", 0.0), -1.0, 1.0))

        factor_values = {
            "donchian_breakout": donchian,
            "carry": carry,
            "basis_momentum": basis_mom,
            "receipt_change": receipt,
        }
        w = dict(weights or self.DEFAULT_FOUR_FACTOR_WEIGHTS)
        # 加权融合
        raw_signal = sum(
            factor_values.get(name, 0.0) * weight for name, weight in w.items()
        )
        return float(np.clip(raw_signal, -1.0, 1.0))

    @staticmethod
    def compute_four_factor_weights(
        has_basis: bool,
        has_receipt: bool,
        base_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """根据数据可用性调整四因子权重。

        Args:
            has_basis: 品种是否可计算 basis_momentum（有 far_close 数据）
            has_receipt: 品种是否有仓单数据
            base_weights: 基础权重，默认使用 DEFAULT_FOUR_FACTOR_WEIGHTS

        Returns:
            调整后的权重字典 {因子名: 权重}，总和 = 1.0
        """
        base = dict(base_weights or SignalAbstractionLayer.DEFAULT_FOUR_FACTOR_WEIGHTS)
        if has_basis and has_receipt:
            return base

        available = ["donchian_breakout", "carry"]
        if has_basis:
            available.append("basis_momentum")
        if has_receipt:
            available.append("receipt_change")

        # 重新归一化：仅使用可用因子的权重
        avail_weight_sum = sum(base.get(k, 0.0) for k in available)
        if avail_weight_sum <= 0:
            # 全部缺失 → 等权
            n = len(available)
            return {k: 1.0 / n for k in available}
        # 等比放大保持原比例
        scale = 1.0 / avail_weight_sum
        return {k: base.get(k, 0.0) * scale for k in available}

    def get_four_factor_signal_dynamic(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        bar_idx: int,
        cross_section_z: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """四因子融合 + 方向二（横截面作为仓位缩放因子）。

        设计思想（2026-06-19）：
          - 方向由四因子融合信号决定（保留多因子互补能力）
          - 横截面信号只影响仓位大小：信号强→满仓，信号弱→半仓
          - 四因子与横截面异号时 → 额外减仓（方向二核心逻辑）

        公式：
          four_factor = get_four_factor_signal(...)
          position_scale = base + (ceiling-base)*|cross_section_z|
          if four_factor * cross_section_z < 0:  # 异号
              position_scale *= opposite_penalty
          final = four_factor * position_scale

        默认参数（b=0.25, p=0.4）：
          - 横截面无信息时，四因子仓位 = 0.25（半仓的半仓）
          - 横截面强且与四因子同向时，四因子仓位 = 1.0（满仓）
          - 异号时，四因子仓位 = 0.1（25% × 40%）

        Args:
            symbol: 品种代码
            ohlcv: OHLCV DataFrame
            bar_idx: 当前 bar 索引
            cross_section_z: 来自 FactorScoringEngine 的品种横截面 z-score
            weights: 四因子权重

        Returns:
            仓位缩放后的信号，符号同 four_factor，幅度 ≤ |four_factor|
        """
        four_factor = self.get_four_factor_signal(symbol, ohlcv, bar_idx, weights)
        # 横截面强度
        cross_strength = float(np.clip(abs(cross_section_z), 0.0, 1.0))
        # 基础仓位缩放
        position_scale = (
            self.xs_position_base
            + (self.xs_position_ceiling - self.xs_position_base) * cross_strength
        )
        # 异号 → 额外减仓
        if float(four_factor) * float(cross_section_z) < 0.0:
            position_scale *= self.xs_opposite_penalty
        # 缩放后的信号
        scaled = float(four_factor) * float(np.clip(position_scale, 0.0, 1.0))
        return float(np.clip(scaled, -1.0, 1.0))

    def set_four_factor_weights(self, weights: Dict[str, float]) -> None:
        """设置四因子融合权重（运行时覆盖）。"""
        self.DEFAULT_FOUR_FACTOR_WEIGHTS = dict(weights)

    # ────────────────────────────────────────────────────────────
    # 工具
    # ────────────────────────────────────────────────────────────

    @property
    def pool(self) -> UnifiedFactorPool:
        """获取底层 UnifiedFactorPool 实例。"""
        return self._pool
