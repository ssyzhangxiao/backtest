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
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.execution.factor_pool import (
    ALL_SIGNAL_NAMES,
    CTA_SIGNAL_NAMES,
    DEFAULT_FACTOR_NAMES,
    UnifiedFactorPool,
)

__all__ = [
    "SignalMode",
    "SignalAbstractionLayer",
    "DEFAULT_CTA_WEIGHTS",
]

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
        w = weights or DEFAULT_CTA_WEIGHTS
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
    # 工具
    # ────────────────────────────────────────────────────────────

    @property
    def pool(self) -> UnifiedFactorPool:
        """获取底层 UnifiedFactorPool 实例。"""
        return self._pool
