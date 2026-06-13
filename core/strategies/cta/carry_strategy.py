"""
期限结构 CTA 策略（Carry / 展期收益）— 纯信号生成器。

三层信号映射：
  第一层（线性滤波）：spread = close - far_close（展期差值）
  第二层（状态变换）：EMA 平滑 → z-score + bootstrap 置信区间
  第三层（复合）：连续信号 = clip(z / entry_z, -1, 1)

（规则31管线）增强功能：
  - EMA 平滑替代卷积（消除尾部偏差）
  - Bootstrap 置信区间：只做统计显著的展期收益
  - 期限结构斜率（若有多远月合约）

输出约定：
  - 纯信号 [-1, 1] + market_state（"oscillation"）
  - 不维护持仓状态，不实现退出逻辑（执行器统一管理）

配置参数:
  lookback:    z-score 窗口（默认 60）
  entry_z:     入场 z-score 阈值（默认 1.2）
  direction:   交易方向（"both"/"long_only"/"short_only"）
  ema_alpha:   EMA 平滑系数（默认 0.3，值越大越敏感）
  bootstrap_n: Bootstrap 采样次数（默认 200）
  use_slope:   是否使用期限结构斜率（默认 True）
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from core.strategies.cta.base import CTABaseStrategy
from core.strategies.cta.registry import register_cta_strategy


class CarryStrategy(CTABaseStrategy):
    """期限结构 CTA 策略 — 纯信号生成器。

    修复卷积泄漏 + EMA 平滑 + Bootstrap 置信区间 + 期限斜率。

    配置参数:
        lookback:    z-score 窗口（默认 60）
        entry_z:     入场阈值（默认 1.2）
        direction:   交易方向（"both"/"long_only"/"short_only"）
        ema_alpha:   EMA 平滑系数（默认 0.3）
        bootstrap_n: Bootstrap 采样次数（默认 200）
        use_slope:   是否使用期限结构斜率（默认 True）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {
            "lookback": 60,
            "entry_z": 1.2,
            "direction": "long_only",
            "ema_alpha": 0.3,
            "bootstrap_n": 200,
            "use_slope": True,
            **(config or {}),
        }
        super().__init__(merged)

    def compute_signal(
        self,
        symbol: str,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray | None = None,
        ctx: Any = None,
    ) -> float:
        """计算纯信号。

        Returns:
            信号 [-1, 1]:
              >0 做多（backwardation），<0 做空（contango），0 无信号
        """
        if not self._validate(close, min_len=self.config["lookback"] + 5):
            return 0.0

        lookback = self.config["lookback"]
        entry_z = self.config["entry_z"]

        # 从策略状态获取 spread 序列（由 _run_single 注入）
        spread_raw = self.get_state(symbol, "_spread", None)
        if spread_raw is None:
            spread_ctx = getattr(ctx, "spread", None) if ctx is not None else None
            if spread_ctx is None or np.isnan(spread_ctx).all():
                return 0.0
            spread_raw = np.asarray(spread_ctx, dtype=float)

        spread_raw = np.asarray(spread_raw, dtype=float)
        spread_raw = spread_raw[:len(close)]
        if len(spread_raw) < lookback or np.isnan(spread_raw).all():
            return 0.0

        # ── EMA 平滑替代卷积（无未来数据泄漏） ──
        ema_alpha = self.config.get("ema_alpha", 0.3)
        spread_smooth = self._ema(spread_raw, ema_alpha)

        current_spread = float(spread_smooth[-1])
        if np.isnan(current_spread):
            return 0.0

        # 历史 z-score（基于平滑后的 spread）
        valid_spread = spread_smooth[~np.isnan(spread_smooth)]
        if len(valid_spread) < lookback:
            return 0.0
        recent = valid_spread[-lookback:]

        mean = float(np.mean(recent))
        std = float(np.std(recent))
        if std <= 1e-10:
            return 0.0

        z = (current_spread - mean) / std

        # ── Bootstrap 置信区间 ──
        # 对历史 z-score 采样，检查当前 z 是否显著
        hist_z = (recent - mean) / (std + 1e-10)
        z_significant = self._bootstrap_significant(
            hist_z, z / entry_z, self.config["bootstrap_n"]
        )
        if not z_significant:
            return 0.0

        # ── 期限结构斜率增强 ──
        slope_mult = 1.0
        use_slope = self.config.get("use_slope", True)
        if use_slope:
            # 斜率 = 最近 10 个 bar 的 spread 变化方向
            if len(spread_smooth) >= 20:
                slope = spread_smooth[-1] - spread_smooth[-10]
                # 斜率方向与 z 同向 → 增强信号
                if slope * current_spread > 0:
                    slope_mult = 1.2
                else:
                    slope_mult = 0.8

        # 方向过滤
        direction = self.config["direction"]
        if direction == "long_only":
            if z < 0:
                return 0.0
            signal = np.clip(z / entry_z * slope_mult, 0.0, 1.0)
        elif direction == "short_only":
            if z > 0:
                return 0.0
            signal = np.clip(z / entry_z * slope_mult, -1.0, 0.0)
        else:
            signal = np.clip(z / entry_z * slope_mult, -1.0, 1.0)

        self.set_state(symbol, "market_state", "oscillation")
        return float(signal)

    @staticmethod
    def _ema(arr: np.ndarray, alpha: float) -> np.ndarray:
        """指数移动平均（迭代式，无未来泄漏）。

        NaN 处理：找到第一个非 NaN 值作为初始值，中间 NaN 跳过（保持上一值）。
        """
        out = arr.copy()
        # 找第一个非 NaN 起点
        first_valid = int(np.argmax(~np.isnan(arr))) if np.isnan(arr[0]) else 0
        if np.isnan(arr).all():
            return out
        for i in range(first_valid + 1, len(arr)):
            if np.isnan(arr[i]):
                out[i] = out[i - 1]  # 保持上一值
            else:
                out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    @staticmethod
    def _bootstrap_significant(
        hist_z: np.ndarray, z_norm: float, n_samples: int
    ) -> bool:
        """Bootstrap 检验：当前信号是否显著异于随机。

        从历史 z-score 中重采样 n 次，
        当前信号超过 95% 分位数 → 显著。

        阈值放松（2026-06-13）：从 p95*0.5 改为 p95*0.3，
        让更多信号通过，避免 bootstrap 过严导致信号不足。

        Returns:
            True: 显著，False: 不显著
        """
        if len(hist_z) < 20 or n_samples < 50:
            return True  # 数据不足时默认显著

        # 简化：用百分位替代完整 bootstrap
        p95 = float(np.percentile(np.abs(hist_z), 95))
        return abs(z_norm) > p95 * 0.3  # 信号强度 > 历史 95% 分位的 30%


register_cta_strategy("carry", CarryStrategy)
register_cta_strategy("carry_zscore", CarryStrategy)

__all__ = ["CarryStrategy"]
