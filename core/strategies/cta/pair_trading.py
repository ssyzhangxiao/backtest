"""
配对交易 CTA 策略 — 纯信号生成器。

核心逻辑：
  1. 动态对冲比 β：滚动窗口 OLS 回归（90天）
  2. 协整检验：每月 ADF 检验，p>0.05 暂停开仓
  3. spread 的 z-score → 连续信号 [-1, 1]

注意：此策略在 CTA 单品种框架下运行，symbol 用 "SYM_A/SYM_B" 格式，
price_A 和 price_B 分别通过 ctx.close 和 ctx.far_close 传入。

配置参数:
  lookback:     z-score 窗口（默认 60）
  entry_z:      开仓 z-score 阈值（默认 2.0）
  ols_window:   滚动 OLS 窗口（默认 90）
  adf_interval: ADF 检验间隔（默认 20，约每月）
  adf_pvalue:   ADF 显著性阈值（默认 0.05）
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from core.strategies.cta.base import CTABaseStrategy
from core.strategies.cta.registry import register_cta_strategy

# statsmodels 是必需依赖（不能无 statsmodels 运行）
try:
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    raise ImportError(
        "pair_trading 策略需要 statsmodels 库，请运行: pip install statsmodels"
    )


class PairTradingStrategy(CTABaseStrategy):
    """配对交易 CTA 策略 — 纯信号生成器。

    动态 β(rolling OLS) + 协整检验(ADF) + 连续信号。

    配置参数:
        lookback:     z-score 窗口（默认 60）
        entry_z:      开仓 z-score 阈值（默认 2.0）
        ols_window:   滚动 OLS 窗口（默认 90）
        adf_interval: ADF 检验间隔（默认 20，约每月）
        adf_pvalue:   ADF 显著性阈值（默认 0.05）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {
            "lookback": 60,
            "entry_z": 2.0,
            "ols_window": 90,
            "adf_interval": 20,
            "adf_pvalue": 0.05,
            **(config or {}),
        }
        super().__init__(merged)
        self._adf_counter: Dict[str, int] = {}
        self._adf_passed: Dict[str, bool] = {}

    def compute_signal(
        self,
        symbol: str,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray | None = None,
        ctx: Any = None,
    ) -> float:
        """计算纯信号（动态 β + 协整检验 + 连续信号）。

        Returns:
            信号 [-1, 1]:
              >0 做多 spread（z < -entry_z），<0 做空 spread（z > entry_z）
              0 无信号或协整不通过
        """
        ols_w = self.config["ols_window"]
        min_len = max(self.config["lookback"] + 10, ols_w + 10)
        if not self._validate(close, min_len=min_len):
            return 0.0

        # 获取两条价格序列
        # price_B 从 far_close 或 spread 反推
        spread_raw = self.get_state(symbol, "_spread", None)
        far_close = self.get_state(symbol, "_far_price", None)

        if far_close is None:
            # 回退：从 ctx.far_close 读取
            far_close_arr = getattr(ctx, "far_close", None)
            if far_close_arr is None:
                return 0.0
            far_close_arr = np.asarray(far_close_arr, dtype=float)
            far_close_arr = far_close_arr[:len(close)]
            self.set_state(symbol, "_far_price", far_close_arr)
            far_close = far_close_arr

        far_close = np.asarray(far_close, dtype=float)[:len(close)]

        if len(far_close) < ols_w + 5:
            return 0.0

        # ── 协整检验（按月频率） ──
        counter = self._adf_counter.get(symbol, 0)
        if counter % self.config["adf_interval"] == 0:
            passed = self._check_cointegration(close, far_close, ols_w)
            self._adf_passed[symbol] = passed
            self._adf_counter[symbol] = 0
        self._adf_counter[symbol] = counter + 1

        if not self._adf_passed.get(symbol, True):
            # 协整不通过 → 无信号
            return 0.0

        # ── 动态 β：滚动 OLS ──
        beta = self._estimate_beta(close, far_close, ols_w)
        if beta is None or np.isnan(beta):
            return 0.0

        # spread = price_A - β * price_B
        price_A = close
        price_B = far_close
        spread = price_A[-ols_w:] - beta * price_B[-ols_w:]

        # z-score
        recent = spread
        mean = float(np.nanmean(recent))
        std = float(np.nanstd(recent))
        if std <= 1e-10:
            return 0.0

        current_spread = float(spread[-1])
        if np.isnan(current_spread):
            return 0.0
        z = (current_spread - mean) / std

        # 连续信号（无死区）
        signal = np.clip(-z / self.config["entry_z"], -1.0, 1.0)

        self.set_state(symbol, "market_state", "oscillation")
        return float(signal)

    def _estimate_beta(
        self, price_A: np.ndarray, price_B: np.ndarray, window: int
    ) -> float | None:
        """滚动 OLS 估计 β = Cov(A, B) / Var(B)。"""
        n = min(window, len(price_A) - 2, len(price_B) - 2)
        if n < 30:
            return None

        a = price_A[-n:]
        b = price_B[-n:]
        cov = np.cov(a, b)
        var_b = cov[1, 1]
        if var_b < 1e-10:
            return None
        return float(cov[0, 1] / var_b)

    def _check_cointegration(
        self, price_A: np.ndarray, price_B: np.ndarray, window: int
    ) -> bool:
        """ADF 检验价差平稳性。

        Returns:
            True: 价差平稳（协整），False: 不平稳
        """
        n = min(window, len(price_A) - 2, len(price_B) - 2)
        if n < 30:
            return True  # 数据不足时默认允许

        beta = self._estimate_beta(price_A, price_B, n)
        if beta is None:
            return True

        spread = price_A[-n:] - beta * price_B[-n:]

        try:
            result = adfuller(spread, maxlag=min(10, n // 4), autolag="AIC")
            p_value = float(result[1])
            return p_value < self.config["adf_pvalue"]
        except Exception:
            return True


register_cta_strategy("pair_trading", PairTradingStrategy)

__all__ = ["PairTradingStrategy"]
