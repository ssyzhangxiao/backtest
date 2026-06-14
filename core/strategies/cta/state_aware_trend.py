"""
趋势信噪比策略（TSI-GARCH）— 纯信号生成器。

三层信号映射：
  第一层（线性滤波）：滚动 t 统计量（log 价格回归，斜率 / 标准误）
  第二层（状态变换）：t 值 tanh 压缩 → 方向信号
  第三层（复合）：sigma 用于风险平价（执行器读取）

配置参数（config）:
    reg_window: 回归窗口（默认 60）
    min_obs: 最小观测数（默认 30）
    t_stat_threshold: t 统计量入场阈值（默认 1.0）
    cache_update_freq: 模型更新频率（bar 数，默认 5）
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from core.strategies.cta.base import CTABaseStrategy
from core.strategies.cta.registry import register_cta_strategy

logger = logging.getLogger(__name__)


class TSIGarchStrategy(CTABaseStrategy):
    """趋势信噪比策略 — 纯信号 + sigma 风险平价。

    主信号：log 价格回归的滚动 t 统计量（跨品种可比）
    GARCH sigma 降级为风险调节器（执行器读取做波动率平价）

    配置参数（config）:
        reg_window: 回归窗口（默认 60）
        min_obs: 最小观测数（默认 30）
        t_stat_threshold: t 统计量入场阈值（默认 1.0）
        cache_update_freq: 模型更新频率（bar 数，默认 5）
    """

    _arch_available: bool = False

    @classmethod
    def _ensure_arch(cls) -> None:
        if not cls._arch_available:
            try:
                import arch  # noqa: F401
                cls._arch_available = True
            except ImportError:
                raise ImportError(
                    "arch 库未安装，请运行: pip install arch "
                    "(或 pip install -r requirements-models.txt)"
                )

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {
            "reg_window": 60,
            "min_obs": 30,
            "t_stat_threshold": 2.0,
            "cache_update_freq": 5,
            **(config or {}),
        }
        super().__init__(config=merged)
        self._model_cache: Dict[str, Dict[str, float]] = {}
        self._call_counter: Dict[str, int] = {}

    def compute_signal(
        self,
        symbol: str,
        close: np.ndarray,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
        volume: np.ndarray | None = None,
        ctx: Any = None,
    ) -> float:
        """计算纯信号。

        主信号：对 log(prices) 做 OLS 回归投在时间上，
        斜率 = log 收益率 / bar，跨品种可比。
        t 统计量 = 斜率 / 标准误 → 信号显著性。

        Returns:
            信号 [-1, 1]
        """
        if not self._validate(close, min_len=self.config["min_obs"]):
            return 0.0

        reg_w = self.config["reg_window"]

        if len(close) < reg_w + 10:
            return 0.0

        # ── 主信号：对 log 价格回归，斜率 = 对数收益率/bar ──
        log_prices = np.log(close[-reg_w:])
        x = np.arange(reg_w, dtype=float)
        y = log_prices

        n = reg_w
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))

        # 斜率 b
        cov_xy = float(np.sum((x - x_mean) * (y - y_mean)))
        var_x = float(np.sum((x - x_mean) ** 2))  # Sxx = sum((x-x_mean)^2)
        if var_x < 1e-10:
            return 0.0
        slope = cov_xy / var_x

        # 残差标准误
        y_pred = y_mean + slope * (x - x_mean)
        residuals = y - y_pred
        rss = float(np.sum(residuals ** 2))
        # se(slope) = sqrt(RSS / ((n-2) * Sxx))
        se = np.sqrt(rss / ((n - 2) * var_x + 1e-10))

        t_stat = slope / (se + 1e-10)

        # 信号：tanh(t_stat / threshold)
        threshold = self.config["t_stat_threshold"]
        signal = float(np.tanh(t_stat / threshold))

        # ── 市场状态判定 ──
        if abs(t_stat) > threshold:
            self.set_state(symbol, "market_state", "trend")
        else:
            self.set_state(symbol, "market_state", "oscillation")

        # ── GARCH sigma（风险调节器） ──
        counter = self._call_counter.get(symbol, 0)
        ret = np.diff(close) / close[:-1]
        if counter % self.config["cache_update_freq"] == 0 and len(ret) > 30:
            _, _, sigma = self._estimate_vol(ret * 100.0)
            self._model_cache[symbol] = {"sigma": sigma}
            self._call_counter[symbol] = 0
        else:
            cached = self._model_cache.get(symbol, {})
            sigma = cached.get("sigma", float(np.std(ret[-20:]) * 100)) if len(ret) >= 20 else 1.0
        self._call_counter[symbol] = counter + 1

        # 存储 sigma 供执行器做风险平价
        self.set_state(symbol, "sigma", sigma)

        logger.debug("%s tsi_garch: t_stat=%.2f slope=%.6f signal=%.4f sigma=%.2f",
                     symbol, t_stat, slope, signal, sigma)
        return signal

    def _estimate_vol(self, ret: np.ndarray) -> tuple[float, float, float]:
        """估计波动率（优先 GARCH，回退 rolling std）。"""
        self._ensure_arch()
        from arch import arch_model

        n = min(len(ret), 252)
        ret_window = ret[-n:]

        try:
            model = arch_model(ret_window, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="normal")
            res = model.fit(disp="off", show_warning=False)
            mu = float(res.params.get("mu", 0.0))
            rho = float(res.params.get("AR[1]", 0.0))
            sigma = float(res.conditional_volatility.iloc[-1])
        except Exception as e:
            logger.debug("GARCH 估计失败 (%s), 回退到 rolling std", e)
            mu = 0.0
            rho = 0.0
            sigma = float(np.std(ret_window[-20:])) if len(ret_window) >= 20 else 1.0
            if sigma < 1e-8:
                sigma = 1.0

        return mu, rho, sigma


class MomentumMAStrategy(CTABaseStrategy):
    """均线动量策略 — 纯信号生成器。

    简化版：RSI 偏离直接作为信号，ADX 方向限制 + 多周期动量冲突时归零。

    配置参数:
        rsi_window:   RSI 窗口（默认 14）
        momentum_fast:  快速动量窗口（默认 5）
        momentum_mid:   中速动量窗口（默认 20）
        momentum_slow:  慢速动量窗口（默认 60）
        adx_window:     ADX 窗口（默认 14）
        adx_threshold:  ADX 阈值（默认 20）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {
            "rsi_window": 14,
            "momentum_fast": 5,
            "momentum_mid": 20,
            "momentum_slow": 60,
            "adx_window": 14,
            "adx_threshold": 20,
            **(config or {}),
        }
        super().__init__(config=merged)

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

        核心逻辑：
          - RSI 偏离 (50-RSI)/30 作为原始信号
          - 多周期动量方向一致时增强，冲突时归零
          - ADX +DI/-DI 方向验证

        Returns:
            信号 [-1, 1]:
              >0 做多（RSI < 50 + 动量向上），<0 做空（RSI > 50 + 动量向下）
              0 无信号
        """
        max_len = max(
            self.config["momentum_slow"], self.config["rsi_window"],
        ) + 10
        if not self._validate(close, min_len=max_len):
            return 0.0

        # ── RSI 偏离信号 (50-RSI)/30 → [-1, 1] ──
        # RSI 是均值回复指标：RSI 高 → 做空，RSI 低 → 做多
        rsi_val = self._compute_rsi(close, self.config["rsi_window"])
        rsi_signal = (50.0 - rsi_val) / 30.0
        signal = float(np.clip(rsi_signal, -1.0, 1.0))

        # ── 多周期动量投票（仅用于择时相位，不做信号融合） ──
        ret = np.diff(close) / close[:-1]
        # 动量作为噪声过滤器：动量太弱（|mean| < 0.5 × 标准差）视为无方向 → 不做
        mom_std = float(np.std(ret[-self.config["momentum_slow"]:]))
        mom_mean = float(np.mean(ret[-self.config["momentum_slow"]:]))
        if abs(mom_mean) < 0.5 * mom_std / np.sqrt(self.config["momentum_slow"]):
            return 0.0

        # ── ADX 方向限制 ──
        if len(close) > self.config["adx_window"] * 2 + 5:
            adx_val, plus_di, minus_di = self._compute_adx_di(
                close, high, low, self.config["adx_window"]
            )
            if adx_val >= self.config["adx_threshold"]:
                if signal > 0 and plus_di < minus_di:
                    return 0.0
                if signal < 0 and plus_di > minus_di:
                    return 0.0

        self.set_state(symbol, "market_state", "trend" if abs(signal) > 0.3 else "oscillation")

        momentum_dir = int(np.sign(mom_mean))
        logger.debug("%s momentum_ma: RSI=%.1f rsi_sig=%.3f mom_dir=%d sig=%.4f",
                     symbol, rsi_val, rsi_signal, momentum_dir, signal)
        return signal

    def _compute_rsi(self, close: np.ndarray, window: int) -> float:
        if len(close) < window + 2:
            return 50.0
        ret = np.diff(close)
        gains = ret[ret > 0]
        losses = -ret[ret < 0]
        avg_gain = float(np.mean(gains[-window:])) if len(gains) > 0 else 0.0
        avg_loss = float(np.mean(losses[-window:])) if len(losses) > 0 else 1e-10
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def _compute_adx_di(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray, window: int
    ) -> tuple[float, float, float]:
        if len(close) < window * 2:
            return 20.0, 50.0, 50.0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        def wilder(arr, w):
            out = np.zeros(w)
            out[0] = float(np.mean(arr[-w:]))
            for i in range(1, w):
                out[i] = (out[i-1] * (w-1) + arr[-(w-i)]) / w
            return out

        sw = min(window, len(tr) - 1)
        tr_s = wilder(tr, sw)
        pdm_s = wilder(plus_dm, sw)
        mdm_s = wilder(minus_dm, sw)

        pdi = 100.0 * pdm_s[-1] / (tr_s[-1] + 1e-10)
        mdi = 100.0 * mdm_s[-1] / (tr_s[-1] + 1e-10)
        dx = 100.0 * abs(pdi - mdi) / (pdi + mdi + 1e-10)
        adx = float(np.mean(dx))
        return adx, pdi, mdi


# ── 注册 ──
register_cta_strategy("tsi_garch", TSIGarchStrategy)
register_cta_strategy("state_aware_trend", TSIGarchStrategy)
register_cta_strategy("momentum_ma", MomentumMAStrategy)
register_cta_strategy("simple_trend", MomentumMAStrategy)

__all__ = ["TSIGarchStrategy", "MomentumMAStrategy"]
