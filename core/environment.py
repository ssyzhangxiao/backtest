"""
自适应市场状态引擎模块。

独立于 PyBroker 的纯 pandas 工具模块，融合多维度指标判断市场状态：
- 趋势强度比率（EMA间距/ATR）：自适应跨品种的趋势检测
- 波动率压缩（短ATR/长ATR）：识别突破前的低波动区间
- ADX：传统趋势强度指标（保留兼容）
- 动量压力（RSI偏离度）
- 衰竭检测（价格与振荡器背离）

输出连续的趋势分数（0~1）和自适应策略权重，避免离散跳变。
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


class EnvironmentAdapter:
    """
    自适应市场状态引擎。

    多指标融合计算市场环境状态，输出连续趋势分数和动态权重。
    不依赖 PyBroker，纯 pandas 实现。

    Note: When used in a backtest with many rows, consider caching the result.

    Attributes:
        adx_period: ADX 计算周期
        atr_period: ATR 计算周期
        trend_threshold: ADX 趋势判定阈值
        trend_fast: 趋势强度快 EMA 周期
        trend_slow: 趋势强度慢 EMA 周期
        compression_short: 波动率压缩短周期
        compression_long: 波动率压缩长周期
        rsi_period: RSI 计算周期
        exhaustion_lookback: 衰竭检测回看周期
        regime_confirm_days: 市场状态确认天数
        normalize_window: 归一化滚动窗口
        weight_config: 权重配置
    """

    _DEFAULT_WEIGHT_CONFIG = {
        "trend_base": 0.2,
        "trend_range": 0.5,
        "reversal_base": 0.2,
        "reversal_range": 0.2,
        "spread_base": 0.2,
        "momentum_base": 0.2,
        "momentum_range": 0.2,
        "liquidity_base": 0.3,
        "liquidity_range": 0.1,
        "compression_base": 0.3,
        "compression_range": 0.1,
    }

    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        trend_threshold: float = 30.0,
        trend_fast: int = 10,
        trend_slow: int = 30,
        compression_short: int = 5,
        compression_long: int = 20,
        rsi_period: int = 14,
        exhaustion_lookback: int = 20,
        regime_confirm_days: int = 3,
        normalize_window: int = 100,
        weight_config: Optional[Dict[str, float]] = None,
    ):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.trend_threshold = trend_threshold
        self.trend_fast = trend_fast
        self.trend_slow = trend_slow
        self.compression_short = compression_short
        self.compression_long = compression_long
        self.rsi_period = rsi_period
        self.exhaustion_lookback = exhaustion_lookback
        self.regime_confirm_days = regime_confirm_days
        self.normalize_window = normalize_window
        self.weight_config = weight_config or type(self)._DEFAULT_WEIGHT_CONFIG.copy()

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        if len(tr) > 0:
            tr.iloc[0] = tr1.iloc[0]
        return tr

    def compute_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: Optional[int] = None,
    ) -> pd.Series:
        tr = self._true_range(high, low, close)
        p = period or self.atr_period
        atr = tr.rolling(window=p, min_periods=p).mean()
        return atr

    def _compute_adx_components(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        period = self.adx_period

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = self._true_range(high, low, close)
        atr = tr.rolling(window=period, min_periods=period).mean()

        atr_safe = atr.replace(0, np.nan)
        plus_di = 100 * (
            plus_dm.rolling(window=period, min_periods=period).mean() / atr_safe
        )
        minus_di = 100 * (
            minus_dm.rolling(window=period, min_periods=period).mean() / atr_safe
        )

        dx_denom = (plus_di + minus_di).abs()
        dx = np.where(dx_denom > 0, 100 * (plus_di - minus_di).abs() / dx_denom, 0.0)
        dx = pd.Series(dx, index=high.index)

        adx = dx.rolling(window=period, min_periods=period).mean()
        return adx, plus_di, minus_di

    def compute_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        adx, _, _ = self._compute_adx_components(high, low, close)
        return adx

    def _compute_rsi(self, close: pd.Series, period: Optional[int] = None) -> pd.Series:
        p = period or self.rsi_period
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=p, min_periods=p).mean()
        avg_loss = loss.rolling(window=p, min_periods=p).mean()

        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        rsi = 100 - 100 / (1 + rs)
        return pd.Series(rsi, index=close.index)

    @staticmethod
    def _normalize(series: pd.Series, window: int) -> pd.Series:
        rolling_min = series.shift(1).rolling(window=window, min_periods=1).min()
        rolling_max = series.shift(1).rolling(window=window, min_periods=1).max()
        denom = rolling_max - rolling_min
        normalized = np.where(denom > 0, (series - rolling_min) / denom, 0.5)
        normalized = pd.Series(normalized, index=series.index)
        return normalized.fillna(0.5)

    def compute_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        ema_fast = df["close"].ewm(span=self.trend_fast, min_periods=self.trend_fast).mean()
        ema_slow = df["close"].ewm(span=self.trend_slow, min_periods=self.trend_slow).mean()
        atr = self.compute_atr(df["high"], df["low"], df["close"])

        trend_raw = np.where(atr > 0, (ema_fast - ema_slow).abs() / atr, 0.0)
        trend_raw = pd.Series(trend_raw, index=df.index)
        trend_raw = trend_raw.fillna(0.0)
        return self._normalize(trend_raw, self.normalize_window)

    def compute_compression(self, df: pd.DataFrame) -> pd.Series:
        atr_short = self.compute_atr(
            df["high"], df["low"], df["close"], period=self.compression_short
        )
        atr_long = self.compute_atr(
            df["high"], df["low"], df["close"], period=self.compression_long
        )

        ratio = np.where(atr_short > 0, atr_long / atr_short, 1.0)
        ratio = pd.Series(ratio, index=df.index)
        compression_raw = 1 / (1 + ratio)
        return self._normalize(compression_raw, self.normalize_window)

    def compute_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        rsi = self._compute_rsi(df["close"])
        momentum_raw = (rsi - 50).abs() / 50
        return self._normalize(momentum_raw, self.normalize_window)

    def compute_liquidity_score(self, df: pd.DataFrame) -> pd.Series:
        atr = self.compute_atr(df["high"], df["low"], df["close"])
        intraday_range = df["high"] - df["low"]
        liquidity_raw = np.where(atr > 0, intraday_range / atr, 0.0)
        liquidity_raw = pd.Series(liquidity_raw, index=df.index)
        liquidity_raw = liquidity_raw.clip(upper=10.0)
        return self._normalize(liquidity_raw, self.normalize_window)

    def detect_exhaustion(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        lookback = self.exhaustion_lookback
        rsi = self._compute_rsi(df["close"])
        rolling_max_close = df["close"].rolling(window=lookback, min_periods=lookback).max()
        rolling_max_rsi = rsi.rolling(window=lookback, min_periods=lookback).max()
        rolling_min_close = df["close"].rolling(window=lookback, min_periods=lookback).min()
        rolling_min_rsi = rsi.rolling(window=lookback, min_periods=lookback).min()
        bearish = (df["close"] == rolling_max_close) & (rsi != rolling_max_rsi) & rolling_max_close.notna()
        bullish = (df["close"] == rolling_min_close) & (rsi != rolling_min_rsi) & rolling_min_close.notna()
        return bearish.fillna(False), bullish.fillna(False)

    def compute_dynamic_weights(self, trend_score: pd.Series) -> pd.DataFrame:
        cfg = self.weight_config

        w_trend = cfg["trend_base"] + cfg["trend_range"] * trend_score
        w_reversal = cfg["reversal_base"] + cfg["reversal_range"] * (1 - trend_score)
        w_spread = pd.Series(cfg["spread_base"], index=trend_score.index)

        total = w_trend + w_reversal + w_spread
        total = total.clip(lower=1e-8)
        w_trend = w_trend / total
        w_reversal = w_reversal / total
        w_spread = w_spread / total

        return pd.DataFrame(
            {
                "weight_trend": w_trend,
                "weight_reversal": w_reversal,
                "weight_spread": w_spread,
            }
        )

    def _compute_single_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        result["atr"] = self.compute_atr(result["high"], result["low"], result["close"])

        adx, plus_di, minus_di = self._compute_adx_components(
            result["high"], result["low"], result["close"]
        )
        result["adx"] = adx
        result["plus_di"] = plus_di
        result["minus_di"] = minus_di

        result["trend_score"] = self.compute_trend_strength(result)
        result["compression_score"] = self.compute_compression(result)
        result["momentum_score"] = self.compute_momentum_score(result)
        result["liquidity_score"] = self.compute_liquidity_score(result)

        bearish_exh, bullish_exh = self.detect_exhaustion(result)
        result["bearish_exhaustion"] = bearish_exh
        result["bullish_exhaustion"] = bullish_exh

        raw_regime = np.where(result["adx"] > self.trend_threshold, 1, 0)
        regime_series = pd.Series(raw_regime, index=result.index)
        confirm_window = self.regime_confirm_days
        confirmed = regime_series.rolling(window=confirm_window, min_periods=1).sum()
        result["market_regime"] = np.where(
            confirmed >= confirm_window, "trend", "range"
        )

        weights_df = self.compute_dynamic_weights(result["trend_score"])
        result["weight_trend"] = weights_df["weight_trend"]
        result["weight_reversal"] = weights_df["weight_reversal"]
        result["weight_spread"] = weights_df["weight_spread"]

        return result

    def compute_environment(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"缺少必要列: {required_cols}")
        if df.empty:
            return df

        if "symbol" in df.columns and df["symbol"].nunique() > 1:
            results = []
            for sym, group in df.groupby("symbol"):
                group = group.sort_values("date")
                results.append(self._compute_single_symbol(group))
            return pd.concat(results, ignore_index=True)
        else:
            return self._compute_single_symbol(df)

    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        cfg = self.weight_config
        if regime == "trend":
            w_t = cfg["trend_base"] + cfg["trend_range"]
            w_r = cfg["reversal_base"]
        elif regime == "range":
            w_t = cfg["trend_base"]
            w_r = cfg["reversal_base"] + cfg["reversal_range"]
        else:
            w_t = cfg["trend_base"] + cfg["trend_range"] * 0.5
            w_r = cfg["reversal_base"] + cfg["reversal_range"] * 0.5
        w_s = cfg["spread_base"]
        total = w_t + w_r + w_s
        return {"trend": w_t / total, "reversal": w_r / total, "spread": w_s / total}

    def compute_for_pybroker(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.compute_environment(df)
        rename_map = {
            "atr": "env_atr",
            "adx": "env_adx",
            "plus_di": "env_plus_di",
            "minus_di": "env_minus_di",
            "trend_score": "env_trend_score",
            "compression_score": "env_compression_score",
            "momentum_score": "env_momentum_score",
            "liquidity_score": "env_liquidity_score",
            "bearish_exhaustion": "env_bearish_exhaustion",
            "bullish_exhaustion": "env_bullish_exhaustion",
            "market_regime": "env_market_regime",
            "weight_trend": "env_weight_trend",
            "weight_reversal": "env_weight_reversal",
            "weight_spread": "env_weight_spread",
        }
        result = result.rename(columns=rename_map)
        result = result.loc[:, ~result.columns.duplicated()]
        return result
