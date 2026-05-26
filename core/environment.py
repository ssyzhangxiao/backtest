"""
自适应市场状态引擎模块。

独立于 PyBroker 的纯 pandas 工具模块，融合多维度指标判断市场状态：
- 趋势强度比率（EMA间距/ATR）：自适应跨品种的趋势检测
- 波动率压缩（短ATR/长ATR）：识别突破前的低波动区间
- ADX：传统趋势强度指标（保留兼容）
- 动量压力（RSI偏离度）
- 衰竭检测（价格与振荡器背离）

输出连续的趋势分数（0~1）和自适应策略权重，避免离散跳变。

v4: 内部委托给 MarketRegimeDetector 消除重复计算，
     保留独立的连续评分和动态权重功能。
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from .market_regime import MarketRegimeDetector


class EnvironmentAdapter:
    """
    自适应市场状态引擎。

    多指标融合计算市场环境状态，输出连续趋势分数和动态权重。
    不依赖 PyBroker，纯 pandas 实现。

    v4: 内部使用 MarketRegimeDetector 消除 ADX/ATR/RSI 重复计算，
        保留独特的连续评分（trend_score, compression_score 等）和
        动态权重功能。

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

        # ── v4: 内部委托 MarketRegimeDetector ──
        from .market_regime import RegimeConfig

        self._detector = MarketRegimeDetector(
            RegimeConfig(
                adx_period=adx_period,
                atr_period=atr_period,
                rsi_period=rsi_period,
                confirm_days=regime_confirm_days,
                normalize_window=normalize_window,
            )
        )

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
        """委托给 MarketRegimeDetector 计算 ATR。"""
        p = period or self.atr_period
        return self._detector.compute_atr(high, low, close, period=p)

    def compute_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """委托给 MarketRegimeDetector 计算 ADX。"""
        adx, _, _ = self._detector.compute_adx(high, low, close)
        return adx

    def _compute_rsi(self, close: pd.Series, period: Optional[int] = None) -> pd.Series:
        """委托给 MarketRegimeDetector 计算 RSI。"""
        p = period or self.rsi_period
        return self._detector.compute_rsi(close, period=p)

    @staticmethod
    def _normalize(series: pd.Series, window: int) -> pd.Series:
        rolling_min = series.shift(1).rolling(window=window, min_periods=1).min()
        rolling_max = series.shift(1).rolling(window=window, min_periods=1).max()
        denom = rolling_max - rolling_min
        normalized = np.where(denom > 0, (series - rolling_min) / denom, 0.5)
        normalized = pd.Series(normalized, index=series.index)
        return normalized.fillna(0.5)

    def compute_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """趋势强度：EMA间距 / ATR（归一化）。"""
        ema_fast = (
            df["close"].ewm(span=self.trend_fast, min_periods=self.trend_fast).mean()
        )
        ema_slow = (
            df["close"].ewm(span=self.trend_slow, min_periods=self.trend_slow).mean()
        )
        atr = self.compute_atr(df["high"], df["low"], df["close"])

        trend_raw = np.where(atr > 0, (ema_fast - ema_slow).abs() / atr, 0.0)
        trend_raw = pd.Series(trend_raw, index=df.index)
        trend_raw = trend_raw.fillna(0.0)
        return self._normalize(trend_raw, self.normalize_window)

    def compute_compression(self, df: pd.DataFrame) -> pd.Series:
        """波动率压缩：ATR_short / ATR_long（反转归一化）。"""
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
        """动量分数：RSI偏离50的程度。"""
        rsi = self._compute_rsi(df["close"])
        momentum_raw = (rsi - 50).abs() / 50
        return self._normalize(momentum_raw, self.normalize_window)

    def compute_liquidity_score(self, df: pd.DataFrame) -> pd.Series:
        """流动性分数：日振幅 / ATR（归一化）。"""
        atr = self.compute_atr(df["high"], df["low"], df["close"])
        intraday_range = df["high"] - df["low"]
        liquidity_raw = np.where(atr > 0, intraday_range / atr, 0.0)
        liquidity_raw = pd.Series(liquidity_raw, index=df.index)
        liquidity_raw = liquidity_raw.clip(upper=10.0)
        return self._normalize(liquidity_raw, self.normalize_window)

    def detect_exhaustion(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """委托给 MarketRegimeDetector 检测顶底背离。"""
        rsi = self._compute_rsi(df["close"])
        return self._detector.detect_divergence(df["close"], rsi)

    def compute_dynamic_weights(self, trend_score: pd.Series) -> pd.DataFrame:
        """根据趋势分数计算动态策略权重。"""
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
        """使用 MarketRegimeDetector 计算基准指标，再叠加连续评分。"""
        result = df.copy()

        # ── v4: 使用 MarketRegimeDetector 计算原子指标 ──
        indicators = self._detector.compute_indicators(result)
        result["atr"] = indicators["atr"]
        result["adx"] = indicators["adx"]
        result["plus_di"] = indicators["plus_di"]
        result["minus_di"] = indicators["minus_di"]

        # ── v4: EnvironmentAdapter 独有的连续评分 ──
        result["trend_score"] = self.compute_trend_strength(result)
        result["compression_score"] = self.compute_compression(result)
        result["momentum_score"] = self.compute_momentum_score(result)
        result["liquidity_score"] = self.compute_liquidity_score(result)

        bearish_exh, bullish_exh = self.detect_exhaustion(result)
        result["bearish_exhaustion"] = bearish_exh
        result["bullish_exhaustion"] = bullish_exh

        # 双标签：离散regime（来自detector）+ 简化标签（向后兼容）
        raw_regime = np.where(result["adx"] > self.trend_threshold, 1, 0)
        regime_series = pd.Series(raw_regime, index=result.index)
        confirm_window = self.regime_confirm_days
        confirmed = regime_series.rolling(window=confirm_window, min_periods=1).sum()
        result["market_regime"] = np.where(
            confirmed >= confirm_window, "trend", "range"
        )

        # 附加 MarketRegimeDetector 的完整分类（可选）
        try:
            regime_df = self._detector.classify_regime(indicators)
            result["regime"] = regime_df["regime"]
            result["regime_confidence"] = regime_df["regime_confidence"]
        except Exception:
            pass

        weights_df = self.compute_dynamic_weights(result["trend_score"])
        result["weight_trend"] = weights_df["weight_trend"]
        result["weight_reversal"] = weights_df["weight_reversal"]
        result["weight_spread"] = weights_df["weight_spread"]

        return result

    def compute_environment(self, df: pd.DataFrame) -> pd.DataFrame:
        """主入口：计算完整环境指标。"""
        required_cols = ["high", "low", "close"]
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
        """
        根据市场状态获取策略权重。

        v4: 委托给 MarketRegimeDetector.get_regime_weights()。

        Args:
            regime: 市场状态，'trend' 或 'range'

        Returns:
            {策略类型: 权重} 的字典
        """
        return self._detector.get_regime_weights(regime)

    def compute_for_pybroker(self, df: pd.DataFrame) -> pd.DataFrame:
        """转为 PyBroker 可用格式（列名加 env_ 前缀）。"""
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
