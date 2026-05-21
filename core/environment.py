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

    DEFAULT_WEIGHT_CONFIG = {
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
        trend_threshold: float = 25.0,
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
        self.weight_config = weight_config or self.DEFAULT_WEIGHT_CONFIG.copy()

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
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
        atr = tr.rolling(window=p, min_periods=1).mean()
        return atr

    def _compute_adx_components(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算 ADX 及 ±DI，复用中间结果避免重复计算。

        Returns:
            (adx, plus_di, minus_di) 元组
        """
        period = self.adx_period

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = self._true_range(high, low, close)
        atr = tr.rolling(window=period, min_periods=1).mean()

        atr_safe = atr.replace(0, np.nan)
        plus_di = 100 * (
            plus_dm.rolling(window=period, min_periods=1).mean() / atr_safe
        )
        minus_di = 100 * (
            minus_dm.rolling(window=period, min_periods=1).mean() / atr_safe
        )

        dx_denom = (plus_di + minus_di).abs()
        dx = np.where(dx_denom > 0, 100 * (plus_di - minus_di).abs() / dx_denom, 0.0)
        dx = pd.Series(dx, index=high.index)

        adx = dx.rolling(window=period, min_periods=1).mean()
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

        avg_gain = gain.rolling(window=p, min_periods=1).mean()
        avg_loss = loss.rolling(window=p, min_periods=1).mean()

        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        rsi = 100 - 100 / (1 + rs)
        return pd.Series(rsi, index=close.index)

    @staticmethod
    def _normalize(series: pd.Series, window: int) -> pd.Series:
        rolling_min = series.rolling(window=window, min_periods=1).min()
        rolling_max = series.rolling(window=window, min_periods=1).max()
        denom = rolling_max - rolling_min
        normalized = np.where(denom > 0, (series - rolling_min) / denom, 0.5)
        return pd.Series(normalized, index=series.index)

    def compute_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        趋势强度比率：快慢 EMA 差 / ATR。

        自适应跨品种，不依赖固定 ADX 阈值。
        与滚动中位数比较判断是否为趋势市。
        """
        ema_fast = df["close"].ewm(span=self.trend_fast, min_periods=1).mean()
        ema_slow = df["close"].ewm(span=self.trend_slow, min_periods=1).mean()
        atr = self.compute_atr(df["high"], df["low"], df["close"])

        trend_raw = np.where(atr > 0, (ema_fast - ema_slow).abs() / atr, 0.0)
        trend_raw = pd.Series(trend_raw, index=df.index)
        return self._normalize(trend_raw, self.normalize_window)

    def compute_compression(self, df: pd.DataFrame) -> pd.Series:
        """
        波动率压缩：长周期 ATR / 短周期 ATR。

        比值越大表示波动率越压缩（即将突破），
        归一化后 compression_score 越大表示越压缩。
        """
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
        """
        动量压力：RSI 偏离 50 的程度，归一化到 [0, 1]。
        """
        rsi = self._compute_rsi(df["close"])
        momentum_raw = (rsi - 50).abs() / 50
        return self._normalize(momentum_raw, self.normalize_window)

    def compute_liquidity_score(self, df: pd.DataFrame) -> pd.Series:
        """
        流动性/扫荡反转指标：日内振幅 / ATR。

        大振幅表示扫荡行为（流动性真空后反转）。
        """
        atr = self.compute_atr(df["high"], df["low"], df["close"])
        intraday_range = df["high"] - df["low"]
        liquidity_raw = np.where(atr > 0, intraday_range / atr, 0.0)
        liquidity_raw = pd.Series(liquidity_raw, index=df.index)
        return self._normalize(liquidity_raw, self.normalize_window)

    def detect_exhaustion(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        衰竭检测：价格创新高但 RSI 未创新高（看跌背离），
        价格创新低但 RSI 未创新低（看涨背离）。

        Returns:
            (bearish_exhaustion, bullish_exhaustion) 元组
        """
        lookback = self.exhaustion_lookback
        rsi = self._compute_rsi(df["close"])

        rolling_max_close = df["close"].rolling(window=lookback, min_periods=1).max()
        rolling_min_close = df["close"].rolling(window=lookback, min_periods=1).min()
        rolling_max_rsi = rsi.rolling(window=lookback, min_periods=1).max()
        rolling_min_rsi = rsi.rolling(window=lookback, min_periods=1).min()

        prev_max_close = rolling_max_close.shift(lookback)
        prev_min_close = rolling_min_close.shift(lookback)
        prev_max_rsi = rolling_max_rsi.shift(lookback)
        prev_min_rsi = rolling_min_rsi.shift(lookback)

        bearish = (
            (df["close"] >= prev_max_close)
            & (rsi < prev_max_rsi)
            & prev_max_close.notna()
            & prev_max_rsi.notna()
        )

        bullish = (
            (df["close"] <= prev_min_close)
            & (rsi > prev_min_rsi)
            & prev_min_close.notna()
            & prev_min_rsi.notna()
        )

        return bearish.fillna(False), bullish.fillna(False)

    def compute_dynamic_weights(self, trend_score: pd.Series) -> pd.DataFrame:
        """
        根据连续趋势分数计算动态策略权重。

        趋势市时 trend 权重高，震荡市时 reversal 权重高，
        权重随趋势分数连续变化，避免离散跳变。

        Args:
            trend_score: 趋势分数序列 (0~1)

        Returns:
            DataFrame 包含 weight_trend, weight_reversal, weight_spread 列
        """
        cfg = self.weight_config

        w_trend = cfg["trend_base"] + cfg["trend_range"] * trend_score
        w_reversal = cfg["reversal_base"] + cfg["reversal_range"] * (1 - trend_score)
        w_spread = pd.Series(cfg["spread_base"], index=trend_score.index)

        total = w_trend + w_reversal + w_spread
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

    def compute_environment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算完整的市场环境指标并附加到 DataFrame。

        新增列：
        - atr: 平均真实波幅
        - adx: 平均趋向指数
        - plus_di: +DI
        - minus_di: -DI
        - trend_score: 趋势分数 (0~1，连续)
        - compression_score: 波动率压缩分数 (0~1)
        - momentum_score: 动量压力分数 (0~1)
        - liquidity_score: 流动性分数 (0~1)
        - bearish_exhaustion: 看跌衰竭标志
        - bullish_exhaustion: 看涨衰竭标志
        - market_regime: 市场状态 ('trend' / 'range'，带确认周期)
        - weight_trend: 趋势策略动态权重
        - weight_reversal: 反转策略动态权重
        - weight_spread: 套利策略动态权重
        """
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

    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """
        根据市场状态返回策略权重建议（兼容旧接口）。

        新代码建议使用 compute_dynamic_weights 获取连续权重。

        Args:
            regime: 市场状态，'trend' 或 'range'

        Returns:
            策略权重字典 {'trend': float, 'reversal': float, 'spread': float}
        """
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
        """
        计算环境指标，输出格式兼容 PyBroker 注册自定义列。

        列名加 env_ 前缀避免与 PyBroker 内置列冲突。

        Args:
            df: 包含 high, low, close 列的 DataFrame

        Returns:
            带有 env_ 前缀环境指标列的 DataFrame
        """
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
        return result
