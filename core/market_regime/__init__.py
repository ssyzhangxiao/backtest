"""
市场环境分类系统（v3 - 消除前视偏差 + 修复逻辑 + 动态权重/阈值）。

基于多维度量化指标对市场环境进行分类，支持8种典型市场状态，
输出连续的环境分数和离散的环境标签。

v3 核心改进:
  1. 消除前视偏差：
     - 删除全局 future_return 计算，改为滚动IC权重（仅用历史数据）
     - validate 使用 fit/transform 模式，样本外不重新计算参数
  2. 修复逻辑错误：
     - 背离检测：要求价格创新高/低且RSI未创新高/低
     - 确认窗口：状态机实现，无后视检查
     - 波动率压缩：直接使用 atr_short / atr_long
  3. 动态权重：基于滚动IC计算，背离指标纳入IC体系
  4. 动态阈值：滚动百分位数，裁剪到合理范围
  5. 连续分数：每个环境输出0-1连续分数
  6. 样本外验证：fit/transform分离，KL散度+IC衰减+Sharpe差异

环境类型:
  - TREND_UP: 趋势上涨
  - TREND_DOWN: 趋势下跌
  - RANGE_BOUND: 区间震荡
  - HIGH_VOLATILITY: 高波动
  - LOW_VOLATILITY: 低波动
  - BREAKOUT: 突破
  - EXHAUSTION_BULL: 牛市衰竭
  - EXHAUSTION_BEAR: 熊市衰竭

量化指标（12个）:
  1. ADX - 趋势强度
  2. 趋势方向 (EMA间距符号)
  3. 波动率水平 (ATR/Close)
  4. 波动率压缩 (短ATR/长ATR)
  5. 动量 (RSI偏离度)
  6. 成交量相对强度
  7. 持仓量变化率
  8. 布林带宽度
  9. 价格位置 (相对布林带)
  10. 趋势一致性 (DI差值)
  11. 加速度 (二阶动量)
  12. 背离检测 (价格vs RSI)
"""

import warnings
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """市场环境类型枚举。"""

    UNKNOWN = "unknown"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    EXHAUSTION_BULL = "exhaustion_bull"
    EXHAUSTION_BEAR = "exhaustion_bear"


class TrendType(Enum):
    """市场趋势类型。"""

    STRONG_UP = "strong_up"  # 强上涨趋势
    WEAK_UP = "weak_up"  # 弱上涨趋势
    SIDEWAYS = "sideways"  # 震荡/横盘
    WEAK_DOWN = "weak_down"  # 弱下跌趋势
    STRONG_DOWN = "strong_down"  # 强下跌趋势


@dataclass
class TrendConfig:
    """趋势判断配置。"""

    # 均线参数
    ma_short: int = 5
    ma_medium: int = 20
    ma_long: int = 60

    # MACD参数
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # RSI参数
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # ADX参数
    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    adx_strong_threshold: float = 40.0

    # 波动率参数
    atr_period: int = 14

    # 状态转换
    state_transition_threshold: float = 0.6  # 状态转换置信度阈值
    confirm_days: int = 3  # 确认窗口天数
    confidence_window: int = 10  # 置信度计算窗口


@dataclass
class TrendResult:
    """趋势判断结果。"""

    trend_type: TrendType
    trend_name: str
    confidence: float
    indicators: Dict[str, float]
    trend_scores: Dict[str, float]  # 各趋势类型的分数


@dataclass
class RegimeConfig:
    """市场环境分类配置。"""

    # ADX参数
    adx_period: int = 14
    adx_trend_percentile: float = 60.0
    adx_strong_percentile: float = 85.0

    # ATR/波动率参数
    atr_period: int = 14
    vol_short_period: int = 5
    vol_long_period: int = 20
    high_vol_percentile: float = 75.0
    low_vol_percentile: float = 25.0

    # EMA参数
    ema_fast: int = 10
    ema_slow: int = 30

    # RSI参数
    rsi_period: int = 14
    rsi_overbought_percentile: float = 85.0
    rsi_oversold_percentile: float = 15.0

    # 布林带参数
    bb_period: int = 20
    bb_std: float = 2.0

    # 背离检测
    divergence_lookback: int = 20

    # 确认窗口（防抖动）
    confirm_days: int = 3

    # 归一化窗口
    normalize_window: int = 100

    # 动态阈值计算窗口
    threshold_window: int = 100

    # IC权重计算窗口
    ic_window: int = 60

    # IC权重重算频率（天），0=每天重算，1=每天，5=每周，20=每月
    ic_recalc_freq: int = 20

    # 样本外验证
    validation_split: float = 0.3

    # 滚动窗口环境分类（v2）
    rolling_window: int = 252
    rolling_update_freq: int = 20

    # 环境稳定性指标（v2）
    stability_window: int = 20
    stability_threshold: float = 0.6


@dataclass
class RegimeResult:
    """市场环境识别结果。"""

    regime: MarketRegime
    regime_name: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    indicators: Dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """样本外验证结果。"""

    in_sample_regime_dist: Dict[str, float]
    out_sample_regime_dist: Dict[str, float]
    distribution_stability: float  # KL散度，越小越稳定
    ic_in_sample: Dict[str, float]
    ic_out_sample: Dict[str, float]
    ic_decay: float  # IC衰减率，越小越好
    regime_sharpe_diff: Dict[str, float]  # 各环境下策略Sharpe差异


# 用于IC计算的指标列表（含背离强度）
IC_INDICATORS = [
    "adx",
    "vol_level_norm",
    "compression",
    "rsi",
    "bb_position",
    "trend_consistency",
    "acceleration",
    "volume_strength",
    "divergence_strength",
]

# 等权默认值
EQUAL_WEIGHT = 1.0 / len(IC_INDICATORS)


class MarketRegimeDetector:
    """
    市场环境识别引擎（v3 - 无前视偏差）。

    核心设计:
    - fit(df): 在数据上计算滚动参数（IC权重、动态阈值），存储为内部状态
    - transform(df): 使用已拟合的参数对新数据分类（不使用未来信息）
    - fit_transform(df): 一步完成（回测场景）
    - detect(df): 兼容旧接口，内部调用fit_transform
    - validate(df): 样本外验证，用样本内参数对样本外分类
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        # 拟合状态
        self._fitted: bool = False
        # 滚动IC权重矩阵：DataFrame，每行一个时间点，列为各指标权重
        self._ic_weights_matrix: Optional[pd.DataFrame] = None
        # 动态阈值：{阈值名: Series}
        self._dynamic_thresholds: Dict[str, pd.Series] = None
        # 拟合时使用的阈值终值（用于transform）
        self._fitted_threshold_values: Dict[str, float] = {}
        # 拟合时使用的IC权重终值（用于transform）
        self._fitted_ic_weights: Dict[str, float] = {}

    # ----------------------------------------------------------------
    # 指标计算（12个量化指标）
    # ----------------------------------------------------------------

    def _true_range(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """计算真实波幅。"""
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
        """计算平均真实波幅。"""
        p = period or self.config.atr_period
        tr = self._true_range(high, low, close)
        return tr.rolling(window=p, min_periods=p).mean()

    def compute_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算ADX、+DI、-DI。"""
        period = self.config.adx_period
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

    def compute_rsi(self, close: pd.Series, period: Optional[int] = None) -> pd.Series:
        """计算RSI。"""
        p = period or self.config.rsi_period
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=p, min_periods=p).mean()
        avg_loss = loss.rolling(window=p, min_periods=p).mean()
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        return pd.Series(100 - 100 / (1 + rs), index=close.index)

    def compute_bollinger_width(self, close: pd.Series) -> pd.Series:
        """布林带宽度（归一化）。"""
        p = self.config.bb_period
        s = self.config.bb_std
        ma = close.rolling(window=p, min_periods=p).mean()
        std = close.rolling(window=p, min_periods=p).std()
        width = (s * 2 * std) / ma.replace(0, np.nan)
        return width

    def compute_bollinger_position(self, close: pd.Series) -> pd.Series:
        """价格在布林带中的位置（0=下轨，1=上轨）。"""
        p = self.config.bb_period
        s = self.config.bb_std
        ma = close.rolling(window=p, min_periods=p).mean()
        std = close.rolling(window=p, min_periods=p).std()
        upper = ma + s * std
        lower = ma - s * std
        denom = upper - lower
        return pd.Series(
            np.where(denom > 0, (close - lower) / denom, 0.5),
            index=close.index,
        )

    def compute_volume_strength(self, volume: pd.Series) -> pd.Series:
        """成交量相对强度。"""
        p = 20
        avg_vol = volume.rolling(window=p, min_periods=p).mean()
        avg_vol_safe = avg_vol.replace(0, np.nan)
        return volume / avg_vol_safe

    def compute_oi_change(self, open_interest: pd.Series) -> pd.Series:
        """持仓量变化率。"""
        return open_interest.pct_change()

    def compute_trend_direction(self, close: pd.Series) -> pd.Series:
        """趋势方向（EMA间距符号），+1=上涨，-1=下跌。"""
        ema_fast = close.ewm(
            span=self.config.ema_fast, min_periods=self.config.ema_fast
        ).mean()
        ema_slow = close.ewm(
            span=self.config.ema_slow, min_periods=self.config.ema_slow
        ).mean()
        return pd.Series(np.sign(ema_fast - ema_slow), index=close.index)

    def compute_trend_consistency(
        self, plus_di: pd.Series, minus_di: pd.Series
    ) -> pd.Series:
        """趋势一致性（DI差值绝对值/DI和）。"""
        denom = (plus_di + minus_di).abs()
        return pd.Series(
            np.where(denom > 0, (plus_di - minus_di).abs() / denom, 0.0),
            index=plus_di.index,
        )

    def compute_acceleration(self, close: pd.Series) -> pd.Series:
        """价格加速度（二阶动量）。"""
        momentum = close.pct_change()
        return momentum.diff()

    def detect_divergence(
        self, close: pd.Series, rsi: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        检测顶背离和底背离。

        顶背离：价格创滚动窗口新高（且非与前一天相同）但RSI未创新高。
        底背离：价格创滚动窗口新低（且非与前一天相同）但RSI未创新低。
        """
        lb = self.config.divergence_lookback
        rolling_max_close = close.rolling(window=lb, min_periods=lb).max()
        rolling_max_rsi = rsi.rolling(window=lb, min_periods=lb).max()
        rolling_min_close = close.rolling(window=lb, min_periods=lb).min()
        rolling_min_rsi = rsi.rolling(window=lb, min_periods=lb).min()

        # 顶背离：价格创新高 + RSI未创新高
        # 移除 close != close.shift(1) — 价格新高后次日持平且RSI下降仍是有效背离
        bearish = (
            (close == rolling_max_close)
            & (rsi < rolling_max_rsi.shift(1))
            & rolling_max_close.notna()
        )
        # 底背离：价格创新低 + RSI未创新低
        bullish = (
            (close == rolling_min_close)
            & (rsi > rolling_min_rsi.shift(1))
            & rolling_min_close.notna()
        )
        return bearish.fillna(False), bullish.fillna(False)

    def _normalize(self, series: pd.Series, lag: bool = True) -> pd.Series:
        """
        滚动归一化到[0,1]。

        Args:
            series: 输入序列
            lag: 是否使用shift(1)避免使用当前值。默认True，
                 严格避免当前点参与计算min/max导致的轻微泄露。
        """
        w = self.config.normalize_window
        src = series.shift(1) if lag else series
        rolling_min = src.rolling(window=w, min_periods=1).min()
        rolling_max = src.rolling(window=w, min_periods=1).max()
        denom = rolling_max - rolling_min
        normalized = np.where(denom > 0, (series - rolling_min) / denom, 0.5)
        return pd.Series(normalized, index=series.index).fillna(0.5)

    # ----------------------------------------------------------------
    # 指标计算入口
    # ----------------------------------------------------------------

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算全部12个量化指标。

        输入DataFrame需包含: close, high, low
        可选: volume, open_interest

        Raises:
            ValueError: 缺少必需列时抛出
        """
        # 必需列检查
        required = {"close", "high", "low"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"缺少必需列: {missing}")

        result = df.copy()
        close = result["close"]
        high = result["high"]
        low = result["low"]

        # volume处理
        if "volume" in result.columns:
            volume = result["volume"]
        else:
            warnings.warn("volume列缺失，volume_strength将设为1.0")
            volume = pd.Series(np.ones(len(result)), index=result.index)
            result["_volume_missing"] = True

        # 1. ADX
        adx, plus_di, minus_di = self.compute_adx(high, low, close)
        result["adx"] = adx
        result["plus_di"] = plus_di
        result["minus_di"] = minus_di

        # 2. 趋势方向
        result["trend_direction"] = self.compute_trend_direction(close)

        # 3. 波动率水平 (ATR/Close)
        atr = self.compute_atr(high, low, close)
        result["atr"] = atr
        vol_level = atr / close.replace(0, np.nan)
        result["vol_level"] = vol_level
        result["vol_level_norm"] = self._normalize(vol_level, lag=True)

        # 4. 波动率压缩（直接使用 atr_short / atr_long）
        atr_short = self.compute_atr(
            high, low, close, period=self.config.vol_short_period
        )
        atr_long = self.compute_atr(
            high, low, close, period=self.config.vol_long_period
        )
        compression_raw = atr_short / atr_long.replace(0, np.nan)
        result["compression"] = self._normalize(compression_raw, lag=True)

        # 5. RSI
        rsi = self.compute_rsi(close)
        result["rsi"] = rsi

        # 6. 成交量强度
        result["volume_strength"] = self.compute_volume_strength(volume)

        # 7. 持仓量变化
        if "open_interest" in result.columns:
            result["oi_change"] = self.compute_oi_change(result["open_interest"])
        else:
            warnings.warn("open_interest列缺失，oi_change将设为0.0")
            result["oi_change"] = 0.0
            result["_oi_missing"] = True

        # 8. 布林带宽度
        result["bb_width"] = self.compute_bollinger_width(close)

        # 9. 布林带位置
        result["bb_position"] = self.compute_bollinger_position(close)

        # 10. 趋势一致性
        result["trend_consistency"] = self.compute_trend_consistency(plus_di, minus_di)

        # 11. 加速度
        result["acceleration"] = self.compute_acceleration(close)

        # 12. 背离检测
        bearish_div, bullish_div = self.detect_divergence(close, rsi)
        result["bearish_divergence"] = bearish_div
        result["bullish_divergence"] = bullish_div

        # 13. 背离强度（合并顶底背离为连续指标，用于IC计算）
        result["divergence_strength"] = (
            bearish_div.astype(float) * (-1) + bullish_div.astype(float) * 1
        )

        return result

    # ----------------------------------------------------------------
    # 动态阈值（滚动百分位数，无前视偏差）
    # ----------------------------------------------------------------

    def _compute_rolling_threshold(
        self, series: pd.Series, percentile: float
    ) -> pd.Series:
        """
        计算滚动百分位数阈值。

        使用shift(1)确保仅用历史数据，避免未来信息泄露。

        Args:
            series: 原始指标序列
            percentile: 百分位数（0-100）

        Returns:
            滚动阈值序列
        """
        w = self.config.threshold_window
        return (
            series.shift(1)
            .rolling(window=w, min_periods=max(w // 2, 20))
            .quantile(percentile / 100.0)
        )

    def compute_dynamic_thresholds(
        self, indicators: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        计算所有动态阈值。

        所有阈值基于shift(1)的滚动百分位数，无前视偏差。
        阈值裁剪到合理范围。

        Returns:
            {阈值名: 阈值Series}
        """
        cfg = self.config
        thresholds = {}

        # ADX阈值
        thresholds["adx_trend"] = self._compute_rolling_threshold(
            indicators["adx"], cfg.adx_trend_percentile
        )
        thresholds["adx_strong"] = self._compute_rolling_threshold(
            indicators["adx"], cfg.adx_strong_percentile
        )

        # RSI阈值
        thresholds["rsi_overbought"] = self._compute_rolling_threshold(
            indicators["rsi"], cfg.rsi_overbought_percentile
        )
        thresholds["rsi_oversold"] = self._compute_rolling_threshold(
            indicators["rsi"], cfg.rsi_oversold_percentile
        )

        # 波动率阈值（裁剪到[0,1]）
        thresholds["vol_high"] = self._compute_rolling_threshold(
            indicators["vol_level_norm"], cfg.high_vol_percentile
        ).clip(0, 1)
        thresholds["vol_low"] = self._compute_rolling_threshold(
            indicators["vol_level_norm"], cfg.low_vol_percentile
        ).clip(0, 1)

        # 布林带位置阈值（裁剪到[0,1]）
        bb_pos = pd.Series(indicators["bb_position"].values, index=indicators.index)
        thresholds["bb_upper"] = self._compute_rolling_threshold(bb_pos, 90.0).clip(
            0, 1
        )
        thresholds["bb_lower"] = self._compute_rolling_threshold(bb_pos, 10.0).clip(
            0, 1
        )

        self._dynamic_thresholds = thresholds
        return thresholds

    # ----------------------------------------------------------------
    # 动态权重（滚动IC，无前视偏差）
    # ----------------------------------------------------------------

    def compute_ic_weights_rolling(
        self, indicators: pd.DataFrame, returns: pd.Series
    ) -> pd.DataFrame:
        """
        基于滚动IC计算各时间点的动态权重。

        t时刻的权重仅基于[t-ic_window, t-1]的数据计算，
        不使用任何未来信息。

        Args:
            indicators: 指标DataFrame
            returns: 收益率序列（应为历史收益率，非未来收益率）

        Returns:
            DataFrame，每行一个时间点，列为各指标权重
        """
        cfg = self.config
        ic_window = cfg.ic_window

        # 计算每个指标的滚动IC
        ic_abs_dict = {}
        for name in IC_INDICATORS:
            if name not in indicators.columns:
                continue
            ind_series = indicators[name].astype(float)
            # 滚动IC：过去ic_window天内指标与收益的相关系数
            ic_series = ind_series.rolling(
                window=ic_window, min_periods=ic_window // 2
            ).corr(returns)
            ic_abs_dict[name] = ic_series.abs()

        if not ic_abs_dict:
            # 无可用指标，返回等权
            return pd.DataFrame(
                {name: EQUAL_WEIGHT for name in IC_INDICATORS},
                index=indicators.index,
            )

        ic_abs_df = pd.DataFrame(ic_abs_dict, index=indicators.index)

        # 按ic_recalc_freq频率重算权重（降低开销）
        freq = cfg.ic_recalc_freq
        if freq > 1:
            # 只在每隔freq天重算，中间用前值填充
            mask = np.zeros(len(ic_abs_df), dtype=bool)
            indices = np.arange(0, len(ic_abs_df), freq)
            mask[indices] = True
            mask[0] = True
            ic_abs_df = ic_abs_df.where(pd.Series(mask, index=ic_abs_df.index)).ffill()

        # 归一化为权重
        row_sums = ic_abs_df.sum(axis=1)
        # IC均值过低时退化为等权
        mean_ic = row_sums.mean() / len(ic_abs_dict)
        if mean_ic < 0.02:
            weights_df = pd.DataFrame(
                {name: EQUAL_WEIGHT for name in ic_abs_dict},
                index=indicators.index,
            )
        else:
            weights_df = ic_abs_df.div(row_sums.replace(0, np.nan), axis=0)
            weights_df = weights_df.fillna(EQUAL_WEIGHT)

        # 确保所有IC_INDICATORS列都存在
        for name in IC_INDICATORS:
            if name not in weights_df.columns:
                weights_df[name] = EQUAL_WEIGHT

        self._ic_weights_matrix = weights_df
        return weights_df

    def compute_ic_weights(
        self, indicators: pd.DataFrame, returns: pd.Series
    ) -> Dict[str, float]:
        """
        计算全局IC权重（兼容旧接口）。

        注意：此方法使用全量数据计算IC，仅用于分析，不应用于回测。
        回测请使用 compute_ic_weights_rolling。

        Args:
            indicators: 指标DataFrame
            returns: 收益率序列

        Returns:
            {指标名: 权重}
        """
        ic_values = {}
        for name in IC_INDICATORS:
            if name not in indicators.columns:
                continue
            valid = indicators[name].notna() & returns.notna()
            if valid.sum() > 20:
                ic = indicators[name][valid].corr(returns[valid])
                ic_values[name] = abs(ic) if not pd.isna(ic) else 0.0

        mean_ic = np.mean(list(ic_values.values())) if ic_values else 0
        if mean_ic < 0.02 or not ic_values:
            return {name: EQUAL_WEIGHT for name in IC_INDICATORS}

        total = sum(ic_values.values())
        return {name: val / total for name, val in ic_values.items()}

    # ----------------------------------------------------------------
    # 环境分类
    # ----------------------------------------------------------------

    def classify_regime(
        self,
        indicators: pd.DataFrame,
        thresholds: Optional[Dict[str, pd.Series]] = None,
        weights_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        基于指标进行市场环境分类。

        使用动态阈值、IC权重，输出连续分数和离散标签。

        Args:
            indicators: 指标DataFrame
            thresholds: 动态阈值字典，None则使用默认值
            weights_df: 滚动IC权重矩阵，None则使用等权

        Returns:
            DataFrame，包含连续分数、离散标签和置信度
        """
        cfg = self.config
        n = len(indicators)

        # 获取指标列
        adx = indicators["adx"]
        trend_dir = indicators["trend_direction"]
        vol_level = indicators["vol_level_norm"]
        compression = indicators["compression"]
        rsi = indicators["rsi"]
        bb_pos = indicators["bb_position"]
        trend_consistency = indicators["trend_consistency"]
        bearish_div = indicators["bearish_divergence"]
        bullish_div = indicators["bullish_divergence"]
        acceleration = indicators["acceleration"]

        # 计算动态阈值
        if thresholds is None:
            thresholds = self.compute_dynamic_thresholds(indicators)

        # 提取阈值为numpy数组
        adx_trend_arr = self._get_threshold_arr(thresholds, "adx_trend", 25.0, n)
        rsi_ob_arr = self._get_threshold_arr(thresholds, "rsi_overbought", 70.0, n)
        rsi_os_arr = self._get_threshold_arr(thresholds, "rsi_oversold", 30.0, n)
        vol_high_arr = self._get_threshold_arr(thresholds, "vol_high", 0.75, n)
        vol_low_arr = self._get_threshold_arr(thresholds, "vol_low", 0.25, n)
        bb_upper_arr = self._get_threshold_arr(thresholds, "bb_upper", 0.9, n)
        bb_lower_arr = self._get_threshold_arr(thresholds, "bb_lower", 0.1, n)

        # 转为numpy数组，NaN先前向填充再补中性值（避免 hard-coded fill 扭曲早期信号）
        adx_arr = adx.ffill().fillna(0.0).values  # ADX NaN → 0（无趋势）
        dir_arr = trend_dir.ffill().fillna(0.0).values
        vol_arr = vol_level.ffill().fillna(0.5).values
        comp_arr = compression.ffill().fillna(0.5).values
        rsi_arr = rsi.ffill().fillna(50.0).values  # RSI NaN → 50（中性）
        bb_arr = bb_pos.ffill().fillna(0.5).values
        cons_arr = trend_consistency.ffill().fillna(0.0).values
        bearish_arr = bearish_div.ffill().fillna(False).values.astype(float)
        bullish_arr = bullish_div.ffill().fillna(False).values.astype(float)
        acc_arr = acceleration.ffill().fillna(0.0).values

        # 提取权重为numpy数组（每个时间点可能不同）
        if weights_df is not None:
            w_adx = self._get_weight_arr(weights_df, "adx", n)
            w_vol = self._get_weight_arr(weights_df, "vol_level_norm", n)
            w_comp = self._get_weight_arr(weights_df, "compression", n)
            w_rsi = self._get_weight_arr(weights_df, "rsi", n)
            w_bb = self._get_weight_arr(weights_df, "bb_position", n)
            w_cons = self._get_weight_arr(weights_df, "trend_consistency", n)
            w_acc = self._get_weight_arr(weights_df, "acceleration", n)
            w_div = self._get_weight_arr(weights_df, "divergence_strength", n)
        else:
            # 等权
            w_adx = np.full(n, 1.0 / len(IC_INDICATORS))
            w_vol = np.full(n, 1.0 / len(IC_INDICATORS))
            w_comp = np.full(n, 1.0 / len(IC_INDICATORS))
            w_rsi = np.full(n, 1.0 / len(IC_INDICATORS))
            w_bb = np.full(n, 1.0 / len(IC_INDICATORS))
            w_cons = np.full(n, 1.0 / len(IC_INDICATORS))
            w_acc = np.full(n, 1.0 / len(IC_INDICATORS))
            w_div = np.full(n, 1.0 / len(IC_INDICATORS))

        # --- 向量化计算各环境连续分数 ---
        adx_norm = np.clip(adx_arr / np.maximum(adx_trend_arr, 1.0), 0, 2) / 2.0

        # 趋势上涨
        s_trend_up = (
            w_adx * adx_norm * (dir_arr > 0).astype(float)
            + w_cons * cons_arr * (dir_arr > 0).astype(float)
            + w_div * (1 - bearish_arr)
            + w_acc * np.clip(acc_arr, 0, None)
        )

        # 趋势下跌
        s_trend_down = (
            w_adx * adx_norm * (dir_arr < 0).astype(float)
            + w_cons * cons_arr * (dir_arr < 0).astype(float)
            + w_div * (1 - bullish_arr)
            + w_acc * np.clip(-acc_arr, 0, None)
        )

        # 区间震荡
        s_range = w_adx * (1 - adx_norm) + w_cons * (1 - cons_arr)

        # 高波动
        vol_above = np.clip(
            (vol_arr - vol_high_arr) / np.maximum(1 - vol_high_arr, 0.01), 0, 1
        )
        s_high_vol = w_vol * vol_above + w_comp * (1 - comp_arr)

        # 低波动
        vol_below = np.clip(
            (vol_low_arr - vol_arr) / np.maximum(vol_low_arr, 0.01), 0, 1
        )
        s_low_vol = w_vol * vol_below + w_comp * comp_arr

        # 突破
        s_breakout = w_comp * comp_arr * 0.6 + w_adx * adx_norm * 0.4

        # 牛市衰竭
        rsi_above_ob = np.clip(
            (rsi_arr - rsi_ob_arr) / np.maximum(100 - rsi_ob_arr, 1.0), 0, 1
        )
        bb_above = np.clip(
            (bb_arr - bb_upper_arr) / np.maximum(1 - bb_upper_arr, 0.01), 0, 1
        )
        s_exh_bull = w_div * bearish_arr + w_rsi * rsi_above_ob + w_bb * bb_above

        # 熊市衰竭
        rsi_below_os = np.clip(
            (rsi_os_arr - rsi_arr) / np.maximum(rsi_os_arr, 1.0), 0, 1
        )
        bb_below = np.clip(
            (bb_lower_arr - bb_arr) / np.maximum(bb_lower_arr, 0.01), 0, 1
        )
        s_exh_bear = w_div * bullish_arr + w_rsi * rsi_below_os + w_bb * bb_below

        # 归一化各分数到[0,1]
        all_raw = np.column_stack(
            [
                s_trend_up,
                s_trend_down,
                s_range,
                s_high_vol,
                s_low_vol,
                s_breakout,
                s_exh_bull,
                s_exh_bear,
            ]
        )
        row_max = np.maximum(all_raw.max(axis=1), 1e-8)
        all_norm = all_raw / row_max[:, np.newaxis]

        # 方案A：对每个连续分数应用滚动均值平滑（与确认窗口同参数），
        # 使离散标签与连续分数保持一致。
        regime_values = [r.value for r in MarketRegime if r != MarketRegime.UNKNOWN]
        score_df = pd.DataFrame(all_norm, columns=[f"score_{v}" for v in regime_values])
        smooth_window = max(1, cfg.confirm_days)
        smoothed = score_df.rolling(window=smooth_window, min_periods=1).mean()
        smoothed_arr = smoothed.values

        # 用平滑后的分数重新确定离散标签
        best_idx = np.argmax(smoothed_arr, axis=1)
        regimes_raw = [regime_values[i] for i in best_idx]

        # ── 精简为4种核心环境类型 ──
        # trend_up, trend_down, breakout → trend（强趋势）
        # low_volatility → weak_trend（弱趋势）
        # range_bound, exhaustion_bull, exhaustion_bear → range（震荡）
        # high_volatility → high_vol（高波动）
        REGIME_SIMPLIFY = {
            "trend_up": "trend_up",
            "trend_down": "trend_down",
            "breakout": "trend_up",  # 突破归入趋势
            "low_volatility": "low_volatility",  # 保留低波动
            "range_bound": "range_bound",
            "exhaustion_bull": "range_bound",  # 牛市衰竭归入震荡
            "exhaustion_bear": "range_bound",  # 熊市衰竭归入震荡
            "high_volatility": "high_volatility",
        }
        regimes = [REGIME_SIMPLIFY.get(r, r) for r in regimes_raw]

        # 置信度 = 最高分 - 第二高分（基于平滑分数）
        sorted_scores = np.sort(smoothed_arr, axis=1)
        confidences = sorted_scores[:, -1] - sorted_scores[:, -2]

        # 确认窗口（状态机，无后视）
        regimes, confidences = self._apply_confirm_window(
            regimes, confidences, cfg.confirm_days
        )

        # 构建结果
        result = pd.DataFrame(
            {
                "regime": regimes,
                "regime_label": [MarketRegime(r).name for r in regimes],
                "regime_confidence": confidences,
            }
        )

        # 输出平滑后的连续分数（与确认后的标签一致）
        for idx, regime_val in enumerate(regime_values):
            result[f"score_{regime_val}"] = smoothed_arr[:, idx]

        return result

    @staticmethod
    def _get_threshold_arr(
        thresholds: Dict[str, pd.Series], key: str, default: float, n: int
    ) -> np.ndarray:
        """从阈值字典提取numpy数组，缺失时用默认值填充。"""
        if key in thresholds:
            arr = thresholds[key].values
            return np.nan_to_num(arr, nan=default)
        return np.full(n, default)

    @staticmethod
    def _get_weight_arr(weights_df: pd.DataFrame, name: str, n: int) -> np.ndarray:
        """从权重矩阵提取numpy数组。"""
        if name in weights_df.columns:
            arr = weights_df[name].values
            return np.nan_to_num(arr, nan=EQUAL_WEIGHT)
        return np.full(n, EQUAL_WEIGHT)

    def _apply_confirm_window(
        self, regimes: List[str], confidences: np.ndarray, confirm_days: int
    ) -> Tuple[List[str], np.ndarray]:
        """
        确认窗口：经典状态机实现，无后视检查。

        维护 current（已确认的regime）和 candidate（候选regime）及连续计数。
        只有当候选regime连续出现 confirm_days 次后，才将 current 切换为该候选。
        """
        if confirm_days <= 1 or len(regimes) == 0:
            return regimes, confidences

        confirmed = []
        confirmed_conf = []
        current = regimes[0]  # 已确认的regime
        candidate = regimes[0]  # 正在计数的候选regime
        counter = 0  # 候选regime连续出现的次数

        for i in range(len(regimes)):
            if regimes[i] == candidate:
                counter += 1
            else:
                candidate = regimes[i]
                counter = 1

            # 候选regime连续出现confirm_days次，切换current
            if counter >= confirm_days:
                current = candidate

            confirmed.append(current)
            confirmed_conf.append(confidences[i])

        return confirmed, np.array(confirmed_conf)

    # ----------------------------------------------------------------
    # fit / transform / fit_transform（消除前视偏差核心）
    # ----------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "MarketRegimeDetector":
        """
        在数据上拟合参数（IC权重、动态阈值）。

        所有参数计算仅使用历史数据，无前视偏差。
        拟合后可调用transform对新数据分类。

        Args:
            df: 包含OHLCV数据的DataFrame

        Returns:
            self
        """
        # 计算指标
        indicators = self.compute_indicators(df)

        # 计算历史收益率（非未来收益率！）
        close = df["close"]
        hist_return = close.pct_change(5)  # 过去5日收益，无shift(-k)

        # 计算滚动IC权重
        self.compute_ic_weights_rolling(indicators, hist_return)

        # 计算动态阈值
        self.compute_dynamic_thresholds(indicators)

        # 存储拟合终值（用于transform）
        if self._ic_weights_matrix is not None:
            last_valid = (
                self._ic_weights_matrix.dropna().iloc[-1]
                if len(self._ic_weights_matrix.dropna()) > 0
                else None
            )
            if last_valid is not None:
                self._fitted_ic_weights = last_valid.to_dict()

        if self._dynamic_thresholds is not None:
            for key, series in self._dynamic_thresholds.items():
                last_val = (
                    series.dropna().iloc[-1] if len(series.dropna()) > 0 else None
                )
                if last_val is not None:
                    self._fitted_threshold_values[key] = float(last_val)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用已拟合的参数对新数据分类（v2: 滚动窗口阈值更新 + 稳定性指标）。

        v2 改进：
          - 每 rolling_update_freq 个交易日使用滚动窗口重新计算阈值
          - 计算环境稳定性指标（过去 stability_window 次分类中同一环境的比例）
          - 输出 regime_stability 列（0~1），低于 stability_threshold 时标记为不稳定

        Args:
            df: 包含OHLCV数据的DataFrame

        Returns:
            带有环境标签、连续分数、置信度和稳定性的DataFrame

        Raises:
            RuntimeError: 未调用fit时抛出
        """
        if not self._fitted:
            raise RuntimeError("请先调用fit()方法")

        indicators = self.compute_indicators(df)
        n = len(indicators)
        cfg = self.config

        # ── 滚动窗口阈值更新 ──
        # 每 rolling_update_freq 个 bar 使用过去 rolling_window 个 bar 重新计算阈值
        # 其余 bar 使用上一次计算的阈值（或拟合终值作为初始值）
        thresholds = {}
        for key, val in self._fitted_threshold_values.items():
            thresholds[key] = np.full(n, val)

        if n > cfg.rolling_window:
            for update_idx in range(cfg.rolling_window, n, cfg.rolling_update_freq):
                window_start = max(0, update_idx - cfg.rolling_window)
                window_indicators = indicators.iloc[window_start:update_idx]
                window_thresholds = self.compute_dynamic_thresholds(window_indicators)
                for key, series in window_thresholds.items():
                    if key in thresholds and len(series.dropna()) > 0:
                        last_val = float(series.dropna().iloc[-1])
                        thresholds[key][update_idx:] = last_val

        for key in thresholds:
            thresholds[key] = pd.Series(thresholds[key], index=indicators.index)

        # 使用拟合的固定IC权重（IC权重计算成本高，保持静态）
        weights_df = pd.DataFrame(
            {
                name: self._fitted_ic_weights.get(name, EQUAL_WEIGHT)
                for name in IC_INDICATORS
            },
            index=indicators.index,
        )

        # 分类
        regime_df = self.classify_regime(indicators, thresholds, weights_df)

        # ── 环境稳定性指标 ──
        # 计算过去 stability_window 次分类中同一环境出现的比例
        regimes = regime_df["regime"].values
        stability = np.ones(n)
        sw = cfg.stability_window
        for i in range(sw, n):
            window_regimes = regimes[i - sw : i]
            current_regime = regimes[i]
            same_count = np.sum(window_regimes == current_regime)
            stability[i] = same_count / sw
        for i in range(min(sw, n)):
            if i > 0:
                window_regimes = regimes[:i]
                same_count = np.sum(window_regimes == regimes[i])
                stability[i] = same_count / i if i > 0 else 1.0

        regime_df["regime_stability"] = stability

        # 不稳定时降低置信度
        unstable_mask = stability < cfg.stability_threshold
        if "regime_confidence" in regime_df.columns:
            original_conf = regime_df["regime_confidence"].values.copy()
            adjusted_conf = original_conf.copy()
            adjusted_conf[unstable_mask] *= 0.7
            regime_df["regime_confidence"] = adjusted_conf

        # 合并
        indicator_cols = [c for c in indicators.columns if c not in regime_df.columns]
        combined = pd.concat(
            [
                indicators[indicator_cols].reset_index(drop=True),
                regime_df.reset_index(drop=True),
            ],
            axis=1,
        )

        return combined

    def fit_transform(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        拟合并转换（回测场景）。

        在整个数据集上计算滚动参数（无未来偏差），然后变换。
        滚动参数在每个时间点仅使用历史数据。

        Args:
            df: 包含OHLCV数据的DataFrame
            verbose: 是否打印进度

        Returns:
            带有环境标签、连续分数和置信度的DataFrame
        """
        # 计算指标
        indicators = self.compute_indicators(df)

        # 计算历史收益率（非未来收益率）
        close = df["close"]
        hist_return = close.pct_change(5)

        # 计算滚动IC权重（无前视偏差）
        if verbose:
            print("计算滚动IC权重...")
        weights_df = self.compute_ic_weights_rolling(indicators, hist_return)

        # 计算动态阈值（无前视偏差）
        if verbose:
            print("计算动态阈值...")
        thresholds = self.compute_dynamic_thresholds(indicators)

        # 分类
        if verbose:
            print("环境分类...")
        regime_df = self.classify_regime(indicators, thresholds, weights_df)

        # 合并
        indicator_cols = [c for c in indicators.columns if c not in regime_df.columns]
        combined = pd.concat(
            [
                indicators[indicator_cols].reset_index(drop=True),
                regime_df.reset_index(drop=True),
            ],
            axis=1,
        )

        # 存储拟合状态
        if weights_df is not None:
            last_valid = (
                weights_df.dropna().iloc[-1] if len(weights_df.dropna()) > 0 else None
            )
            if last_valid is not None:
                self._fitted_ic_weights = last_valid.to_dict()
        if thresholds is not None:
            for key, series in thresholds.items():
                last_val = (
                    series.dropna().iloc[-1] if len(series.dropna()) > 0 else None
                )
                if last_val is not None:
                    self._fitted_threshold_values[key] = float(last_val)
        self._fitted = True

        return combined

    # ----------------------------------------------------------------
    # 权重映射（向前兼容 EnvironmentAdapter）
    # ----------------------------------------------------------------

    # 8种环境 → 策略类型映射
    _REGIME_WEIGHT_MAP = {
        "trend_up": {"trend": 0.55, "reversal": 0.15, "spread": 0.15, "momentum": 0.15},
        "trend_down": {
            "trend": 0.50,
            "reversal": 0.20,
            "spread": 0.15,
            "momentum": 0.15,
        },
        "range_bound": {
            "trend": 0.15,
            "reversal": 0.55,
            "spread": 0.20,
            "momentum": 0.10,
        },
        "high_volatility": {
            "trend": 0.20,
            "reversal": 0.30,
            "spread": 0.10,
            "momentum": 0.40,
        },
        "low_volatility": {
            "trend": 0.25,
            "reversal": 0.25,
            "spread": 0.20,
            "momentum": 0.30,
        },
        "breakout": {"trend": 0.30, "reversal": 0.10, "spread": 0.10, "momentum": 0.50},
        "exhaustion_bull": {
            "trend": 0.10,
            "reversal": 0.60,
            "spread": 0.10,
            "momentum": 0.20,
        },
        "exhaustion_bear": {
            "trend": 0.10,
            "reversal": 0.60,
            "spread": 0.10,
            "momentum": 0.20,
        },
    }

    # 向后兼容的趋势/震荡 → 策略权重映射
    _SIMPLE_WEIGHT_MAP = {
        "trend": {"trend": 0.50, "reversal": 0.25, "spread": 0.25},
        "range": {"trend": 0.20, "reversal": 0.50, "spread": 0.30},
    }

    def get_regime_weights(self, regime: str, separator: str = "_") -> Dict[str, float]:
        """
        根据市场环境获取策略权重分配。

        支持两种输入格式:
          1. 8种细分环境: trend_up, trend_down, range_bound, high_volatility,
             low_volatility, breakout, exhaustion_bull, exhaustion_bear
          2. 向后兼容的简化格式: "trend" / "range"
             → 默认从当前活跃的细分环境中推断简化标签

        Args:
            regime: 市场环境标签
            separator: 用于构建归一化键的分隔符，默认 "_"

        Returns:
            {策略类型: 权重} 的字典
        """
        # 先尝试直接匹配8种环境
        normalized = regime.lower().strip()
        if normalized in self._REGIME_WEIGHT_MAP:
            return dict(self._REGIME_WEIGHT_MAP[normalized])

        # 简化标签匹配（向后兼容 EnvironmentAdapter 的 "trend"/"range"）
        if normalized in self._SIMPLE_WEIGHT_MAP:
            return dict(self._SIMPLE_WEIGHT_MAP[normalized])

        # 尝试用分隔符构建键
        constructed = (
            regime.lower().strip().replace("-", separator).replace(" ", separator)
        )
        if constructed in self._REGIME_WEIGHT_MAP:
            return dict(self._REGIME_WEIGHT_MAP[constructed])

        # 兜底: 等权分配
        return {"trend": 0.34, "reversal": 0.33, "spread": 0.33}

    # ----------------------------------------------------------------
    # 主入口（兼容旧接口）
    # ----------------------------------------------------------------

    def detect(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        完整的市场环境识别流程（兼容旧接口）。

        内部调用fit_transform，确保无前视偏差。

        Args:
            df: 包含OHLCV数据的DataFrame
            verbose: 是否打印进度

        Returns:
            带有环境标签、连续分数和置信度的DataFrame
        """
        if "symbol" in df.columns and df["symbol"].nunique() > 1:
            results = []
            symbols = df["symbol"].unique()
            for idx, sym in enumerate(symbols):
                if verbose:
                    print(f"处理品种 {idx + 1}/{len(symbols)}: {sym}")
                group = df[df["symbol"] == sym].sort_values("date").copy()
                result = self.fit_transform(group, verbose=verbose)
                results.append(result)
            return pd.concat(results, ignore_index=True)
        else:
            return self.fit_transform(df, verbose=verbose)

    # ----------------------------------------------------------------
    # 样本外验证（fit/transform分离，无前视偏差）
    # ----------------------------------------------------------------

    def validate(
        self, df: pd.DataFrame, strategy_returns: Optional[pd.Series] = None
    ) -> ValidationResult:
        """
        样本外验证：用样本内参数对样本外分类。

        核心原则：样本外数据不参与参数计算。
        1. 在样本内fit
        2. 用fit的参数transform样本外
        3. 比较分布稳定性和IC衰减

        Args:
            df: 包含OHLCV数据的DataFrame
            strategy_returns: 可选，策略收益率序列

        Returns:
            ValidationResult
        """
        cfg = self.config
        split_ratio = cfg.validation_split

        # 按时间排序
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        split_idx = int(n * (1 - split_ratio))

        df_in = df.iloc[:split_idx].copy()
        df_out = df.iloc[split_idx:].copy()

        # 样本内：fit_transform（使用滚动参数）
        result_in = self.fit_transform(df_in)

        # 样本外：用样本内fit的参数transform（不重新计算参数！）
        result_out = self.transform(df_out)

        # 1. 环境分布稳定性（KL散度）
        dist_in = result_in["regime"].value_counts(normalize=True).to_dict()
        dist_out = result_out["regime"].value_counts(normalize=True).to_dict()
        kl_div = self._kl_divergence(dist_in, dist_out)

        # 2. IC衰减（使用历史收益率，非未来收益率）
        indicators_in = self.compute_indicators(df_in)
        indicators_out = self.compute_indicators(df_out)

        ret_in = df_in["close"].pct_change(5)
        ret_out = df_out["close"].pct_change(5)

        ic_in = self._compute_ic_dict(indicators_in, ret_in)
        ic_out = self._compute_ic_dict(indicators_out, ret_out)

        # IC衰减率
        ic_decay_values = []
        for key in ic_in:
            if ic_in[key] != 0:
                decay = 1 - abs(ic_out.get(key, 0)) / abs(ic_in[key])
                ic_decay_values.append(max(0, decay))
        ic_decay = np.mean(ic_decay_values) if ic_decay_values else 1.0

        # 3. 环境对策略表现的区分能力
        regime_sharpe_diff = {}
        if strategy_returns is not None:
            # 按日期对齐
            strat_df = pd.DataFrame(
                {
                    "date": df_out["date"].values,
                    "strategy_return": strategy_returns.values[: len(df_out)]
                    if len(strategy_returns) >= len(df_out)
                    else np.pad(
                        strategy_returns.values,
                        (0, max(0, len(df_out) - len(strategy_returns))),
                    ),
                }
            )
            combined_out = result_out.copy()
            combined_out["date"] = df_out["date"].values
            combined_out = combined_out.merge(strat_df, on="date", how="left")

            for regime_val in combined_out["regime"].unique():
                mask = combined_out["regime"] == regime_val
                regime_returns = combined_out.loc[mask, "strategy_return"].dropna()
                if len(regime_returns) > 10 and regime_returns.std() > 0:
                    sharpe = regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                    regime_sharpe_diff[regime_val] = sharpe

        return ValidationResult(
            in_sample_regime_dist=dist_in,
            out_sample_regime_dist=dist_out,
            distribution_stability=kl_div,
            ic_in_sample=ic_in,
            ic_out_sample=ic_out,
            ic_decay=ic_decay,
            regime_sharpe_diff=regime_sharpe_diff,
        )

    def _compute_ic_dict(
        self, indicators: pd.DataFrame, returns: pd.Series
    ) -> Dict[str, float]:
        """计算各指标的IC均值（使用历史收益率）。"""
        ic_dict = {}
        for name in IC_INDICATORS:
            if name not in indicators.columns:
                continue
            valid = indicators[name].notna() & returns.notna()
            if valid.sum() > 20:
                ic = indicators[name][valid].corr(returns[valid])
                ic_dict[name] = abs(ic) if not pd.isna(ic) else 0.0
        return ic_dict

    @staticmethod
    def _kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
        """计算KL散度 D_KL(P || Q)，加入epsilon平滑。"""
        epsilon = 1e-6
        all_keys = set(p.keys()) | set(q.keys())
        p_smooth = {k: p.get(k, epsilon) for k in all_keys}
        q_smooth = {k: q.get(k, epsilon) for k in all_keys}

        p_total = sum(p_smooth.values())
        q_total = sum(q_smooth.values())
        p_norm = {k: v / p_total for k, v in p_smooth.items()}
        q_norm = {k: v / q_total for k, v in q_smooth.items()}

        kl = sum(p_norm[k] * np.log(p_norm[k] / q_norm[k]) for k in all_keys)
        return kl

    # ----------------------------------------------------------------
    # 兼容接口
    # ----------------------------------------------------------------

    def get_regime_for_pybroker(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        输出兼容PyBroker的环境数据。

        保留env_前缀列名，与现有策略兼容。
        """
        result = self.detect(df)
        rename_map = {
            "atr": "env_atr",
            "adx": "env_adx",
            "plus_di": "env_plus_di",
            "minus_di": "env_minus_di",
            "regime": "env_market_regime",
            "regime_confidence": "env_regime_confidence",
            "compression": "env_compression_score",
            "rsi": "env_rsi",
            "bb_width": "env_bb_width",
            "bb_position": "env_bb_position",
            "trend_direction": "env_trend_direction",
            "trend_consistency": "env_trend_consistency",
            "vol_level_norm": "env_vol_level",
            "volume_strength": "env_volume_strength",
        }
        result = result.rename(columns=rename_map)
        result = result.loc[:, ~result.columns.duplicated()]
        return result

    def get_recommended_strategies(self, regime: MarketRegime) -> List[str]:
        """根据市场环境推荐策略。"""
        mapping = {
            MarketRegime.TREND_UP: ["dual_ma", "vol_breakout"],
            MarketRegime.TREND_DOWN: ["dual_ma", "vol_breakout"],
            MarketRegime.RANGE_BOUND: ["rsi", "term_structure"],
            MarketRegime.HIGH_VOLATILITY: ["term_structure"],
            MarketRegime.LOW_VOLATILITY: ["vol_breakout", "dual_ma"],
            MarketRegime.BREAKOUT: ["vol_breakout", "dual_ma"],
            MarketRegime.EXHAUSTION_BULL: ["term_structure", "rsi"],
            MarketRegime.EXHAUSTION_BEAR: ["dual_ma", "vol_breakout"],
        }
        return mapping.get(regime, ["dual_ma"])


class EnhancedTrendDetector:
    """
    增强版市场趋势判断引擎：
    - 多维度技术指标组合分析
    - 状态转换逻辑与置信度
    - 过滤机制防抖动
    """

    def __init__(self, config: Optional[TrendConfig] = None):
        self.config = config or TrendConfig()
        self._state_history: List[TrendType] = []
        self._confidence_history: List[float] = []

    def _compute_ma_system(self, close: pd.Series) -> Dict[str, pd.Series]:
        """
        均线系统指标（多周期均线）。
        包含短期、中期、长期均线及相互关系。
        """
        ma_short = close.rolling(
            window=self.config.ma_short, min_periods=self.config.ma_short
        ).mean()
        ma_medium = close.rolling(
            window=self.config.ma_medium, min_periods=self.config.ma_medium
        ).mean()
        ma_long = close.rolling(
            window=self.config.ma_long, min_periods=self.config.ma_long
        ).mean()

        ma_up = (ma_short > ma_medium) & (ma_medium > ma_long)
        ma_down = (ma_short < ma_medium) & (ma_medium < ma_long)

        spread_short_medium = (ma_short - ma_medium) / close
        spread_medium_long = (ma_medium - ma_long) / close

        pos_short = (close - ma_short) / ma_short
        pos_medium = (close - ma_medium) / ma_medium
        pos_long = (close - ma_long) / ma_long

        return {
            "ma_short": ma_short,
            "ma_medium": ma_medium,
            "ma_long": ma_long,
            "ma_up": ma_up.astype(float),
            "ma_down": ma_down.astype(float),
            "spread_sm": spread_short_medium,
            "spread_ml": spread_medium_long,
            "pos_s": pos_short,
            "pos_m": pos_medium,
            "pos_l": pos_long,
        }

    def _compute_macd(self, close: pd.Series) -> Dict[str, pd.Series]:
        """
        MACD 指标系统。
        包含 DIF、DEA、MACD柱状线、金叉/死叉。
        """
        ema_fast = close.ewm(
            span=self.config.macd_fast, min_periods=self.config.macd_fast
        ).mean()
        ema_slow = close.ewm(
            span=self.config.macd_slow, min_periods=self.config.macd_slow
        ).mean()

        dif = ema_fast - ema_slow
        dea = dif.ewm(
            span=self.config.macd_signal, min_periods=self.config.macd_signal
        ).mean()
        macd_bar = (dif - dea) * 2

        golden_cross = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        death_cross = (dif < dea) & (dif.shift(1) >= dea.shift(1))

        macd_up = (dif > 0) & (macd_bar > 0)
        macd_down = (dif < 0) & (macd_bar < 0)

        return {
            "macd_dif": dif,
            "macd_dea": dea,
            "macd_bar": macd_bar,
            "macd_golden_cross": golden_cross.astype(float),
            "macd_death_cross": death_cross.astype(float),
            "macd_up": macd_up.astype(float),
            "macd_down": macd_down.astype(float),
        }

    def _compute_rsi_system(self, close: pd.Series) -> Dict[str, pd.Series]:
        """
        RSI 指标系统。
        包含 RSI、超买/超卖、RSI趋势。
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(
            window=self.config.rsi_period, min_periods=self.config.rsi_period
        ).mean()
        avg_loss = loss.rolling(
            window=self.config.rsi_period, min_periods=self.config.rsi_period
        ).mean()
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        rsi = pd.Series(100 - 100 / (1 + rs), index=close.index)

        overbought = rsi >= self.config.rsi_overbought
        oversold = rsi <= self.config.rsi_oversold

        rsi_ma = rsi.rolling(window=5, min_periods=5).mean()
        rsi_rising = rsi > rsi_ma
        rsi_falling = rsi < rsi_ma

        return {
            "rsi": rsi,
            "rsi_overbought": overbought.astype(float),
            "rsi_oversold": oversold.astype(float),
            "rsi_rising": rsi_rising.astype(float),
            "rsi_falling": rsi_falling.astype(float),
        }

    def _compute_adx_system(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        ADX 指标系统。
        包含 ADX、+DI、-DI、趋势强度判断。
        """
        period = self.config.adx_period

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        if len(tr) > 0:
            tr.iloc[0] = tr1.iloc[0]

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

        strong_trend = adx >= self.config.adx_strong_threshold
        weak_trend = (adx >= self.config.adx_trend_threshold) & (
            adx < self.config.adx_strong_threshold
        )
        no_trend = adx < self.config.adx_trend_threshold

        up_direction = plus_di > minus_di
        down_direction = minus_di > plus_di

        return {
            "adx": adx,
            "adx_plus_di": plus_di,
            "adx_minus_di": minus_di,
            "adx_strong_trend": strong_trend.astype(float),
            "adx_weak_trend": weak_trend.astype(float),
            "adx_no_trend": no_trend.astype(float),
            "adx_up_direction": up_direction.astype(float),
            "adx_down_direction": down_direction.astype(float),
        }

    def _compute_bollinger_system(self, close: pd.Series) -> Dict[str, pd.Series]:
        """
        布林带系统指标。
        包含中轨、上下轨、带宽、位置、收缩/扩张。
        """
        period = 20
        std_dev = 2.0

        ma = close.rolling(window=period, min_periods=period).mean()
        std = close.rolling(window=period, min_periods=period).std()
        upper = ma + std_dev * std
        lower = ma - std_dev * std
        bandwidth = (upper - lower) / ma.replace(0, np.nan)
        position = (close - lower) / (upper - lower).replace(0, np.nan)

        bandwidth_ma = bandwidth.rolling(window=10, min_periods=10).mean()
        contracting = bandwidth < bandwidth_ma
        expanding = bandwidth > bandwidth_ma

        breakout_up = close > upper
        breakout_down = close < lower

        return {
            "bb_middle": ma,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_position": position,
            "bb_contracting": contracting.astype(float),
            "bb_expanding": expanding.astype(float),
            "bb_breakout_up": breakout_up.astype(float),
            "bb_breakout_down": breakout_down.astype(float),
        }

    def _compute_momentum_system(self, close: pd.Series) -> Dict[str, pd.Series]:
        """
        动量系统指标。
        包含简单动量、ROC、加速度。
        """
        mom_5 = (close - close.shift(5)) / close.shift(5)
        mom_20 = (close - close.shift(20)) / close.shift(20)

        roc_10 = (close / close.shift(10) - 1) * 100

        acceleration = mom_5.diff()

        mom_rising = mom_5 > mom_5.shift(1)
        mom_falling = mom_5 < mom_5.shift(1)

        return {
            "momentum_5": mom_5,
            "momentum_20": mom_20,
            "roc_10": roc_10,
            "acceleration": acceleration,
            "momentum_rising": mom_rising.astype(float),
            "momentum_falling": mom_falling.astype(float),
        }

    def compute_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标（6大类）。
        """
        if not {"close", "high", "low"}.issubset(df.columns):
            raise ValueError("数据缺少必要列: close, high, low")

        result = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]

        ma_ind = self._compute_ma_system(close)
        result = result.assign(**ma_ind)

        macd_ind = self._compute_macd(close)
        result = result.assign(**macd_ind)

        rsi_ind = self._compute_rsi_system(close)
        result = result.assign(**rsi_ind)

        adx_ind = self._compute_adx_system(high, low, close)
        result = result.assign(**adx_ind)

        bb_ind = self._compute_bollinger_system(close)
        result = result.assign(**bb_ind)

        mom_ind = self._compute_momentum_system(close)
        result = result.assign(**mom_ind)

        return result

    def compute_trend_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算各趋势类型的得分（0-1）。
        """
        indicators = self.compute_all_indicators(df)
        n = len(indicators)

        scores = pd.DataFrame(
            {
                "strong_up": np.zeros(n),
                "weak_up": np.zeros(n),
                "sideways": np.zeros(n),
                "weak_down": np.zeros(n),
                "strong_down": np.zeros(n),
            },
            index=indicators.index,
        )

        scores["strong_up"] = (
            indicators["adx_strong_trend"] * indicators["adx_up_direction"] * 0.3
            + indicators["ma_up"] * 0.2
            + indicators["macd_up"] * 0.2
            + indicators["momentum_rising"]
            * (indicators["momentum_5"] > 0).astype(float)
            * 0.15
            + indicators["bb_breakout_up"] * 0.15
        )

        scores["weak_up"] = (
            indicators["adx_weak_trend"] * indicators["adx_up_direction"] * 0.3
            + indicators["ma_up"] * indicators["adx_no_trend"] * 0.25
            + indicators["macd_up"] * 0.2
            + (indicators["pos_m"] > 0).astype(float) * 0.25
        )

        scores["sideways"] = (
            indicators["adx_no_trend"] * 0.35
            + indicators["bb_contracting"] * 0.2
            + (
                (indicators["bb_position"] > 0.2) & (indicators["bb_position"] < 0.8)
            ).astype(float)
            * 0.25
            + (indicators["momentum_5"].abs() < 0.02).astype(float) * 0.2
        )

        scores["weak_down"] = (
            indicators["adx_weak_trend"] * indicators["adx_down_direction"] * 0.3
            + indicators["ma_down"] * indicators["adx_no_trend"] * 0.25
            + indicators["macd_down"] * 0.2
            + (indicators["pos_m"] < 0).astype(float) * 0.25
        )

        scores["strong_down"] = (
            indicators["adx_strong_trend"] * indicators["adx_down_direction"] * 0.3
            + indicators["ma_down"] * 0.2
            + indicators["macd_down"] * 0.2
            + indicators["momentum_falling"]
            * (indicators["momentum_5"] < 0).astype(float)
            * 0.15
            + indicators["bb_breakout_down"] * 0.15
        )

        total = scores.sum(axis=1)
        total_safe = total.replace(0, 1)
        for col in scores.columns:
            scores[col] = scores[col] / total_safe

        return scores

    def detect_trend(self, df: pd.DataFrame, use_filter: bool = True) -> pd.DataFrame:
        """
        检测市场趋势。

        Args:
            df: OHLCV 数据
            use_filter: 是否启用状态转换过滤

        Returns:
            包含趋势判断结果的 DataFrame
        """
        indicators = self.compute_all_indicators(df)
        scores = self.compute_trend_scores(df)

        result = indicators.copy()

        result["trend_confidence"] = scores.max(axis=1)
        trend_type_map = {
            "strong_up": TrendType.STRONG_UP,
            "weak_up": TrendType.WEAK_UP,
            "sideways": TrendType.SIDEWAYS,
            "weak_down": TrendType.WEAK_DOWN,
            "strong_down": TrendType.STRONG_DOWN,
        }
        result["trend_type_str"] = scores.idxmax(axis=1)
        result["trend_type"] = result["trend_type_str"].map(trend_type_map)

        if use_filter:
            result = self._apply_state_transition_filter(result)

        self._state_history = result["trend_type"].tolist()
        self._confidence_history = result["trend_confidence"].tolist()

        return result

    def _apply_state_transition_filter(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        应用状态转换过滤机制。
        - 状态转换需达到置信度阈值
        - 确认窗口
        - 防抖动
        """
        filtered = result.copy()
        n = len(filtered)
        confirm_days = self.config.confirm_days
        threshold = self.config.state_transition_threshold

        if n == 0:
            return filtered

        current_trend = filtered["trend_type"].iloc[0]
        current_confidence = filtered["trend_confidence"].iloc[0]
        consecutive_count = 1

        filtered_trends = [current_trend]
        filtered_confidences = [current_confidence]

        for i in range(1, n):
            candidate_trend = filtered["trend_type"].iloc[i]
            candidate_confidence = filtered["trend_confidence"].iloc[i]

            if candidate_trend == current_trend:
                consecutive_count += 1
                current_confidence = candidate_confidence
            else:
                if (
                    candidate_confidence >= threshold
                    and consecutive_count >= confirm_days
                ):
                    current_trend = candidate_trend
                    current_confidence = candidate_confidence
                    consecutive_count = 1

            filtered_trends.append(current_trend)
            filtered_confidences.append(current_confidence)

        filtered["trend_type"] = filtered_trends
        filtered["trend_confidence"] = filtered_confidences

        return filtered

    def get_summary(self, result_df: pd.DataFrame) -> Dict:
        """
        获取趋势判断结果摘要。
        """
        if result_df.empty:
            return {}

        trend_counts = result_df["trend_type_str"].value_counts()
        avg_confidence = result_df["trend_confidence"].mean()

        return {
            "date_range": (
                str(result_df["date"].min().date()),
                str(result_df["date"].max().date()),
            ),
            "total_days": len(result_df),
            "trend_distribution": trend_counts.to_dict(),
            "average_confidence": avg_confidence,
        }
