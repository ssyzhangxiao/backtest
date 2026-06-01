"""
环境指标 — 将 MarketRegimeDetector 包装为 PyBroker 指标。

位置: core/engine/regime_indicator.py

提供:
  - RegimeIndicator: 环境检测器封装，支持 PyBroker indicator 注册
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from core.market_regime import MarketRegimeDetector

logger = logging.getLogger(__name__)


class RegimeIndicator:
    """
    将 MarketRegimeDetector 包装为可在 PyBroker 中使用的指标。

    提供 fit() / detect() 用于非 PyBroker 路径，
    以及 create_pybroker_fn() 返回可用于 @pybroker.indicator 的函数。

    注意：PyBroker 每个 bar 调用一次 indicator，缓存意义不大，
    因此不再使用实例级缓存，而是利用 PyBroker 的内置缓存机制。
    """

    def __init__(self, detector: Optional[MarketRegimeDetector] = None):
        self._detector = detector or MarketRegimeDetector()
        self._is_fitted = False

    def fit(self, df: pd.DataFrame):
        """在样本内数据上拟合探测器。"""
        dominant = df.copy()
        if "is_dominant" in dominant.columns:
            dominant = dominant[dominant["is_dominant"]]
        dominant = dominant.sort_values("date")
        self._detector.fit(dominant)
        self._is_fitted = True
        logger.info("RegimeIndicator 已拟合，样本内 %d 行", len(dominant))

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对整个 DataFrame 执行环境检测（非 PyBroker 路径使用）。

        Args:
            df: 行情数据 DataFrame。

        Returns:
            含 regime, regime_confidence 列的 DataFrame。
        """
        dominant = df.copy()
        if "is_dominant" in dominant.columns:
            dominant = dominant[dominant["is_dominant"]]
        dominant = dominant.sort_values("date")

        if self._is_fitted:
            result = self._detector.transform(dominant)
        else:
            result = self._detector.detect(dominant)
        return result

    def create_pybroker_regime_fn(self):
        """
        创建可用于 @pybroker.indicator('regime') 的函数。

        该函数接受 bar_data 并返回 numpy array：
          - regime: 环境标签字符串
          - regime_confidence: 置信度浮点数
          - regime_stability: 稳定性分数

        Returns:
            (fn_regime, fn_confidence, fn_stability) 三个可注册的函数。
        """
        detector = self._detector
        _min_bars = 20

        def _build_and_transform(bar_data, column, default):
            """构建 DataFrame、执行 transform、提取指定列（三闭包公共逻辑）。"""
            n = len(bar_data.date)
            if n < _min_bars:
                return default(n) if callable(default) else np.array([default] * n)
            try:
                data_dict = {
                    "open": bar_data.open,
                    "high": bar_data.high,
                    "low": bar_data.low,
                    "close": bar_data.close,
                    "volume": bar_data.volume,
                }
                if hasattr(bar_data, "open_interest") and bar_data.open_interest is not None:
                    data_dict["open_interest"] = bar_data.open_interest
                df = pd.DataFrame(
                    data_dict,
                    index=pd.to_datetime(bar_data.date),
                )
                result = detector.transform(df)
                if column in result.columns:
                    col_series = pd.Series(result[column].values, index=df.index)
                    fill_val = default if not callable(default) else (
                        0.5 if column == "regime_confidence" else ("unknown" if column == "regime" else 1.0)
                    )
                    return col_series.reindex(df.index, fill_value=fill_val).to_numpy()
            except Exception as e:
                logger.debug("%s 计算失败，回退: %s", column, e)
            return default(n) if callable(default) else np.array([default] * n)

        def regime_fn(bar_data):
            return _build_and_transform(bar_data, "regime", lambda n: np.array(["unknown"] * n))

        def regime_conf_fn(bar_data):
            return _build_and_transform(bar_data, "regime_confidence", lambda n: np.full(n, 0.5))

        def regime_stab_fn(bar_data):
            return _build_and_transform(bar_data, "regime_stability", lambda n: np.full(n, 1.0))

        return regime_fn, regime_conf_fn, regime_stab_fn
