"""
v3 环境感知回测运行器（手动分析用）。

从 core/market_regime/__init__.py 中提取，仅用于手动分析场景。
主回测流程已切换为因子打分调仓模式，不再使用此类。
"""

from typing import Dict, Any, List

from loguru import logger

from core.market_regime import MarketRegimeDetector, MarketRegime
from core.param_manager import V3RegimeParamManager


class V3RegimeAwareRunner:
    """
    v3 环境感知回测运行器。

    使用 v3 MarketRegimeDetector 进行环境识别，接口与 simple_regime.RegimeAwareRunner 兼容。
    """

    def __init__(self, detector=None, param_manager=None):
        self.detector = detector or MarketRegimeDetector()
        self.param_manager = param_manager or V3RegimeParamManager()
        self._is_fitted = False
        self._regime_history: List[Dict[str, Any]] = []

    def detect_regime_series(self, df):
        """对整个 DataFrame 逐行识别市场环境。"""
        required_cols = {"date", "close", "high", "low"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"缺少必要列: {missing}")

        df_copy = df.copy()

        try:
            result = self.detector.fit_transform(df_copy, verbose=False)
            if "regime" not in result.columns:
                result["regime"] = "range_bound"
                result["regime_confidence"] = 0.5
                result["regime_stability"] = 1.0
            self._is_fitted = True
            return result
        except Exception as e:
            logger.error(f"v3 环境识别失败: {e}，使用默认值")
            df_copy["regime"] = "range_bound"
            df_copy["regime_confidence"] = 0.5
            df_copy["regime_stability"] = 1.0
            return df_copy

    def get_regime_distribution(self, df) -> Dict[str, float]:
        """计算市场环境分布占比。"""
        if "regime" not in df.columns:
            df = self.detect_regime_series(df)

        total = len(df)
        if total == 0:
            return {}

        dist = {}
        for regime in df["regime"].unique():
            count = (df["regime"] == regime).sum()
            dist[str(regime)] = round(count / total, 4)

        all_regimes = [
            "trend_up", "trend_down", "range_bound", "high_volatility",
            "low_volatility",
        ]
        for r in all_regimes:
            if r not in dist:
                dist[r] = 0.0

        return dist

    def run_with_regime_switch(self, runner, df, strategy_names: List[str],
                               start_date: str, end_date: str) -> Dict[str, Any]:
        """执行环境感知的回测。"""
        regime_df = self.detect_regime_series(df)
        dist = self.get_regime_distribution(regime_df)
        logger.info(f"v3 环境分布: {dist}")

        self._regime_history = []
        for regime_name, ratio in dist.items():
            if ratio > 0:
                self._regime_history.append({"regime": regime_name, "ratio": ratio})

        custom_params = {}
        for sname in strategy_names:
            params = self.param_manager.get_params(
                MarketRegime.RANGE_BOUND, sname, confidence=1.0,
            )
            if params:
                custom_params[sname] = params

        try:
            result = runner.run(start_date, end_date, custom_params=custom_params if custom_params else None)
        except Exception as e:
            logger.error(f"回测执行失败: {e}")
            result = {"error": str(e)}

        return result
