"""
v3 环境感知参数管理器。

从 core/market_regime/__init__.py 中提取，作为独立模块使用。
当前系统已切换为因子打分调仓模式，本模块仅用于辅助分析和手动参数调优。
"""

from typing import Dict, Any

import pandas as pd

from core.market_regime import MarketRegime

REGIME_TO_LEGACY: Dict[str, str] = {
    "trend_up": "trend_up",
    "trend_down": "trend_down",
    "range_bound": "range_bound",
    "high_volatility": "high_volatility",
    "low_volatility": "high_volatility",
    "breakout": "trend_up",
    "exhaustion_bull": "range_bound",
    "exhaustion_bear": "range_bound",
}


class V3RegimeParamManager:
    """
    v3 环境感知参数管理器。

    支持5类环境（trend_up, trend_down, range_bound, high_volatility, low_volatility），
    每类有独立的策略参数映射。
    向下兼容：通过 REGIME_TO_LEGACY 映射到原有4类环境的参数。
    """

    def __init__(self, legacy_manager=None):
        self._regime_params: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._default_params: Dict[str, Dict[str, Any]] = {}
        self._legacy = legacy_manager
        self._init_defaults()

    def _init_defaults(self):
        """初始化5类环境的策略参数。"""
        base_params = {
            "ts_momentum": {"window": 20, "position_size": 0.2},
            "roll_yield": {"lookback": 20, "entry_threshold": 2.0, "exit_threshold": 0.5, "position_size": 0.2},
            "alpha019": {"short_window": 7, "long_window": 250, "position_size": 0.2},
            "alpha032": {"ma_window": 7, "corr_window": 230, "position_size": 0.2},
        }

        self._default_params = dict(base_params)

        self._regime_params = {
            "trend_up": {
                "ts_momentum": {"window": 20, "position_size": 0.25},
                "roll_yield": {"lookback": 20, "entry_threshold": 2.0, "position_size": 0.15},
                "alpha019": {"short_window": 7, "long_window": 250, "position_size": 0.2},
                "alpha032": {"ma_window": 7, "corr_window": 230, "position_size": 0.15},
            },
            "trend_down": {
                "ts_momentum": {"window": 15, "position_size": 0.2},
                "roll_yield": {"lookback": 30, "entry_threshold": 2.5, "position_size": 0.1},
                "alpha019": {"short_window": 7, "long_window": 250, "position_size": 0.15},
                "alpha032": {"ma_window": 7, "corr_window": 230, "position_size": 0.1},
            },
            "range_bound": {
                "ts_momentum": {"window": 40, "position_size": 0.1},
                "roll_yield": {"lookback": 20, "entry_threshold": 1.5, "position_size": 0.25},
                "alpha019": {"short_window": 7, "long_window": 250, "position_size": 0.2},
                "alpha032": {"ma_window": 7, "corr_window": 230, "position_size": 0.2},
            },
            "high_volatility": {
                "ts_momentum": {"window": 30, "position_size": 0.1},
                "roll_yield": {"lookback": 10, "entry_threshold": 3.0, "position_size": 0.15},
                "alpha019": {"short_window": 7, "long_window": 250, "position_size": 0.1},
                "alpha032": {"ma_window": 7, "corr_window": 230, "position_size": 0.15},
            },
            "low_volatility": {
                "ts_momentum": {"window": 40, "position_size": 0.1},
                "roll_yield": {"lookback": 30, "entry_threshold": 1.5, "position_size": 0.2},
                "alpha019": {"short_window": 7, "long_window": 250, "position_size": 0.15},
                "alpha032": {"ma_window": 7, "corr_window": 230, "position_size": 0.1},
            },
        }

    def get_params(self, regime, strategy_name: str, confidence: float = 1.0) -> Dict[str, Any]:
        """获取指定环境下某策略的参数。"""
        if confidence < 0.5:
            return dict(self._default_params.get(strategy_name, {}))

        regime_value = regime.value if isinstance(regime, MarketRegime) else str(regime)
        regime_params = self._regime_params.get(regime_value)
        if regime_params is None:
            mapped = REGIME_TO_LEGACY.get(regime_value, regime_value)
            regime_params = self._regime_params.get(mapped)

        if regime_params:
            strategy_params = regime_params.get(strategy_name)
            if strategy_params:
                return dict(strategy_params)

        return dict(self._default_params.get(strategy_name, {}))

    def get_regime_weight(self, regime) -> float:
        """获取某环境下的组合权重。"""
        weights = {
            "trend_up": 1.0, "trend_down": 0.8, "range_bound": 1.0,
            "high_volatility": 0.6, "low_volatility": 0.7,
        }
        regime_value = regime.value if isinstance(regime, MarketRegime) else str(regime)
        if regime_value not in weights:
            mapped = REGIME_TO_LEGACY.get(regime_value, regime_value)
            return weights.get(mapped, 1.0)
        return weights[regime_value]

    def get_params_comparison_table(self) -> pd.DataFrame:
        """获取各环境参数对比表。"""
        strategies = set()
        for regime_params in self._regime_params.values():
            strategies.update(regime_params.keys())

        all_regimes = [
            "trend_up", "trend_down", "range_bound", "high_volatility",
            "low_volatility",
        ]

        rows = []
        for sname in sorted(strategies):
            row: Dict[str, Any] = {"strategy": sname}
            for regime_name in all_regimes:
                regime_params = self._regime_params.get(regime_name, {})
                strategy_params = regime_params.get(sname)
                row[regime_name] = str(strategy_params) if strategy_params else "-"
            rows.append(row)
        return pd.DataFrame(rows)
