"""
策略库管理系统。

为每种市场环境匹配策略，维护策略性能档案，
支持动态扩展和参数调整。

核心功能:
  - 策略注册与发现
  - 策略-环境映射
  - 策略性能档案
  - 动态参数调整
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np

from core.market_regime import MarketRegime


@dataclass
class StrategyProfile:
    """策略性能档案。"""
    name: str
    description: str = ""
    strategy_class_name: str = ""
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_ranges: Dict[str, List[Any]] = field(default_factory=dict)

    # 各环境下的历史表现
    performance_by_regime: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # 适用环境
    suitable_regimes: List[MarketRegime] = field(default_factory=list)

    # 风控参数
    max_position_pct: float = 0.2
    stop_loss_pct: float = 0.05
    trailing_stop_pct: float = 0.03
    time_stop_days: int = 15

    # 新增：绩效指标用于综合评分（任务要求：支持Sharpe+drawdown+turnover）
    max_drawdown: float = 0.2
    avg_turnover: float = 0.5

    # 统计
    total_backtests: int = 0
    last_updated: str = ""

    def update_performance(self, regime: MarketRegime, metrics: Dict[str, float]):
        """更新某环境下的性能数据。"""
        self.performance_by_regime[regime.value] = metrics
        self.total_backtests += 1
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_performance(self, regime: MarketRegime) -> Dict[str, float]:
        """获取某环境下的性能数据。"""
        return self.performance_by_regime.get(regime.value, {})

    def get_sharpe(self, regime: MarketRegime) -> float:
        """获取某环境下的Sharpe比率。"""
        perf = self.get_performance(regime)
        return perf.get("sharpe", 0.0)


class StrategyLibrary:
    """
    策略库管理器。

    管理所有可用策略及其与市场环境的映射关系。
    """

    def __init__(self):
        self._profiles: Dict[str, StrategyProfile] = {}
        self._regime_mapping: Dict[MarketRegime, List[str]] = {
            regime: [] for regime in MarketRegime
        }
        self._init_default_library()

    def _init_default_library(self):
        """初始化默认策略库。"""
        # 双均线趋势跟随
        self.register(StrategyProfile(
            name="dual_ma",
            description="双均线趋势跟随策略。短期均线上穿长期均线做多，下穿做空。",
            strategy_class_name="DualMAStrategy",
            default_params={
                "short_ma": 5, "long_ma": 20, "adx_threshold": 30.0,
                "position_size": 0.2, "trailing_stop_pct": 0.03, "time_stop_days": 15,
            },
            param_ranges={
                "short_ma": [3, 5, 8, 10],
                "long_ma": [15, 20, 30, 40],
                "adx_threshold": [20.0, 25.0, 30.0, 35.0],
            },
            suitable_regimes=[MarketRegime.TREND_UP, MarketRegime.TREND_DOWN, MarketRegime.LOW_VOLATILITY, MarketRegime.BREAKOUT],
            max_position_pct=0.2,
            stop_loss_pct=0.05,
            trailing_stop_pct=0.03,
        ))

        # RSI反转
        self.register(StrategyProfile(
            name="rsi",
            description="RSI反转策略。超卖做多，超买做空，仅在震荡市开仓。",
            strategy_class_name="RSIStrategy",
            default_params={
                "rsi_period": 14, "oversold": 30.0, "overbought": 70.0,
                "adx_threshold": 25.0, "position_size": 0.2,
            },
            param_ranges={
                "rsi_period": [10, 14, 20],
                "oversold": [20.0, 25.0, 30.0],
                "overbought": [70.0, 75.0, 80.0],
            },
            suitable_regimes=[MarketRegime.RANGE_BOUND, MarketRegime.EXHAUSTION_BULL, MarketRegime.EXHAUSTION_BEAR],
            max_position_pct=0.15,
            stop_loss_pct=0.03,
            trailing_stop_pct=0.02,
        ))

        # 期限结构套利
        self.register(StrategyProfile(
            name="term_structure",
            description="期限结构套利策略。基于价格偏离长期均值的均值回归。",
            strategy_class_name="TermStructureStrategy",
            default_params={
                "lookback": 20, "entry_threshold": 8.0, "exit_threshold": 0.5,
                "position_size": 0.2, "trailing_stop_pct": 0.05,
            },
            param_ranges={
                "lookback": [10, 20, 30],
                "entry_threshold": [6.0, 8.0, 10.0],
                "exit_threshold": [0.3, 0.5, 0.8],
            },
            suitable_regimes=[MarketRegime.RANGE_BOUND, MarketRegime.HIGH_VOLATILITY, MarketRegime.EXHAUSTION_BULL],
            max_position_pct=0.2,
            stop_loss_pct=0.05,
            trailing_stop_pct=0.05,
        ))

        # 波动率突破
        self.register(StrategyProfile(
            name="vol_breakout",
            description="波动率突破策略。基于ATR构建动态通道，突破上轨做多，突破下轨做空。",
            strategy_class_name="VolatilityBreakoutStrategy",
            default_params={
                "atr_period": 26, "band_period": 30, "atr_multiplier": 2.0,
                "position_size": 0.2, "trailing_stop_atr_mult": 3.0,
            },
            param_ranges={
                "atr_period": [14, 20, 26],
                "band_period": [20, 30, 40],
                "atr_multiplier": [1.5, 2.0, 2.5],
            },
            suitable_regimes=[MarketRegime.TREND_UP, MarketRegime.TREND_DOWN, MarketRegime.LOW_VOLATILITY, MarketRegime.BREAKOUT],
            max_position_pct=0.2,
            stop_loss_pct=0.04,
            trailing_stop_pct=0.03,
        ))

        # 跨期套利
        self.register(StrategyProfile(
            name="spread",
            description="跨期套利策略。利用近远月价差变化获利。",
            strategy_class_name="SpreadStrategy",
            default_params={
                "spread_ma_period": 20, "spread_entry_threshold": 2.0,
                "position_size": 0.15,
            },
            param_ranges={
                "spread_ma_period": [10, 20, 30],
                "spread_entry_threshold": [1.5, 2.0, 2.5],
            },
            suitable_regimes=[MarketRegime.RANGE_BOUND, MarketRegime.LOW_VOLATILITY],
            max_position_pct=0.15,
            stop_loss_pct=0.03,
        ))

    def register(self, profile: StrategyProfile):
        """注册策略到库中。"""
        self._profiles[profile.name] = profile
        for regime in profile.suitable_regimes:
            if profile.name not in self._regime_mapping[regime]:
                self._regime_mapping[regime].append(profile.name)

    def get_profile(self, name: str) -> Optional[StrategyProfile]:
        """获取策略档案。"""
        return self._profiles.get(name)

    def get_strategies_for_regime(self, regime: MarketRegime) -> List[StrategyProfile]:
        """获取适用于某市场环境的策略列表。"""
        names = self._regime_mapping.get(regime, [])
        return [self._profiles[n] for n in names if n in self._profiles]

    def get_best_strategy(self, regime: MarketRegime, metric: str = "sharpe") -> Optional[StrategyProfile]:
        """获取某环境下表现最好的策略。"""
        strategies = self.get_strategies_for_regime(regime)
        if not strategies:
            return None
        best = None
        best_val = float("-inf")
        for s in strategies:
            perf = s.get_performance(regime)
            val = perf.get(metric, float("-inf"))
            if val > best_val:
                best_val = val
                best = s
        return best

    def list_all(self) -> List[StrategyProfile]:
        """列出所有策略。"""
        return list(self._profiles.values())

    def update_performance(self, strategy_name: str, regime: MarketRegime,
                           metrics: Dict[str, float]):
        """更新策略性能数据。"""
        profile = self._profiles.get(strategy_name)
        if profile:
            profile.update_performance(regime, metrics)

    def get_regime_mapping(self) -> Dict[str, List[str]]:
        """获取环境-策略映射。"""
        return {regime.value: names for regime, names in self._regime_mapping.items()}

    def summary(self) -> pd.DataFrame:
        """策略库概览。"""
        rows = []
        for name, profile in self._profiles.items():
            regimes_str = ", ".join(r.value for r in profile.suitable_regimes)
            perf_count = len(profile.performance_by_regime)
            rows.append({
                "策略": name,
                "描述": profile.description[:30] + "...",
                "适用环境": regimes_str,
                "性能记录数": perf_count,
                "最后更新": profile.last_updated or "N/A",
            })
        return pd.DataFrame(rows)
