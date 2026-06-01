"""
CTA + Alpha101 多因子量化回测系统 - 核心模块。

模块结构:
  - data_loader: 数据加载与展期处理
  - market_regime: 市场环境分类系统（辅助分析，不参与策略决策）
  - strategy_registry: 策略注册表（策略类映射 + 参数档案 + 因子权重）
  - strategies: 四因子策略（ts_momentum, roll_yield, alpha019, alpha032）
  - engine: 回测引擎 + 因子打分调仓引擎
  - performance: 绩效评估与预警
  - rollover: 展期管理
  - portfolio: 组合管理
  - risk_controller: 风控管理
  - optimizer: 参数优化
"""

from .data_loader import DataLoader
from .strategies import (
    BaseStrategy,
    TSMomentumStrategy,
    RollYieldStrategy,
    Alpha019Strategy,
    Alpha032Strategy,
    STRATEGY_REGISTRY,
    create_strategy,
)
from .rollover import RolloverManager
from .portfolio import PortfolioManager
from .risk_controller import RiskController, RiskConfig
from .optimizer import ParameterOptimizer
from .market_regime import MarketRegimeDetector, MarketRegime, RegimeConfig, RegimeResult
from .strategy_registry import StrategyLibrary, StrategyProfile, register
from .engine import (
    BacktestRunner,
    BacktestConfig,
    FactorScoringEngine,
    StrategySwitchEngine,
    ScoringConfig,
    RebalanceDecision,
    RebalanceReason,
    StrategyResult,
    PortfolioResult,
)
from .performance import PerformanceEvaluator, PerformanceMonitor, PerformanceConfig

__all__ = [
    "DataLoader",
    "MarketRegimeDetector",
    "MarketRegime",
    "RegimeConfig",
    "RegimeResult",
    "BaseStrategy",
    "TSMomentumStrategy",
    "RollYieldStrategy",
    "Alpha019Strategy",
    "Alpha032Strategy",
    "STRATEGY_REGISTRY",
    "create_strategy",
    "StrategyLibrary",
    "StrategyProfile",
    "register",
    "BacktestRunner",
    "BacktestConfig",
    "FactorScoringEngine",
    "StrategySwitchEngine",
    "ScoringConfig",
    "RebalanceDecision",
    "RebalanceReason",
    "StrategyResult",
    "PortfolioResult",
    "PerformanceEvaluator",
    "PerformanceMonitor",
    "PerformanceConfig",
    "RolloverManager",
    "PortfolioManager",
    "RiskController",
    "RiskConfig",
    "ParameterOptimizer",
]
