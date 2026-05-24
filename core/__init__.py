"""
组合交易策略系统 - 核心模块。

模块结构:
  - data_loader: 数据加载与展期处理
  - market_regime: 市场环境分类系统
  - strategy_library: 策略库管理
  - strategies: 策略实现（dual_ma, rsi, vol_breakout, term_structure, spread）
  - engine: 回测引擎与策略切换
  - performance: 绩效评估与预警
  - environment: 环境适配器（兼容旧接口）
  - rollover: 展期管理
  - portfolio: 组合管理
  - risk_manager: 风控管理
  - optimizer: 参数优化
"""

from .data_loader import DataLoader
from .environment import EnvironmentAdapter
from .strategies import BaseStrategy, DualMAStrategy, RSIStrategy, SpreadStrategy
from .strategies import TermStructureStrategy, VolatilityBreakoutStrategy
from .strategies import STRATEGY_REGISTRY, create_strategy
from .rollover import RolloverManager
from .portfolio import PortfolioManager
from .risk_manager import RiskManager
from .optimizer import ParameterOptimizer
from .market_regime import MarketRegimeDetector, MarketRegime, RegimeConfig, RegimeResult
from .strategy_library import StrategyLibrary, StrategyProfile
from .engine import (
    BacktestRunner, BacktestConfig,
    StrategySwitchEngine, SwitchConfig,
    StrategyResult, PortfolioResult,
)
from .performance import PerformanceEvaluator, PerformanceMonitor, PerformanceConfig

__all__ = [
    # 数据
    "DataLoader",
    # 环境
    "EnvironmentAdapter",
    "MarketRegimeDetector",
    "MarketRegime",
    "RegimeConfig",
    "RegimeResult",
    # 策略
    "BaseStrategy",
    "DualMAStrategy",
    "RSIStrategy",
    "SpreadStrategy",
    "TermStructureStrategy",
    "VolatilityBreakoutStrategy",
    "STRATEGY_REGISTRY",
    "create_strategy",
    # 策略库
    "StrategyLibrary",
    "StrategyProfile",
    # 引擎
    "BacktestRunner",
    "BacktestConfig",
    "StrategySwitchEngine",
    "SwitchConfig",
    "StrategyResult",
    "PortfolioResult",
    # 绩效
    "PerformanceEvaluator",
    "PerformanceMonitor",
    "PerformanceConfig",
    # 管理
    "RolloverManager",
    "PortfolioManager",
    "RiskManager",
    "ParameterOptimizer",
]
