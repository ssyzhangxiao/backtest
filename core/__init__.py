"""
5子策略量化回测系统 - 核心模块。

模块结构:
  - data_loader: 数据加载与展期处理
  - engine: 回测引擎 + 因子打分调仓引擎 + 策略集成器
  - performance: 绩效评估与预警
  - rollover: 展期管理（仅执行包装与成本统计，标注信号由 DataLoader.rollover_flag 提供）
  - portfolio: 组合管理
  - risk_controller: 风控管理
  - optimizer: 参数优化

2026-06-07 整改：
  - 删除 core/strategies/（规则17 不重复造轮子，子策略信号统一由因子层路径A提供）
  - 删除 core/market_regime/ 兼容层（MarketRegimeDetector 已废弃，不再导出）
  - 删除 core/strategy_registry.py（迁移至 core/config/strategy_profiles.py）
"""

from .data_loader import DataLoader
from .rollover import RolloverManager
from .portfolio import PortfolioManager
from .risk_controller import RiskController, RiskConfig
from .optimizer import ParameterOptimizer
from .engine import (
    CrossValidator,
    FactorScoringEngine,
    ScoringConfig,
    RebalanceDecision,
    RebalanceReason,
)
from .performance import PerformanceEvaluator, PerformanceMonitor, PerformanceConfig

# 策略档案（2026-06-07 从 core.strategy_registry 迁移）
from .config.strategy_profiles import (
    StrategyLibrary,
    StrategyProfile,
    SUB_STRATEGY_NAMES,
    STRATEGY_NAMES,
)

__all__ = [
    "DataLoader",
    "STRATEGY_NAMES",
    "SUB_STRATEGY_NAMES",
    "StrategyLibrary",
    "StrategyProfile",
    "CrossValidator",
    "FactorScoringEngine",
    "ScoringConfig",
    "RebalanceDecision",
    "RebalanceReason",
    "PerformanceEvaluator",
    "PerformanceMonitor",
    "PerformanceConfig",
    "RolloverManager",
    "PortfolioManager",
    "RiskController",
    "RiskConfig",
    "ParameterOptimizer",
]
