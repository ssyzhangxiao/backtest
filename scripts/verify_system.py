"""系统验证脚本。"""
from core.config import BacktestConfig, DEFAULT_FACTOR_WEIGHTS
from core.engine.switch_engine import FactorScoringEngine, StrategySwitchEngine, ScoringConfig
from core.engine.backtest_runner import PyBrokerBacktestRunner
from core.engine.strategy_executor import RiskManagerAdapter
from core.engine.runner import BacktestRunner
from core.risk_controller import RiskController, RiskConfig
from core.data_provider import DataProvider
from core.data_loader import DataLoader
from core.param_manager import V3RegimeParamManager
from core.portfolio import PortfolioManager
from core.strategy_registry import StrategyLibrary, StrategyProfile, register, create_strategy
from core.market_regime import MarketRegimeDetector, MarketRegime
from core.risk_manager import RiskManager

print("OK 所有核心模块导入成功")

assert StrategySwitchEngine is FactorScoringEngine
print("OK StrategySwitchEngine 兼容别名正常")

assert RiskManager is RiskManagerAdapter
print("OK RiskManager 兼容别名指向 RiskManagerAdapter")

config = BacktestConfig()
assert config.rebalance_days == 3
assert not hasattr(config, "fusion_mode")
assert not hasattr(config, "strategy_weights")
assert not hasattr(config, "rebalance_frequency")
print("OK BacktestConfig 废弃字段已移除")

sc = ScoringConfig()
assert sc.factor_weights == DEFAULT_FACTOR_WEIGHTS
print("OK 因子权重统一为 config.DEFAULT_FACTOR_WEIGHTS")

assert issubclass(DataLoader, DataProvider)
print("OK DataLoader 实现 DataProvider 接口")

rc = RiskController()
assert rc.check_stop_loss("RB", 100000, -3000, 3500, trading_day_index=10) is False
assert rc.check_stop_loss("RB", 100000, -6000, 3500, trading_day_index=10) is True
print("OK RiskController 风控逻辑正常")

lib = StrategyLibrary()
assert lib.get_profile("ts_momentum") is not None
assert lib.get_profile("roll_yield") is not None
assert lib.get_profile("alpha019") is not None
assert lib.get_profile("alpha032") is not None
assert lib.get_weights("ts_momentum") == {"ts_momentum": 0.25}
print("OK StrategyLibrary 从 strategy_registry 正常加载")

strat = create_strategy("ts_momentum", window=20, position_size=0.2)
assert strat is not None
print("OK create_strategy 从 strategy_registry 正常工作")

pm = V3RegimeParamManager()
params = pm.get_params(MarketRegime.TREND_UP, "ts_momentum")
assert "window" in params
print("OK V3RegimeParamManager 独立模块正常")

print()
print("SUCCESS 系统验证全部通过!")
