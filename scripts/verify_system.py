"""系统验证脚本。

P0 整改（2026-06-07）：core/strategies/ 已整体删除，
create_strategy 接口已废弃。改为验证 StrategyLibrary + STRATEGY_NAMES
统一入口。
"""
from core.config import BacktestConfig, DEFAULT_FACTOR_WEIGHTS
from core.config.backtest_config import BacktestConfig as _BC  # noqa: F401
from core.engine.switch_engine import ScoringConfig
from core.risk_controller import RiskController
from core.data_provider import DataProvider
from core.data_loader import DataLoader
from core.config.strategy_profiles import StrategyLibrary, STRATEGY_NAMES
from core.factors.alpha_futures.config import AlphaFuturesConfig

print("OK 所有核心模块导入成功")

config = BacktestConfig()
assert config.rebalance_days == 3
assert not hasattr(config, "fusion_mode")
assert not hasattr(config, "strategy_weights")
assert not hasattr(config, "rebalance_frequency")
print("OK BacktestConfig 废弃字段已移除")

sc = ScoringConfig()
assert sc.factor_weights == DEFAULT_FACTOR_WEIGHTS
print("OK 因子权重统一为 config.DEFAULT_FACTOR_WEIGHTS")

# AlphaFuturesConfig 字段完整性
afc = AlphaFuturesConfig()
assert afc.gap_weight == 0.5
assert afc.gap_weight_window == 20
assert afc.momentum_orth_window > 0
print("OK AlphaFuturesConfig 关键字段完整 (gap_weight / gap_weight_window / momentum_orth_window)")

assert issubclass(DataLoader, DataProvider)
print("OK DataLoader 实现 DataProvider 接口")

rc = RiskController()
assert rc.check_stop_loss("RB", 100000, -3000, 3500, trading_day_index=10) is False
assert rc.check_stop_loss("RB", 100000, -6000, 3500, trading_day_index=10) is True
print("OK RiskController 风控逻辑正常")

lib = StrategyLibrary()
assert lib.get_profile("trend") is not None
assert lib.get_profile("term_structure") is not None
assert lib.get_profile("mean_reversion") is not None
assert lib.get_profile("vol_breakout") is not None
assert lib.get_profile("composite_resonance") is not None
# 验证档案内部结构（param_ranges 必须存在）
trend = lib.get_profile("trend")
assert hasattr(trend, "param_ranges") and len(trend.param_ranges) > 0
print("OK StrategyLibrary 档案结构完整 (param_ranges 非空)")

# P0 整改：不再验证 create_strategy（已删除）
# 改为验证 STRATEGY_NAMES 公共入口
assert "trend" in STRATEGY_NAMES
assert "cross_sectional" in STRATEGY_NAMES
print("OK STRATEGY_NAMES 公共入口正常")

# P0 整改：明确验证废弃模块已移除
import importlib
for removed in ("core.strategies", "core.adaptive", "core.position", "core.instrument", "core.market_regime"):
    try:
        importlib.import_module(removed)
        raise AssertionError(f"{removed} 应已被移除，但可被导入")
    except (ImportError, ModuleNotFoundError):
        pass
print("OK 废弃模块已移除: core.strategies / core.adaptive / core.position / core.instrument / core.market_regime")

print()
print("SUCCESS 系统验证全部通过!")
