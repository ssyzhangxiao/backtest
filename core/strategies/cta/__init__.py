"""
单品种 CTA 策略模块。

按品种独立运行的时序策略，与横截面多策略打分体系解耦。

设计原则：
  - 每个品种独立计算信号，**不做**横截面标准化
  - 所有策略的退出逻辑已移至 CTAExecutorBuilder
  - 策略仅负责三层信号计算 + market_state 标记
  - 策略实例内部按品种维护状态

用法::

    from core.strategies.cta import get_cta_strategy

    strategy = get_cta_strategy("momentum_ma", {"fast_ma": 10, "slow_ma": 30})
    signal = strategy.compute_signal("SHFE.AU", close, high, low, volume)

旧名别名仍可用（向后兼容）:
  - "simple_trend" → momentum_ma
  - "state_aware_trend" → tsi_garch
"""

# 导入具体策略以触发注册
from core.strategies.cta.state_aware_trend import MomentumMAStrategy, TSIGarchStrategy  # noqa: F401
from core.strategies.cta.donchian_breakout import DonchianBreakoutStrategy  # noqa: F401
from core.strategies.cta.carry_strategy import CarryStrategy  # noqa: F401
from core.strategies.cta.vol_mean_reversion import VolMeanReversionStrategy  # noqa: F401
from core.strategies.cta.pair_trading import PairTradingStrategy  # noqa: F401
from core.strategies.cta.oi_strategy import OISignalStrategy  # noqa: F401  # 方向四 P1

from core.strategies.cta.base import CTABaseStrategy
from core.strategies.cta.registry import CTA_STRATEGY_REGISTRY, get_cta_strategy

__all__ = [
    "CTABaseStrategy",
    "CTA_STRATEGY_REGISTRY",
    "get_cta_strategy",
    # 新名
    "TSIGarchStrategy",
    "MomentumMAStrategy",
    "DonchianBreakoutStrategy",
    "CarryStrategy",
    "VolMeanReversionStrategy",
    "PairTradingStrategy",
    "OISignalStrategy",
    # 旧名（别名，兼容）
    "StateAwareTrendStrategy",  # → TSIGarchStrategy
    "SimpleTrendStrategy",  # → MomentumMAStrategy
]

# 旧名导出（型别名，IDE 自动补全用）
StateAwareTrendStrategy = TSIGarchStrategy
SimpleTrendStrategy = MomentumMAStrategy
