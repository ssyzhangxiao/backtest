"""
策略注册表。

合并原 core/strategies/registry.py（策略类映射）和
core/strategy_library/（策略档案），统一管理策略注册、参数和因子权重。

核心功能:
  - 策略类注册与发现（装饰器 + 字典）
  - 策略参数档案（默认参数、参数搜索空间）
  - 因子权重查询
  - 动态参数调整
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from core.strategies.base import BaseStrategy
from core.strategies.ts_momentum import TSMomentumStrategy
from core.strategies.roll_yield import RollYieldStrategy
from core.strategies.alpha019 import Alpha019Strategy
from core.strategies.alpha032 import Alpha032Strategy


@dataclass
class StrategyProfile:
    """策略档案：参数、搜索空间、因子权重。"""

    name: str
    description: str = ""
    strategy_class_name: str = ""
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_ranges: Dict[str, List[Any]] = field(default_factory=dict)
    factor_weights: Dict[str, float] = field(default_factory=dict)
    max_position_pct: float = 0.2
    stop_loss_pct: float = 0.05
    enabled: bool = True
    total_backtests: int = 0
    last_updated: str = ""


_STRATEGY_CLASS_MAP: Dict[str, type] = {
    "ts_momentum": TSMomentumStrategy,
    "roll_yield": RollYieldStrategy,
    "alpha019": Alpha019Strategy,
    "alpha032": Alpha032Strategy,
}


def get_strategy_class(name: str) -> type:
    """根据策略名称获取策略类。"""
    if name not in _STRATEGY_CLASS_MAP:
        available = ", ".join(_STRATEGY_CLASS_MAP.keys())
        raise ValueError(f"未知策略 '{name}'，可用策略: {available}")
    return _STRATEGY_CLASS_MAP[name]


def create_strategy(name: str, **kwargs) -> BaseStrategy:
    """根据策略名称创建策略实例。"""
    return get_strategy_class(name)(**kwargs)


def register(name: str, factor_weights: Optional[Dict[str, float]] = None):
    """
    装饰器：注册策略类到全局映射表，并关联因子权重。

    用法:
        @register("my_strategy", factor_weights={"my_strategy": 0.5})
        class MyStrategy(BaseStrategy):
            ...
    """
    def decorator(cls):
        _STRATEGY_CLASS_MAP[name] = cls
        return cls
    return decorator


class StrategyLibrary:
    """
    策略库：管理策略档案（参数、搜索空间、因子权重）。

    保留原 StrategyLibrary 接口，移除环境映射逻辑。
    """

    def __init__(self):
        self._profiles: Dict[str, StrategyProfile] = {}
        self._init_default_library()

    def _init_default_library(self):
        """初始化默认策略档案：CTA因子 + Alpha101因子。"""
        self.register(StrategyProfile(
            name="ts_momentum",
            description="时间序列动量策略。20日累计收益率，正收益做多、负收益做空。",
            strategy_class_name="TSMomentumStrategy",
            default_params={"window": 20, "position_size": 0.2},
            param_ranges={"window": [10, 15, 20, 30, 40], "position_size": [0.1, 0.15, 0.2, 0.25]},
            factor_weights={"ts_momentum": 0.25},
            max_position_pct=0.2,
            stop_loss_pct=0.05,
        ))
        self.register(StrategyProfile(
            name="roll_yield",
            description="期限结构/展期收益率策略。升贴水价差回归，升水做空、贴水做多。",
            strategy_class_name="RollYieldStrategy",
            default_params={"lookback": 20, "entry_threshold": 2.0, "exit_threshold": 0.5, "position_size": 0.2},
            param_ranges={"lookback": [10, 20, 30, 60], "entry_threshold": [1.0, 1.5, 2.0, 3.0, 4.0]},
            factor_weights={"roll_yield": 0.25},
            max_position_pct=0.2,
            stop_loss_pct=0.05,
        ))
        self.register(StrategyProfile(
            name="alpha019",
            description="Alpha#019因子。短期反转+长期动量排名，7日反转×250日动量。",
            strategy_class_name="Alpha019Strategy",
            default_params={"short_window": 7, "long_window": 250, "position_size": 0.2},
            param_ranges={"short_window": [5, 7, 10, 14], "long_window": [120, 180, 250, 360]},
            factor_weights={"alpha019": 0.25},
            max_position_pct=0.2,
            stop_loss_pct=0.04,
        ))
        self.register(StrategyProfile(
            name="alpha032",
            description="Alpha#032因子。7日均线偏离+VWAP-收盘价相关性，230天滚动窗口。",
            strategy_class_name="Alpha032Strategy",
            default_params={"ma_window": 7, "corr_window": 230, "position_size": 0.2},
            param_ranges={"ma_window": [5, 7, 10, 14], "corr_window": [120, 150, 180, 230, 300]},
            factor_weights={"alpha032": 0.25},
            max_position_pct=0.2,
            stop_loss_pct=0.04,
        ))

    def register(self, profile: StrategyProfile):
        """注册策略档案。"""
        self._profiles[profile.name] = profile

    def get_profile(self, name: str) -> Optional[StrategyProfile]:
        """获取策略档案。"""
        return self._profiles.get(name)

    def get_weights(self, name: str) -> Dict[str, float]:
        """获取策略的因子权重。"""
        profile = self._profiles.get(name)
        return profile.factor_weights if profile else {}

    def list_all(self, include_disabled: bool = False) -> List[StrategyProfile]:
        """列出所有策略。"""
        if include_disabled:
            return list(self._profiles.values())
        return [p for p in self._profiles.values() if p.enabled]

    def update_default_params(self, strategy_name: str, new_params: Dict[str, Any]) -> bool:
        """更新策略默认参数（仅更新提供的键）。"""
        profile = self._profiles.get(strategy_name)
        if profile is None:
            return False
        profile.default_params.update(new_params)
        profile.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return True

    def export_params_to_yaml(self, strategy_names: Optional[List[str]] = None) -> Dict:
        """导出策略默认参数为 YAML 兼容字典。"""
        names = strategy_names or list(self._profiles.keys())
        result = {}
        for name in names:
            profile = self._profiles.get(name)
            if profile is not None:
                result[name] = dict(profile.default_params)
        return result


STRATEGY_REGISTRY = _STRATEGY_CLASS_MAP
