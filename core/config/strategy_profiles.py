"""
策略档案与策略库（2026-06-07 从 core/strategy_registry.py 迁移）。

原 core/strategy_registry.py 已删除。
- 原因：旧模块名暗示"注册策略类"，但内部已不持有任何类引用（_STRATEGY_CLASS_MAP
  / create_strategy / register_strategy_class 等已删除），是纯元数据管理。
- 新位置：core/config/strategy_profiles.py，与 BacktestConfig 等其他配置数据并列，
  准确反映"策略档案是配置数据"这一本质。
- 公开 API 完全保持兼容：StrategyProfile、StrategyLibrary、SUB_STRATEGY_NAMES、
  STRATEGY_NAMES 仍可从 `from core.config.strategy_profiles import ...` 导入。
- 子策略信号统一由 core.factors.alpha_futures.sub_strategy_aggregator 提供（路径A），
  StrategyProfile 仅作为元数据供参数搜索、UI 展示、文档生成使用。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


# 5 子策略名称清单（权威来源，UI/Registry 共用）
SUB_STRATEGY_NAMES: List[str] = [
    "trend",
    "term_structure",
    "mean_reversion",
    "vol_breakout",
    "composite_resonance",
]


@dataclass
class StrategyProfile:
    """策略档案：参数、搜索空间、因子权重。"""

    name: str
    description: str = ""
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_ranges: Dict[str, List[Any]] = field(default_factory=dict)
    factor_weights: Dict[str, float] = field(default_factory=dict)
    max_position_pct: float = 0.2
    stop_loss_pct: float = 0.05
    enabled: bool = True
    total_backtests: int = 0
    last_updated: str = ""


class StrategyLibrary:
    """
    策略库：管理5子策略档案（参数、搜索空间、因子权重）。

    2026-06-07：从 core/strategy_registry.py 迁移至本模块。
    公开 API 完全兼容：register / get_profile / get_weights / list_all /
    update_default_params / export_params_to_yaml。
    """

    def __init__(self):
        self._profiles: Dict[str, StrategyProfile] = {}
        self._init_default_library()

    def _init_default_library(self):
        """初始化5子策略档案（仅元数据，无类引用）。"""
        self.register(StrategyProfile(
            name="trend",
            description="趋势策略。基于动量和均线趋势跟踪。",
            default_params={"window": 20, "position_size": 0.2},
            param_ranges={"window": [5, 10, 15, 20, 30, 40, 60]},
            factor_weights={"trend": 0.2},
            max_position_pct=0.2,
            stop_loss_pct=0.05,
        ))
        self.register(StrategyProfile(
            name="term_structure",
            description="期限结构策略。基于展期收益率和升贴水价差。",
            default_params={"lookback": 20, "entry_threshold": 1.5, "position_size": 0.2},
            param_ranges={
                "lookback": [5, 10, 15, 20, 30, 40, 60],
                "entry_threshold": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            },
            factor_weights={"term_structure": 0.2},
            max_position_pct=0.2,
            stop_loss_pct=0.05,
        ))
        self.register(StrategyProfile(
            name="mean_reversion",
            description="均值回归策略。基于价格偏离均值后的回归。",
            default_params={
                "short_window": 5, "long_window": 250, "position_size": 0.2,
            },
            param_ranges={
                "short_window": [3, 5, 7, 10, 14],
                "long_window": [120, 180, 250, 300, 360],
            },
            factor_weights={"mean_reversion": 0.2},
            max_position_pct=0.2,
            stop_loss_pct=0.04,
        ))
        self.register(StrategyProfile(
            name="vol_breakout",
            description="波动率突破策略。基于波动率扩张和收缩的突破信号。",
            default_params={"ma_window": 5, "corr_window": 200, "position_size": 0.2},
            param_ranges={
                "ma_window": [3, 5, 7, 10, 14],
                "corr_window": [120, 180, 230, 300, 360],
            },
            factor_weights={"vol_breakout": 0.2},
            max_position_pct=0.2,
            stop_loss_pct=0.04,
        ))
        self.register(StrategyProfile(
            name="composite_resonance",
            description="复合共振策略。多因子共振确认，提高信号可靠性。",
            default_params={"position_size": 0.2},
            param_ranges={},
            factor_weights={"composite_resonance": 0.2},
            max_position_pct=0.2,
            stop_loss_pct=0.05,
        ))
        # 5子策略横截面组合（组合模式由 FactorScoringEngine + PortfolioManager 负责）
        self.register(StrategyProfile(
            name="cross_sectional",
            description="5子策略横截面打分组合。横截面标准化+排名叠加，动态仓位分配。",
            default_params={
                "entry_threshold": 0.05,
                "top_n": 5,
                "rebalance_days": 3,
                "use_cross_section": True,
                "use_rank_score": True,
                "use_rolling_ic": True,
                "merge_method": "equal_weight",
            },
            param_ranges={
                "entry_threshold": [0.02, 0.05, 0.10, 0.15],
                "top_n": [3, 5, 8, 10],
                "rebalance_days": [1, 3, 5, 10],
            },
            factor_weights={
                "trend": 0.2,
                "term_structure": 0.2,
                "mean_reversion": 0.2,
                "vol_breakout": 0.2,
                "composite_resonance": 0.2,
            },
            max_position_pct=0.2,
            stop_loss_pct=0.05,
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
        """更新策略默认参数。"""
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


# 策略名称列表（替代旧 STRATEGY_REGISTRY，UI/Registry 统一使用）
STRATEGY_NAMES = SUB_STRATEGY_NAMES + ["cross_sectional"]


__all__ = [
    "StrategyProfile",
    "StrategyLibrary",
    "SUB_STRATEGY_NAMES",
    "STRATEGY_NAMES",
]
