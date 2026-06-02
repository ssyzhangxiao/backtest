"""
回测主配置（BacktestConfig）。

PyBroker 主引擎和自研验证引擎共用。
因子打分调仓模式：多因子综合得分决定持仓方向和仓位。

规则2：config.yaml 是单一数据源，BacktestConfig 必须与 yaml 完全同步。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import yaml

from .constants import INITIAL_CASH, DEFAULT_FACTOR_WEIGHTS
from .factors_config import FactorModuleConfig
from .adaptive_config import AdaptiveModuleConfig
from .multi_tf_config import MultiTFModuleConfig
from .position_config import PositionModuleConfig
from .stop_config import StopOptimizationConfig
from .instrument_config import InstrumentModuleConfig
from .validation_config import ValidationModuleConfig


@dataclass
class BacktestConfig:
    """回测配置（PyBroker 主引擎 + 自研验证引擎共用）。"""

    # ── 基础参数 ──
    initial_cash: float = INITIAL_CASH
    commission_rate: float = 0.0001
    slippage_rate: float = 0.0001

    # ── 样本内/外分割 ──
    in_sample_end: Optional[str] = None

    # ── 因子打分调仓 ──
    rebalance_days: int = 3
    factor_weights: Dict[str, float] = field(
        default_factory=lambda: DEFAULT_FACTOR_WEIGHTS.copy()
    )
    entry_threshold: float = 0.05
    score_scale: float = 1.0
    stop_loss_cooldown: int = 1

    # ── 风控 ──
    stop_loss_pct: float = 0.03
    max_position_pct: float = 0.15
    max_total_position_pct: float = 0.6
    min_position_pct: float = 0.0

    # ── 横截面标准化与排名叠加 ──
    use_cross_section: bool = True
    use_rank_score: bool = True
    use_rolling_ic: bool = True
    use_trend_filter: bool = False
    top_n_symbols: int = 5

    # ── PyBroker 相关 ──
    pybroker_bootstrap_samples: int = 10000
    pybroker_buy_delay: int = 1
    pybroker_sell_delay: int = 1

    # ── Walkforward 向前滚动分析 ──
    wf_train_bars: int = 252
    wf_test_bars: int = 63
    wf_step_bars: int = 21
    wf_train_ratio: float = 0.6
    wf_step_ratio: float = 0.1

    # ── 交叉验证 ──
    cross_validate: bool = False

    # ── 新模块配置（灰度开关，默认关闭） ──
    factors_config: FactorModuleConfig = field(default_factory=FactorModuleConfig)
    adaptive_config: AdaptiveModuleConfig = field(default_factory=AdaptiveModuleConfig)
    multi_tf_config: MultiTFModuleConfig = field(default_factory=MultiTFModuleConfig)
    position_config: PositionModuleConfig = field(default_factory=PositionModuleConfig)
    stop_optimization_config: StopOptimizationConfig = field(
        default_factory=StopOptimizationConfig
    )
    instrument_config: InstrumentModuleConfig = field(
        default_factory=InstrumentModuleConfig
    )
    validation_config: ValidationModuleConfig = field(
        default_factory=ValidationModuleConfig
    )

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "BacktestConfig":
        """
        从 YAML 文件加载配置。

        规则2：config.yaml 是单一数据源，BacktestConfig 必须与 yaml 完全同步。

        Args:
            path: YAML 文件路径

        Returns:
            BacktestConfig 实例
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        bt = raw.get("backtest", {})
        fw = raw.get("factor_weights", {})

        # 各模块配置委托给各自的from_yaml解析
        factors_cfg = FactorModuleConfig.from_yaml(raw)
        adaptive_cfg = AdaptiveModuleConfig.from_yaml(raw)
        multi_tf_cfg = MultiTFModuleConfig.from_yaml(raw)
        position_cfg = PositionModuleConfig.from_yaml(raw)
        stop_cfg = StopOptimizationConfig.from_yaml(raw)
        instrument_cfg = InstrumentModuleConfig.from_yaml(raw)
        validation_cfg = ValidationModuleConfig.from_yaml(raw)

        return cls(
            initial_cash=float(bt.get("initial_capital", INITIAL_CASH)),
            commission_rate=float(bt.get("commission", 0.0001)),
            slippage_rate=float(bt.get("slippage", 0.0001)),
            rebalance_days=int(bt.get("rebalance_freq", 3)),
            factor_weights=fw if fw else DEFAULT_FACTOR_WEIGHTS.copy(),
            entry_threshold=float(bt.get("entry_threshold", 0.05)),
            stop_loss_pct=float(bt.get("stop_loss_pct", 0.03)),
            max_position_pct=float(bt.get("max_position_pct", 0.15)),
            max_total_position_pct=float(bt.get("max_total_position_pct", 0.6)),
            min_position_pct=float(bt.get("min_position_pct", 0.0)),
            use_cross_section=bool(bt.get("use_cross_section", True)),
            use_rank_score=bool(bt.get("use_rank_score", True)),
            use_rolling_ic=bool(bt.get("use_rolling_ic", True)),
            use_trend_filter=bool(bt.get("use_trend_filter", False)),
            top_n_symbols=int(bt.get("top_n_symbols", 5)),
            factors_config=factors_cfg,
            adaptive_config=adaptive_cfg,
            multi_tf_config=multi_tf_cfg,
            position_config=position_cfg,
            stop_optimization_config=stop_cfg,
            instrument_config=instrument_cfg,
            validation_config=validation_cfg,
        )
