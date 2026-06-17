"""
回测主配置（BacktestConfig）。

PyBroker 主引擎和自研验证引擎共用。
因子打分调仓模式：多因子综合得分决定持仓方向和仓位。

规则2：config.yaml 是单一数据源，BacktestConfig 必须与 yaml 完全同步。

P0/P1/P2 整改（2026-06-07）：
  - from_yaml / to_yaml 补全字段：top_n_symbols / weight_method / min_position_pct
  - P1-1：移除 DEFAULT_FACTOR_WEIGHTS 默认使用，YAML 未提供时记 warning 并设空字典
  - P1-3：stop_loss_pct 与 composite_stop.fixed_stop_pct 明确关系：
      BacktestConfig.stop_loss_pct 是「全局统一固定止损」语义，
      StopOptimizationConfig.composite_fixed_stop_pct 是「复合止损专用」语义。
      当 stop_optimization_config.enabled=True 时，复合止损覆盖全局固定止损。
  - P2-1：_convert_numpy_types 提取至 yaml_utils.convert_numpy_types
  - P2-2：所有字段添加 docstring，说明与规则9/13/15 的关系
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

from .constants import INITIAL_CASH
from .factors_config import FactorModuleConfig
from .stop_config import StopOptimizationConfig
from .validation_config import ValidationModuleConfig
from .yaml_utils import convert_numpy_types, dump_yaml, load_yaml
from .layered_config import load_env_overrides, merge_overrides

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回测配置（PyBroker 主引擎 + 自研验证引擎共用）。"""

    # ── 基础参数 ──
    initial_cash: float = INITIAL_CASH
    """初始资金（元）。规则2：与 config.yaml backtest.initial_capital 同步。"""

    commission_rate: float = 0.0001
    """手续费率（双边）。规则15：默认值适配商品期货。"""

    slippage_rate: float = 0.0001
    """滑点率（双边）。规则15：默认值适配主力合约。"""

    # ── 日期范围（从 config.yaml backtest 段读取） ──
    full_start: str = "2016-01-01"
    full_end: str = "2025-12-31"
    train_start: str = "2016-01-01"
    train_end: str = "2020-12-31"
    test_start: str = "2021-01-01"
    test_end: str = "2025-12-31"

    # ── 样本内/外分割 ──
    in_sample_end: Optional[str] = None
    out_sample_start: Optional[str] = None

    # ── 因子打分调仓 ──
    rebalance_days: int = 3
    """调仓周期（交易日）。规则9：与 backtest.rebalance_freq 同步。"""

    factor_weights: Dict[str, float] = field(default_factory=dict)
    """5子策略权重。P1-1 整改：YAML 未提供时为空字典并打 warning。"""

    entry_threshold: float = 0.05
    """开仓信号阈值。规则9：综合得分超过该阈值才开仓。"""

    score_scale: float = 1.0
    stop_loss_cooldown: int = 1
    """止损后冷却期（交易日）。"""

    # ── 风控 ──
    stop_loss_pct: float = 0.03
    """固定止损比例（全局默认）。规则13：与 composite_stop.fixed_stop_pct 关系见类注释。"""

    max_position_pct: float = 0.15
    """单品种最大持仓占比。"""

    max_total_position_pct: float = 0.6
    """总持仓上限。"""

    min_position_pct: float = 0.0
    """单品种最小持仓占比（P0 整改补全）。"""

    # ── 横截面标准化与排名叠加 ──
    use_cross_section: bool = True
    """是否横截面 Z-Score 标准化。"""

    use_rank_score: bool = True
    """是否使用排名叠加。"""

    use_rolling_ic: bool = True
    """是否使用滚动IC动态权重。"""

    use_trend_filter: bool = False
    """是否使用多时间框架趋势过滤（规则11，规划中）。"""

    top_n_symbols: int = 5
    """横截面选股数量。P0 整改补全：与 config.yaml backtest.top_n_symbols 同步。"""

    # ── 子策略体系 ──
    signal_merge_method: str = "equal_weight"
    """子策略信号合并方法：equal_weight / volatility_inverse / adaptive / majority_vote。"""

    use_sub_strategies: bool = True
    """是否启用 5 子策略体系。P0 整改补全：to_yaml 写入。"""

    use_new_factors: bool = True
    """是否启用新 24 因子引擎（替代旧 basic_factors）。P0 整改补全：to_yaml 写入。"""

    # ── 统一因子池配置（规则32，2026-06-14） ──
    use_signal_abstraction: bool = False
    """是否启用 SignalAbstractionLayer（统一因子池模式）。"""

    signal_mode: str = "cross_sectional"
    """信号模式：cross_sectional / cta / hybrid。"""

    cta_hybrid_weight: float = 0.5
    """混合模式下 CTA 信号权重（0~1），对应 cross_section_z 权重为 1 - cta_hybrid_weight。"""

    # ── 动态混合模式参数（方向二，2026-06-15） ──
    hybrid_blend_method: str = "linear"
    """混合模式合成方法：linear（线性加权）/ dynamic（XS 仓位缩放）。"""

    xs_position_base: float = 0.5
    """动态混合模式下，XS 强度=0 时的 CTA 仓位缩放下限（0~1）。"""

    xs_position_ceiling: float = 1.0
    """动态混合模式下，XS 强度=1 时的 CTA 仓位缩放上限（0~1）。"""

    xs_opposite_penalty: float = 0.5
    """动态混合模式下，CTA 与 XS 异号时的额外减仓系数（0~1）。"""

    weight_method: str = "risk_parity"
    """权重分配方法：equal_weight / risk_parity / score_weighted / top_n。P0 整改补全。"""

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
    """是否执行 PyBroker 与自研引擎并行交叉验证（规则26）。"""

    # ── 品种与策略（从 config.yaml 顶层读取） ──
    symbols: list = field(
        default_factory=lambda: ["SHFE.RB", "DCE.M", "CZCE.TA", "SHFE.CU", "CFFEX.IF"]
    )
    strategy_names: list = field(default_factory=list)

    # ── 输出与风控 ──
    output_dir: str = "output_validation"
    bankruptcy_threshold: float = 0.8

    # ── 新模块配置（灰度开关，默认关闭） ──
    factors_config: FactorModuleConfig = field(default_factory=FactorModuleConfig)
    stop_optimization_config: StopOptimizationConfig = field(
        default_factory=StopOptimizationConfig
    )
    validation_config: ValidationModuleConfig = field(
        default_factory=ValidationModuleConfig
    )

    @classmethod
    def from_yaml(
        cls,
        path: str = "config.yaml",
        overrides: Optional[Dict] = None,
    ) -> "BacktestConfig":
        """
        从 YAML 文件加载配置（支持分层覆盖）。

        规则2：config.yaml 是单一数据源，BacktestConfig 必须与 yaml 完全同步。
        规则23：分层配置 — 优先级 YAML < env vars < overrides。

        P0 整改：补充 top_n_symbols / weight_method / min_position_pct 字段读取。
        P1-1 整改：factor_weights 缺省时记 warning，使用空字典（不再用 DEFAULT_FACTOR_WEIGHTS）。
        规则23: 集成 LayeredConfigLoader，支持 env vars + runtime overrides。

        Args:
            path: YAML 文件路径
            overrides: 运行时覆盖字典，key 格式：
                - 顶层字段:  "rebalance_days"
                - yaml 段路径: "backtest__rebalance_days" / "output__output_dir"
                - 嵌套 dict:  {"backtest": {"rebalance_days": 5}}

        Returns:
            BacktestConfig 实例
        """
        raw = load_yaml(path)
        bt = raw.get("backtest", {})
        fw = raw.get("factor_weights", {})

        # 规则 23: 加载 env vars 覆盖（QUANT_BACKTEST__REBALANCE_FREQ=5）
        env_overrides = load_env_overrides()
        if env_overrides:
            logger.info("检测到 env 覆盖: %s", list(env_overrides.keys()))

        # 规则 23: 应用 overrides + env 覆盖到 raw dict
        # 优先级: yaml < env < overrides
        if env_overrides or overrides:
            raw = cls._apply_layered_overrides(
                raw, env_overrides=env_overrides, runtime_overrides=overrides or {}
            )
            # 重新提取各段（已被覆盖）
            bt = raw.get("backtest", {})
            fw = raw.get("factor_weights", {})

        # P1-1 整改：YAML 未提供 factor_weights 时记 warning 并使用空字典
        if not fw:
            logger.warning(
                "config.yaml 未配置 factor_weights，将使用空字典。"
                "请在 config.yaml factor_weights 节点显式配置 5 子策略权重。"
            )
            fw = {}

        # 各模块配置委托给各自的from_yaml解析
        factors_cfg = FactorModuleConfig.from_yaml(raw)
        stop_cfg = StopOptimizationConfig.from_yaml(raw)
        validation_cfg = ValidationModuleConfig.from_yaml(raw)

        return cls(
            initial_cash=float(bt.get("initial_capital", INITIAL_CASH)),
            commission_rate=float(bt.get("commission", 0.0001)),
            slippage_rate=float(bt.get("slippage", 0.0001)),
            # 日期范围
            full_start=bt.get("full_start_date", "2016-01-01"),
            full_end=bt.get("full_end_date", "2025-12-31"),
            train_start=bt.get(
                "in_sample_start_date", bt.get("full_start_date", "2016-01-01")
            ),
            train_end=bt.get("in_sample_end_date", "2020-12-31"),
            test_start=bt.get("out_sample_start_date", "2021-01-01"),
            test_end=bt.get("out_sample_end_date", "2025-12-31"),
            # 样本分割
            in_sample_end=bt.get("in_sample_end_date"),
            out_sample_start=bt.get("out_sample_start_date"),
            # 因子打分调仓
            rebalance_days=int(bt.get("rebalance_freq", 3)),
            factor_weights=fw,
            entry_threshold=float(bt.get("entry_threshold", 0.05)),
            stop_loss_pct=float(bt.get("stop_loss_pct", 0.03)),
            max_position_pct=float(bt.get("max_position_pct", 0.15)),
            max_total_position_pct=float(bt.get("max_total_position_pct", 0.6)),
            min_position_pct=float(bt.get("min_position_pct", 0.0)),
            use_cross_section=bool(bt.get("use_cross_section", True)),
            use_rank_score=bool(bt.get("use_rank_score", True)),
            use_rolling_ic=bool(bt.get("use_rolling_ic", True)),
            use_trend_filter=bool(bt.get("use_trend_filter", False)),
            # P0 整改：补充字段
            top_n_symbols=int(bt.get("top_n_symbols", 5)),
            weight_method=bt.get("weight_method", "risk_parity"),
            # 子策略体系
            signal_merge_method=bt.get("signal_merge_method", "equal_weight"),
            use_sub_strategies=bool(bt.get("use_sub_strategies", True)),
            use_new_factors=bool(bt.get("use_new_factors", True)),
            use_signal_abstraction=bool(bt.get("use_signal_abstraction", False)),
            signal_mode=str(bt.get("signal_mode", "cross_sectional")),
            cta_hybrid_weight=float(bt.get("cta_hybrid_weight", 0.5)),
            # 动态混合模式（方向二，2026-06-15）
            hybrid_blend_method=str(bt.get("hybrid_blend_method", "linear")),
            xs_position_base=float(bt.get("xs_position_base", 0.5)),
            xs_position_ceiling=float(bt.get("xs_position_ceiling", 1.0)),
            xs_opposite_penalty=float(bt.get("xs_opposite_penalty", 0.5)),
            # 品种与策略
            symbols=raw.get(
                "symbols", ["SHFE.RB", "DCE.M", "CZCE.TA", "SHFE.CU", "CFFEX.IF"]
            ),
            strategy_names=[
                s["name"]
                for s in raw.get("strategies", [])
                if isinstance(s, dict)
                and "name" in s
                and s["name"] != "cross_sectional"  # 模式标志，非子策略名
            ],
            # 输出与风控
            output_dir=raw.get("output", {}).get("output_dir", "output_validation"),
            bankruptcy_threshold=raw.get("risk_management", {}).get(
                "bankruptcy_threshold", 0.8
            ),
            # 模块配置
            factors_config=factors_cfg,
            stop_optimization_config=stop_cfg,
            validation_config=validation_cfg,
            # P0-1 补全：交叉验证开关（规则26）
            cross_validate=bool(bt.get("cross_validate", False)),
        )

    def to_yaml(self, path: str = "config.yaml"):
        """
        保存配置到 YAML 文件。

        P0 整改：补充写入 top_n_symbols / weight_method / use_sub_strategies
                / use_new_factors / min_position_pct。
        """
        # 先读取现有配置，避免覆盖其他配置
        raw = load_yaml(path)

        # 更新配置（P0 整改：补全缺失字段）
        raw["backtest"] = {
            "initial_capital": self.initial_cash,
            "commission": self.commission_rate,
            "slippage": self.slippage_rate,
            "full_start_date": self.full_start,
            "full_end_date": self.full_end,
            "in_sample_start_date": self.train_start,
            "in_sample_end_date": self.train_end,
            "out_sample_start_date": self.test_start,
            "out_sample_end_date": self.test_end,
            "rebalance_freq": self.rebalance_days,
            "entry_threshold": self.entry_threshold,
            "stop_loss_pct": self.stop_loss_pct,
            "max_position_pct": self.max_position_pct,
            "max_total_position_pct": self.max_total_position_pct,
            "min_position_pct": self.min_position_pct,
            "use_cross_section": self.use_cross_section,
            "use_rank_score": self.use_rank_score,
            "use_rolling_ic": self.use_rolling_ic,
            "use_trend_filter": self.use_trend_filter,
            "top_n_symbols": self.top_n_symbols,
            "weight_method": self.weight_method,
            "signal_merge_method": self.signal_merge_method,
            "use_sub_strategies": self.use_sub_strategies,
            "use_new_factors": self.use_new_factors,
            "cross_validate": self.cross_validate,
            "use_signal_abstraction": self.use_signal_abstraction,
            "signal_mode": self.signal_mode,
            "cta_hybrid_weight": self.cta_hybrid_weight,
            # 动态混合模式（方向二，2026-06-15）
            "hybrid_blend_method": self.hybrid_blend_method,
            "xs_position_base": self.xs_position_base,
            "xs_position_ceiling": self.xs_position_ceiling,
            "xs_opposite_penalty": self.xs_opposite_penalty,
        }

        raw["factor_weights"] = self.factor_weights
        raw["symbols"] = self.symbols

        # P2-1 整改：使用 yaml_utils.dump_yaml（自动处理 numpy 类型）
        dump_yaml(path, raw, sort_keys=False)

    @staticmethod
    def _apply_layered_overrides(
        raw: Dict,
        env_overrides: Dict[str, Dict],
        runtime_overrides: Dict,
    ) -> Dict:
        """按优先级合并 env 与 runtime 覆盖到 raw dict（规则 23）。

        优先级: raw (YAML) < env_overrides < runtime_overrides
        env_overrides 的段名已通过 ENV_SECTION_ALIAS 映射到 yaml 段名。
        runtime_overrides 的 key 支持:
            - 顶层字段名: "rebalance_days"  → raw["rebalance_days"]
            - 段路径: "backtest__rebalance_days"  → raw["backtest"]["rebalance_days"]
            - 嵌套 dict: {"backtest": {...}}  → 合并到 raw["backtest"]

        Returns:
            新的 raw dict（不修改入参）
        """
        # Step 1: 应用 env 覆盖（段名已在 load_env_overrides 内做过映射）
        result = merge_overrides(raw, env_overrides)

        # Step 2: 规范化 runtime_overrides（点号/双下划线 → 嵌套）
        normalized: Dict = {}
        for key, value in runtime_overrides.items():
            if isinstance(key, str) and "__" in key:
                section, field = key.split("__", 1)
                normalized.setdefault(section, {})[field] = value
            else:
                normalized[key] = value

        # Step 3: 应用 runtime（最高优先级）
        return merge_overrides(result, normalized)

    def update_strategy_params(self, best_params: dict, path: str = "config.yaml"):
        """
        更新策略参数到 YAML 文件。

        P2-1 整改：复用 yaml_utils.convert_numpy_types（不再内嵌定义）。

        Args:
            best_params: 最佳参数字典 {strategy_name: params_dict}
            path: YAML 文件路径
        """
        raw = load_yaml(path)

        # 更新策略参数
        if "strategies" not in raw:
            raw["strategies"] = []

        for strategy_name, params in best_params.items():
            converted_params = convert_numpy_types(params)
            updated = False
            for s in raw["strategies"]:
                if isinstance(s, dict) and s.get("name") == strategy_name:
                    s["params"] = converted_params
                    updated = True
                    break
            if not updated:
                raw["strategies"].append(
                    {"name": strategy_name, "params": converted_params}
                )

        # P2-1 整改：使用 yaml_utils.dump_yaml
        dump_yaml(path, raw, sort_keys=False)
