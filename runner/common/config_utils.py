"""
配置管理模块。

使用 Pydantic 模型定义配置结构，提供配置验证和加载功能。
符合规则2：config.yaml 是单一数据源。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    # 尝试导入 pydantic v2
    from pydantic import BaseModel, Field, ValidationError

    PYDANTIC_AVAILABLE = True
    PYDANTIC_VERSION = 2
    logger.info("使用 Pydantic v2")
except ImportError:
    try:
        # 尝试导入 pydantic v1
        from pydantic import BaseModel, Field, ValidationError

        PYDANTIC_AVAILABLE = True
        PYDANTIC_VERSION = 1
        logger.info("使用 Pydantic v1")
    except ImportError:
        PYDANTIC_AVAILABLE = False
        PYDANTIC_VERSION = 0
        logger.warning("Pydantic 未安装，将使用简化的配置验证")


# ============================================
# 配置模型定义
# ============================================

if PYDANTIC_AVAILABLE:

    class BacktestConfig(BaseModel):
        """回测核心配置模型"""

        initial_capital: float = Field(1000000, description="初始资金")
        commission: float = Field(0.0001, description="双边手续费率")
        slippage: float = Field(0.0001, description="滑点费率")
        full_start_date: str = Field(..., description="完整回测开始日期")
        full_end_date: str = Field(..., description="完整回测结束日期")
        in_sample_end_date: Optional[str] = Field(None, description="样本内结束日期")
        out_sample_start_date: Optional[str] = Field(None, description="样本外开始日期")
        rebalance_freq: int = Field(5, description="调仓周期（交易日）")
        stop_loss_pct: float = Field(0.03, description="单品种止损百分比")
        max_position_pct: float = Field(0.15, description="单品种最大仓位占比")
        max_total_position_pct: float = Field(0.60, description="总仓位上限")
        min_position_pct: float = Field(0.0, description="最小仓位比例")
        entry_threshold: float = Field(0.07, description="开仓阈值")
        top_n_symbols: int = Field(4, description="品种轮动数量")
        use_cross_section: bool = Field(True, description="启用横截面标准化")
        use_rank_score: bool = Field(True, description="启用排名叠加")
        use_trend_filter: bool = Field(True, description="启用趋势过滤")
        use_rolling_ic: bool = Field(True, description="启用滚动IC动态权重")
        missing_data_method: str = Field("fill_zero", description="缺失值处理方法")

    class WalkForwardConfig(BaseModel):
        """Walk-Forward 配置模型"""

        train_bars: int = Field(252, description="训练窗口（交易日）")
        test_bars: int = Field(63, description="测试窗口（交易日）")
        step_bars: int = Field(21, description="步进长度（交易日）")
        window: int = Field(252, description="训练窗口大小")
        step: int = Field(63, description="测试窗口大小")
        parallel: bool = Field(True, description="是否并行执行")
        max_workers: int = Field(4, description="最大并行数")

    class BootstrapConfig(BaseModel):
        """Bootstrap 配置模型"""

        n_samples: int = Field(5000, description="重采样次数")
        confidence_level: float = Field(0.90, description="置信水平")
        random_seed: int = Field(42, description="随机种子")

    class MonteCarloConfig(BaseModel):
        """蒙特卡洛配置模型"""

        n_simulations: int = Field(1000, description="模拟次数")
        random_seed: int = Field(42, description="随机种子")
        bankruptcy_threshold: float = Field(0.8, description="破产阈值（终值比例）")

    class FullConfig(BaseModel):
        """完整配置模型"""

        backtest: BacktestConfig
        symbols: List[str] = Field(default_factory=list, description="品种列表")
        factor_weights: Dict[str, float] = Field(
            default_factory=dict, description="因子权重"
        )
        strategies: List[Dict[str, Any]] = Field(
            default_factory=list, description="策略配置"
        )
        market_regime: Dict[str, Any] = Field(
            default_factory=dict, description="市场环境配置"
        )
        risk_management: Dict[str, Any] = Field(
            default_factory=dict, description="风控配置"
        )
        walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
        bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
        monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)
        factors: Dict[str, Any] = Field(
            default_factory=dict, description="因子模块配置"
        )
        adaptive: Dict[str, Any] = Field(
            default_factory=dict, description="自适应参数配置"
        )
        multi_tf: Dict[str, Any] = Field(
            default_factory=dict, description="多时间框架配置"
        )
        position: Dict[str, Any] = Field(
            default_factory=dict, description="动态仓位配置"
        )
        risk: Dict[str, Any] = Field(default_factory=dict, description="止损优化配置")
        instrument: Dict[str, Any] = Field(
            default_factory=dict, description="品种选择配置"
        )
        validation: Dict[str, Any] = Field(
            default_factory=dict, description="回测验证配置"
        )
        data: Dict[str, Any] = Field(default_factory=dict, description="数据配置")
        logging: Dict[str, Any] = Field(default_factory=dict, description="日志配置")
        output: Dict[str, Any] = Field(default_factory=dict, description="输出配置")


# ============================================
# 配置验证函数
# ============================================


def validate_backtest_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证配置字典的结构和值。

    Args:
        config: 原始配置字典

    Returns:
        验证通过后的配置字典

    Raises:
        ValueError: 配置验证失败
    """
    if not config:
        raise ValueError("配置字典为空")

    # 检查必需的顶级配置
    required_top_level = ["backtest", "symbols"]
    for key in required_top_level:
        if key not in config:
            raise ValueError(f"缺少必需配置项: {key}")

    # 检查 backtest 配置
    backtest = config.get("backtest", {})
    if not backtest:
        raise ValueError("backtest 配置为空")

    required_backtest = ["full_start_date", "full_end_date"]
    for key in required_backtest:
        if key not in backtest or not backtest[key]:
            raise ValueError(f"backtest 缺少必需配置: {key}")

    # 如果使用 Pydantic，进行详细验证
    if PYDANTIC_AVAILABLE:
        try:
            # 确保有默认配置
            full_config = _merge_with_defaults(config)
            validated = FullConfig(**full_config)
            logger.info("配置验证通过（Pydantic）")
            # 兼容 pydantic v1 和 v2
            if PYDANTIC_VERSION == 2:
                return validated.model_dump()
            else:
                return validated.dict()
        except ValidationError as e:
            logger.error(f"Pydantic 配置验证失败: {e}")
            raise ValueError(f"配置验证失败: {e}") from e

    # 简化的验证逻辑（无 Pydantic 时）
    logger.warning("使用简化配置验证（无 Pydantic）")
    return config


def _merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    将用户配置与默认配置合并。

    Args:
        config: 用户配置字典

    Returns:
        合并后的配置字典
    """
    # 创建默认配置模板
    defaults: Dict[str, Any] = {
        "backtest": {
            "initial_capital": 1000000,
            "commission": 0.0001,
            "slippage": 0.0001,
            "rebalance_freq": 5,
            "stop_loss_pct": 0.03,
            "max_position_pct": 0.15,
            "max_total_position_pct": 0.60,
            "min_position_pct": 0.0,
            "entry_threshold": 0.07,
            "top_n_symbols": 4,
            "use_cross_section": True,
            "use_rank_score": True,
            "use_trend_filter": True,
            "use_rolling_ic": True,
            "missing_data_method": "fill_zero",
        },
        "walk_forward": {
            "train_bars": 252,
            "test_bars": 63,
            "step_bars": 21,
            "window": 252,
            "step": 63,
            "parallel": True,
            "max_workers": 4,
        },
        "bootstrap": {"n_samples": 5000, "confidence_level": 0.90, "random_seed": 42},
        "monte_carlo": {
            "n_simulations": 1000,
            "random_seed": 42,
            "bankruptcy_threshold": 0.8,
        },
        "symbols": [],
        "factor_weights": {},
        "strategies": [],
        "market_regime": {},
        "risk_management": {},
        "factors": {},
        "adaptive": {},
        "multi_tf": {},
        "position": {},
        "risk": {},
        "instrument": {},
        "validation": {},
        "data": {},
        "logging": {},
        "output": {},
    }

    # 深度合并配置
    result = dict(defaults)
    for key, value in config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = dict(result[key])
            result[key].update(value)
        else:
            result[key] = value

    return result


def load_config_from_yaml(yaml_path: Path) -> Dict[str, Any]:
    """
    从 YAML 文件加载配置并验证。

    Args:
        yaml_path: YAML 文件路径

    Returns:
        验证后的配置字典

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 配置验证失败
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return validate_backtest_config(config)


def get_backtest_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取回测配置。

    Args:
        config: 完整配置字典

    Returns:
        backtest 配置字典
    """
    return config.get("backtest", {})


def get_walkforward_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取 Walk-Forward 配置。

    Args:
        config: 完整配置字典

    Returns:
        walk_forward 配置字典
    """
    wf_config = config.get("walk_forward", {})
    # 确保兼容性：同时支持 window/step 和 train_bars/test_bars
    if "window" not in wf_config and "train_bars" in wf_config:
        wf_config["window"] = wf_config["train_bars"]
    if "step" not in wf_config and "test_bars" in wf_config:
        wf_config["step"] = wf_config["test_bars"]
    return wf_config


def get_montecarlo_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取蒙特卡洛配置。

    Args:
        config: 完整配置字典

    Returns:
        monte_carlo 配置字典
    """
    return config.get("monte_carlo", {})


def get_factors_list(config: Dict[str, Any]) -> List[str]:
    """
    从配置获取因子列表。

    Args:
        config: 完整配置字典

    Returns:
        因子名称列表
    """
    # 先从 factor_weights 取键
    factor_weights = config.get("factor_weights", {})
    if factor_weights:
        return list(factor_weights.keys())

    # 备选：从 factors 配置取
    factors_config = config.get("factors", {})
    if "list" in factors_config:
        return factors_config["list"]

    # 默认因子
    return ["ts_momentum", "roll_yield", "alpha019", "alpha032"]


def get_missing_data_method(config: Dict[str, Any]) -> str:
    """
    获取缺失值处理方法。

    Args:
        config: 完整配置字典

    Returns:
        缺失值处理方法字符串
    """
    backtest = config.get("backtest", {})
    return backtest.get("missing_data_method", "fill_zero")
