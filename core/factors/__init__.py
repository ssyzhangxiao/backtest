"""
因子模块。

提供因子评估、变换、筛选、复核和新因子实现的统一框架。

模块拆分：
  - factor_evaluator.py: 因子评估框架（IC/IR/稳定性）
  - factor_transformer.py: 因子变换（对数/指数/交叉项）
  - factor_selector.py: 因子筛选与去冗余
  - factor_review.py: 因子复核（6项质量检查）
  - operators.py: 基础算子库（safe_div, delay, zscore 等）
  - futures_data_cleaners.py: 适用性工程改造数据清洗算子
  - alpha_futures_24.py: 30因子编排入口（委托给 FactorEngine）
  - alpha_futures/: 新因子类体系（BaseFactor + 注册表 + FactorEngine + Pipeline + sub_strategy_aggregator）

P0 整改（2026-06-07）：
  - 废弃 basic_factors.py（compute_factor_scores_from_ohlcv 委托给
    alpha_futures.sub_strategy_aggregator.compute_sub_strategy_scores_from_ohlcv）
"""

from .factor_evaluator import FactorEvaluator, FactorEvalResult
from .factor_transformer import FactorTransformer
from .factor_selector import FactorSelector
from .factor_review import FactorReviewer, FactorReviewReport

# 基础算子（供外部直接使用）
from .operators import (
    safe_div,
    delay,
    delta,
    sma,
    std,
    sum_rolling,
    mean,
    corr,
    zscore,
    tsrank,
    sign,
    abs_,
    log,
    decay_linear,
    sma_ema,
    winsorize,
    clipping,
)

# 商品期货因子配置与清洗算子
from .alpha_futures_24 import AlphaFuturesConfig, OIThresholdType, AlphaFutures24
from .alpha_futures.factor_pipeline import FactorPipeline, PipelineResult
from .alpha_futures.sub_strategy_aggregator import (
    compute_sub_strategy_scores_from_ohlcv,
)
from .alpha_futures.factor_registry import (
    SUB_STRATEGY_FACTOR_GROUPS,
    get_sub_strategy_factors,
)
from .futures_data_cleaners import (
    compute_open_adj,
    compute_intraday_ret,
    compute_carry,
    compute_oi_safe,
    generate_delivery_exclude,
    adjust_price_for_roll,
)

__all__ = [
    # 评估框架
    "FactorEvaluator",
    "FactorEvalResult",
    "FactorTransformer",
    "FactorSelector",
    "FactorReviewer",
    "FactorReviewReport",
    # 基础算子
    "safe_div",
    "delay",
    "delta",
    "sma",
    "std",
    "sum_rolling",
    "mean",
    "corr",
    "zscore",
    "tsrank",
    "sign",
    "abs_",
    "log",
    "decay_linear",
    "sma_ema",
    "winsorize",
    "clipping",
    # 商品期货因子
    "AlphaFuturesConfig",
    "OIThresholdType",
    "compute_open_adj",
    "compute_intraday_ret",
    "compute_carry",
    "compute_oi_safe",
    "generate_delivery_exclude",
    "adjust_price_for_roll",
    "AlphaFutures24",
    # Pipeline
    "FactorPipeline",
    "PipelineResult",
    # P0 整改：从 OHLCV 计算子策略得分（替代 basic_factors）
    "compute_sub_strategy_scores_from_ohlcv",
    "SUB_STRATEGY_FACTOR_GROUPS",
    "get_sub_strategy_factors",
]
