"""
因子模块（向后兼容 shim）— 规则 22 目录迁移 M-05。

⚠️ 物理位置：所有功能已迁移到 `core.ext.factors.*`
   本文件仅作为向后兼容的 re-export shim，给一个 release 周期后删除。

新位置映射：
    core.factors.factor_evaluator     → core.ext.factors.evaluator
    core.factors.factor_selector      → core.ext.factors.selector
    core.factors.factor_transformer   → core.ext.factors.transformer
    core.factors.factor_review        → core.ext.factors.review
    core.factors.operators            → core.ext.factors.operators
    core.factors.futures_data_cleaners→ core.ext.factors.cleaners
    core.factors.alpha_futures        → core.ext.factors.alpha_futures
    core.factors.alpha_futures_24     → core.ext.factors.alpha_futures.factor_engine
                                       （AlphaFutures24 内部委托给 FactorEngine）
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────
# 评估/筛选/变换/复核框架
# ──────────────────────────────────────────────────────
from core.ext.factors.evaluator import FactorEvaluator, FactorEvalResult
from core.ext.factors.selector import FactorSelector
from core.ext.factors.transformer import FactorTransformer
from core.ext.factors.review import FactorReviewer, FactorReviewReport

# ──────────────────────────────────────────────────────
# 基础算子（保留旧的导入路径供外部使用）
# ──────────────────────────────────────────────────────
from core.ext.factors.operators import (
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
    ema,
)

# ──────────────────────────────────────────────────────
# 商品期货清洗算子
# ──────────────────────────────────────────────────────
from core.ext.factors.cleaners import (
    compute_open_adj,
    compute_intraday_ret,
    compute_carry,
    compute_oi_safe,
    generate_delivery_exclude,
    adjust_price_for_roll,
    compute_adaptive_gap_weight,
)

# ──────────────────────────────────────────────────────
# 商品期货因子（24 因子 + 子策略聚合）
# ──────────────────────────────────────────────────────
from core.ext.factors.alpha_futures.config import (
    AlphaFuturesConfig,
    OIThresholdType,
)
from core.ext.factors.alpha_futures.factor_pipeline import (
    FactorPipeline,
    PipelineResult,
)
from core.ext.factors.alpha_futures.sub_strategy_aggregator import (
    compute_sub_strategy_scores_from_ohlcv,
)
from core.ext.factors.alpha_futures.factor_registry import (
    SUB_STRATEGY_FACTOR_GROUPS,
    get_sub_strategy_factors,
)

# ──────────────────────────────────────────────────────
# 旧 AlphaFutures24 API 委托 shim（保留向后兼容）
# ──────────────────────────────────────────────────────
from core.factors.alpha_futures_24 import AlphaFutures24


# ──────────────────────────────────────────────────────
# 模块级 shim：支持 `from core.factors import factor_review` 旧用法
# ──────────────────────────────────────────────────────
import sys as _sys
from core.ext.factors import (
    evaluator as factor_evaluator,
    selector as factor_selector,
    transformer as factor_transformer,
    review as factor_review,
    cleaners as futures_data_cleaners,
    operators as operators,
)
from core.ext.factors import alpha_futures as alpha_futures_pkg

# 让 `from core.factors.alpha_futures import X` 旧用法也可工作
_sys.modules.setdefault(
    "core.factors.alpha_futures",
    alpha_futures_pkg,
)
# 让 `from core.factors.operators import ema` 旧用法也可工作
_sys.modules.setdefault(
    "core.factors.operators",
    operators,
)
# 让 `from core.factors.futures_data_cleaners import X` 旧用法也可工作
_sys.modules.setdefault(
    "core.factors.futures_data_cleaners",
    futures_data_cleaners,
)
# 让 `from core.factors.factor_evaluator import X` 旧用法也可工作
_sys.modules.setdefault(
    "core.factors.factor_evaluator",
    factor_evaluator,
)
# 让 `from core.factors.factor_selector import X` 旧用法也可工作
_sys.modules.setdefault(
    "core.factors.factor_selector",
    factor_selector,
)
# 让 `from core.factors.factor_transformer import X` 旧用法也可工作
_sys.modules.setdefault(
    "core.factors.factor_transformer",
    factor_transformer,
)
# 让 `from core.factors.factor_review import X` 旧用法也可工作
_sys.modules.setdefault(
    "core.factors.factor_review",
    factor_review,
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
    "ema",
    # 商品期货因子
    "AlphaFuturesConfig",
    "OIThresholdType",
    "AlphaFutures24",
    "compute_open_adj",
    "compute_intraday_ret",
    "compute_carry",
    "compute_oi_safe",
    "generate_delivery_exclude",
    "adjust_price_for_roll",
    "compute_adaptive_gap_weight",
    # Pipeline
    "FactorPipeline",
    "PipelineResult",
    # 子策略聚合
    "compute_sub_strategy_scores_from_ohlcv",
    "SUB_STRATEGY_FACTOR_GROUPS",
    "get_sub_strategy_factors",
]
