"""
因子模块。

提供因子评估、变换、筛选和新因子实现的统一框架。

模块拆分：
  - factor_evaluator.py: 因子评估框架（IC/IR/稳定性）
  - factor_transformer.py: 因子变换（对数/指数/交叉项）
  - factor_selector.py: 因子筛选与去冗余
  - basic_factors.py: 基础因子（ts_momentum/roll_yield/alpha019/alpha032）
  - capital_flow.py: 资金流因子
  - term_structure.py: 期限结构因子
  - operators.py: 基础算子库（safe_div, delay, zscore 等）
  - futures_config.py: 商品期货因子配置类
  - futures_data_cleaners.py: 适用性工程改造数据清洗算子
  - alpha_futures_trend.py: 趋势类因子 T_01~T_05
  - alpha_futures_reversal.py: 回归类因子 R_01~R_05
  - alpha_futures_volatility.py: 波动率类因子 V_01~V_04
  - alpha_futures_money_flow.py: 资金流类因子 M_01~M_05
  - alpha_futures_high_order.py: 高阶复合类因子 H_01~H_05
  - alpha_futures_23.py: 编排入口 AlphaFutures24 类
"""

from .factor_evaluator import FactorEvaluator, FactorEvalResult
from .factor_transformer import FactorTransformer
from .factor_selector import FactorSelector
from .basic_factors import (
    compute_factor_scores_from_ohlcv,
    compute_ts_momentum,
    compute_roll_yield,
    compute_alpha019,
    compute_alpha032,
)
from .capital_flow import CapitalFlowFactor
from .term_structure import TermStructureFactor

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
from .futures_config import AlphaFuturesConfig, OIThresholdType
from .futures_data_cleaners import (
    compute_open_adj,
    compute_intraday_ret,
    compute_carry,
    compute_oi_safe,
    generate_delivery_exclude,
    adjust_price_for_roll,
)

# 商品期货因子编排入口
from .alpha_futures_23 import AlphaFutures24, AlphaFutures23

__all__ = [
    # 评估框架
    "FactorEvaluator",
    "FactorEvalResult",
    "FactorTransformer",
    "FactorSelector",
    # 旧因子模块
    "CapitalFlowFactor",
    "TermStructureFactor",
    # 基础因子
    "compute_factor_scores_from_ohlcv",
    "compute_ts_momentum",
    "compute_roll_yield",
    "compute_alpha019",
    "compute_alpha032",
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
    "AlphaFutures23",  # 向后兼容别名
]
