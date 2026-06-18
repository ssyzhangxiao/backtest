"""
方向三：配对交易横截面信号模块。

架构位置：core/factors/pair_trading/

职责：
  1. 滚动 ADF 协整检验 + OLS 对冲比 → 维护有效配对池
  2. 配对价差 z-score → 连续信号 [-1, 1]
  3. 每月重筛配对 + 横截面聚合（每个品种聚合其所在的所有配对的净 z-score）

与现有 core/strategies/cta/pair_trading.py 的区别：
  - 旧：CTA 单信号层（symbol="A/B"），单配对 → 单信号
  - 新：横截面聚合层（symbol="A"），所有配对 → 净 z-score

用法::

    from core.factors.pair_trading import PairSelector, DEFAULT_PARAMS
    selector = PairSelector(close_matrix, **DEFAULT_PARAMS)
    pair_scores = selector.compute_symbol_scores(current_bar)
    # pair_scores = {"SHFE.AL": +0.8, "SHFE.CU": -0.8, ...}
"""

from .cointegration import (
    CointegrationResult,
    adf_pvalue,
    batch_rolling_cointegration,
    rolling_cointegration,
    rolling_ols_hedge_ratio,
)
from .pair_selector import (
    DEFAULT_PARAMS,
    PairSelector,
    PairSelectorParams,
)
from .spread_signal import (
    aggregate_pair_zscores_to_symbols,
    compute_spread_zscore,
    rolling_pair_zscore_matrix,
)

__all__ = [
    # cointegration
    "CointegrationResult",
    "adf_pvalue",
    "rolling_ols_hedge_ratio",
    "rolling_cointegration",
    "batch_rolling_cointegration",
    # spread_signal
    "compute_spread_zscore",
    "rolling_pair_zscore_matrix",
    "aggregate_pair_zscores_to_symbols",
    # pair_selector
    "PairSelector",
    "PairSelectorParams",
    "DEFAULT_PARAMS",
]
