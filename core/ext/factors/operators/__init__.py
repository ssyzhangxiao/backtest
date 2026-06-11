"""算子扩展（operators/）— 规则21。

使用方式：

    # 基础算子（来自 base_ops.py，迁移自 core/factors/operators.py）
    from core.ext.factors.operators import sma, std, delay, delta, safe_div

    # TA-Lib 算子（来自 talib_ops.py，talib_ 前缀）
    from core.ext.factors.operators import talib_rsi, talib_atr
    rsi = talib_rsi(close, timeperiod=14)

    # 批量注册（可选）
    from core.ext.factors.operators import register_operators
    from core.ext.factors.operators.factory import create_operator
    register_operators()  # 把 talib_* 全部注册到工厂
    rsi = create_operator("talib_rsi", close, timeperiod=14)

复用约束（规则21.4）：
    - 不得重写 sma/std/ema 等基础算子（已统一在 base_ops.py）
    - TA-Lib 算子统一以 talib_ 前缀暴露
    - 失败时抛 ImportError（规则21.2），不静默回退
"""

from __future__ import annotations

# 基础算子（迁移自原 core/factors/operators.py）
from .base_ops import (
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

# TA-Lib 算子（规则21扩展）
from .talib_ops import (
    is_talib_available,
    talib_rsi,
    talib_macd,
    talib_atr,
    talib_bollinger,
    talib_adx,
    talib_kdj,
    talib_cci,
    talib_obv,
    talib_stoch,
)


__all__ = [
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
    # TA-Lib 算子
    "is_talib_available",
    "talib_rsi",
    "talib_macd",
    "talib_atr",
    "talib_bollinger",
    "talib_adx",
    "talib_kdj",
    "talib_cci",
    "talib_obv",
    "talib_stoch",
]
