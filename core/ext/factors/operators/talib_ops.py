"""TA-Lib 算子扩展（规则21）。

目标：在 core/factors/operators.py 自建算子之外，提供 TA-Lib 等第三方算子适配。

**复用约束（规则21.4）**：
    - 不得重写 core/factors/operators.py 的 sma/std/ema 等基础算子
    - TA-Lib 算子统一以 talib_ 前缀暴露，避免与核心算子重名
    - 函数签名与 core/factors/operators.py 保持一致：input=ndarray, output=ndarray

**依赖**：TA-Lib C 库 + Python 绑定
    安装：pip install -r requirements-factors.txt  # 包含 TA-Lib
    或：pip install TA-Lib（需先 brew install ta-lib）

**失败行为**（规则21.2）：
    未安装 TA-Lib 时调用任何算子会立即抛 ImportError，提示安装命令。
    不得静默回退到 numpy 实现（避免与核心算子混淆）。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


__all__ = [
    "talib_rsi",
    "talib_macd",
    "talib_atr",
    "talib_bollinger",
    "talib_adx",
    "talib_kdj",
    "talib_cci",
    "talib_obv",
    "talib_stoch",
    "is_talib_available",
]


# ---------------------------------------------------------------------------
# 延迟加载 TA-Lib（规则21.2：第三方 import 按需，未安装时立即报错）
# ---------------------------------------------------------------------------
_TALIB = None
_TALIB_IMPORT_ERROR: Optional[str] = None


def _require_talib():
    """获取 TA-Lib 模块，未安装时抛 ImportError 提示安装命令。"""
    global _TALIB, _TALIB_IMPORT_ERROR
    if _TALIB is not None:
        return _TALIB
    try:
        import talib  # noqa: F401
        _TALIB = talib
        return _TALIB
    except ImportError as e:
        _TALIB_IMPORT_ERROR = str(e)
        raise ImportError(
            "talib_ops 需要 TA-Lib，请执行：\n"
            "  macOS:   brew install ta-lib && pip install TA-Lib\n"
            "  Linux:   apt-get install ta-lib && pip install TA-Lib\n"
            "  Windows: https://github.com/mrjbq7/ta-lib#windows\n"
            "或：pip install -r requirements-factors.txt"
        ) from e


def is_talib_available() -> bool:
    """检查 TA-Lib 是否可用（不抛异常）。"""
    try:
        _require_talib()
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# TA-Lib 算子（与 core/factors/operators.py 签名风格一致：input=ndarray）
# ---------------------------------------------------------------------------
def talib_rsi(close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """RSI 相对强弱指数（TA-Lib 实现）。

    Args:
        close: 收盘价 ndarray
        timeperiod: 回看窗口，默认 14

    Returns:
        RSI 值 ndarray，前 timeperiod 个值为 NaN
    """
    return _require_talib().RSI(close.astype(float), timeperiod=timeperiod)


def talib_macd(
    close: np.ndarray,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD 指数平滑异同移动平均线（TA-Lib 实现）。

    Returns:
        (macd, signal, hist) 三元组，均为 ndarray
    """
    return _require_talib().MACD(
        close.astype(float),
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod,
    )


def talib_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timeperiod: int = 14,
) -> np.ndarray:
    """ATR 平均真实波幅（TA-Lib 实现）。

    Args:
        high/low/close: 最高/最低/收盘价 ndarray
        timeperiod: 回看窗口，默认 14

    Returns:
        ATR 值 ndarray
    """
    return _require_talib().ATR(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        timeperiod=timeperiod,
    )


def talib_bollinger(
    close: np.ndarray,
    timeperiod: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """布林带（TA-Lib 实现）。

    Returns:
        (upper, middle, lower) 三元组
    """
    return _require_talib().BBANDS(
        close.astype(float),
        timeperiod=timeperiod,
        nbdevup=nbdevup,
        nbdevdn=nbdevdn,
    )


def talib_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timeperiod: int = 14,
) -> np.ndarray:
    """ADX 平均趋向指数（TA-Lib 实现）。

    Returns:
        ADX 值 ndarray，范围 0~100
    """
    return _require_talib().ADX(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        timeperiod=timeperiod,
    )


def talib_kdj(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fastk_period: int = 9,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """KDJ 随机指标（TA-Lib 实现）。

    Returns:
        (slowk, slowd) 二元组
    """
    return _require_talib().STOCH(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowd_period=slowd_period,
    )


def talib_cci(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timeperiod: int = 14,
) -> np.ndarray:
    """CCI 顺势指标（TA-Lib 实现）。

    Returns:
        CCI 值 ndarray
    """
    return _require_talib().CCI(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        timeperiod=timeperiod,
    )


def talib_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """OBV 能量潮（TA-Lib 实现）。

    Returns:
        OBV 值 ndarray
    """
    return _require_talib().OBV(close.astype(float), volume.astype(float))


def talib_stoch(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fastk_period: int = 14,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic 随机指标（TA-Lib 实现）。

    Returns:
        (slowk, slowd) 二元组
    """
    return _require_talib().STOCH(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowd_period=slowd_period,
    )
