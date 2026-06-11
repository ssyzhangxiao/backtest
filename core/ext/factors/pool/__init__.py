"""因子池（pool/）— 规则21。

目标：把"通过互斥 IC 筛选 + 动态权重"的因子池管理能力从核心剥离。

复用约束（规则21.4）：
    - manager.py 必须复用 core.ext.factors.evaluator.FactorEvaluator 的 IC 计算
    - manager.py 必须复用 core.ext.factors.selector.FactorSelector 的筛选逻辑
"""

from __future__ import annotations

from core.ext.factors.pool.manager import (
    FactorPoolConfig,
    FactorInfo,
    FactorPoolManager,
)

__all__ = [
    "FactorPoolConfig",
    "FactorInfo",
    "FactorPoolManager",
]
