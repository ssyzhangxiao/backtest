"""因子扩展（factors/）— 规则21 规划中。

子目录：
- generation/  因子挖掘（GP/LLM/AlphaGPT）
- pool/        因子池（互斥 IC + 权重 + 衰减）
- operators/   算子扩展（TA-Lib 等）

复用约束（规则21.4）：
    - generation/  必须继承 core/factors/alpha_futures/base_factor.py 的 BaseFactor
    - pool/        必须复用 core/engine/factor_decay.py + core/factors/factor_evaluator.py
    - operators/   必须复用 core/factors/operators.py 的基础算子
"""

from __future__ import annotations

__all__: list[str] = []
