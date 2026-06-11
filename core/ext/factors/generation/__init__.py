"""因子生成（generation/）— 规则21。

目标：在不污染 core/factors/alpha_futures/ 的前提下，提供可选的因子挖掘能力。

已实现模块：
- gplearn.py         遗传规划因子挖掘（需 pip install -r requirements-factors.txt）

规划模块：
- llm_generator.py   LLM 因子生成（需 pip install -r requirements-llm.txt）
- alphagpt.py        迭代优化编排（融合 GP + LLM）

复用约束（规则21.4 + 规则17）：
    生成的因子必须继承 core/factors/alpha_futures/base_factor.py 的 BaseFactor，
    并通过 register_factor 装饰器注册到 core/factors/alpha_futures/factor_registry。

**延迟加载约定**：本 __init__.py 不主动 import gplearn.py，避免 gplearn 未安装时
整个 core.ext.factors.generation 不可用。调用方须显式::

    from core.ext.factors.generation.gplearn import GPLearnFactorMiner
"""

from __future__ import annotations

__all__: list[str] = []
