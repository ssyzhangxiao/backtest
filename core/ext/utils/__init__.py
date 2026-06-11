"""工具函数扩展（utils/）— 规则21 规划中。

目标：把"与第三方库深度耦合的工具函数"从 core/runner/ 剥离。

注意：通用工具函数（无第三方依赖）仍放 runner/common/utils.py，
utils/ 仅放需要可选依赖的扩展工具。
"""

from __future__ import annotations

__all__: list[str] = []
