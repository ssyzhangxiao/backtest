"""多频/高频数据处理器（handlers/）— 规则21 规划中。

目标：把"分频数据处理/特征工程"能力从核心剥离，仿照 qlib 的 handler 体系。

复用约束（规则21.4）：
    必须复用 core/data_loader.py 的 DataLoader 入口，
    不得绕过 DataLoader 直接读取数据。
"""

from __future__ import annotations

__all__: list[str] = []
