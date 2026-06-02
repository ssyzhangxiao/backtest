"""
回测验证模块。

提供蒙特卡洛模拟、参数敏感性分析和样本外验证。
"""

from .monte_carlo import MonteCarloSimulator
from .sensitivity import SensitivityAnalyzer

__all__ = [
    "MonteCarloSimulator",
    "SensitivityAnalyzer",
]
