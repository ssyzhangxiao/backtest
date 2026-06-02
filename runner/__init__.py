"""
回测流水线编排包。

runner/ 是编排层，仅调用 core/ 和 utils/ 的公共接口，
不重新实现核心逻辑。详见规则17、18。
"""

from .pipeline import Pipeline

__all__ = ["Pipeline"]
