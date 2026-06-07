"""
因子抽象基类。

定义所有因子必须实现的接口。
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import numpy as np


class BaseFactor(ABC):
    """
    因子抽象基类。

    所有因子必须继承此类并实现相应接口。
    """

    # 因子编号，如 "T_01"
    name: str = ""

    # 分类，如 "趋势"
    category: str = ""

    # 公式描述
    formula: str = ""

    # 依赖的公共数据字段，如 ["close", "oi_safe", "carry_orth"]
    dependencies: List[str] = []

    def __init__(self, config: Any):
        """
        初始化因子。

        Args:
            config: 全局配置对象（AlphaFuturesConfig）
        """
        self.config = config
        self._cache: Dict[str, Any] = {}

    @abstractmethod
    def compute(self, **kwargs: np.ndarray) -> np.ndarray:
        """
        纯计算逻辑，仅依赖 kwargs 提供的字段。

        Args:
            **kwargs: 依赖的公共数据字段

        Returns:
            因子值序列
        """
        pass

    def post_process(self, values: np.ndarray) -> np.ndarray:
        """
        可选后处理：缩尾、截断、标准化等，默认不做额外处理。

        Args:
            values: 原始因子值序列

        Returns:
            后处理后的因子值序列
        """
        return values

    def get_needs(self) -> List[str]:
        """
        返回需要从公共数据中提取的字段名。

        Returns:
            依赖字段名列表
        """
        return self.dependencies
