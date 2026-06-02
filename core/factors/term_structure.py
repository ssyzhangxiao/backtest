"""
期限结构因子模块。

基于近月/远月合约价差计算期限结构因子：
  - basis_rate：基差率 = (近月价 - 远月价) / 远月价
  - term_spread：期限价差 = 近月价 - 远月价
  - roll_yield：展期收益 = 远月价 - 近月价（正=远月升水，负=远月贴水）

规则9要求：新因子必须通过IC检验才能入库。
"""

from dataclasses import dataclass
from typing import Dict, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TermStructureConfig:
    """期限结构因子配置。"""

    # 基差率窗口（交易日）
    basis_window: int = 20
    # 展期收益平滑窗口
    roll_yield_smooth_window: int = 5


class TermStructureFactor:
    """
    期限结构因子计算器。

    基于近月/远月合约价格计算期限结构相关因子。

    用法:
        ts = TermStructureFactor()
        factors = ts.compute_all(near_price, far_price)
    """

    def __init__(self, config: Optional[TermStructureConfig] = None):
        self.config = config or TermStructureConfig()

    def compute_basis_rate(
        self, near_price: np.ndarray, far_price: np.ndarray
    ) -> np.ndarray:
        """
        计算基差率因子。

        basis_rate = (near - far) / far

        正值 = 近月升水（backwardation），负值 = 近月贴水（contango）

        Args:
            near_price: 近月合约价格序列
            far_price: 远月合约价格序列

        Returns:
            基差率序列
        """
        near = np.asarray(near_price, dtype=float)
        far = np.asarray(far_price, dtype=float)

        # 对齐长度
        min_len = min(len(near), len(far))
        near = near[-min_len:]
        far = far[-min_len:]

        # 避免除零
        safe_far = np.where(np.abs(far) < 1e-8, np.nan, far)
        basis = (near - far) / safe_far

        return basis

    def compute_term_spread(
        self, near_price: np.ndarray, far_price: np.ndarray
    ) -> np.ndarray:
        """
        计算期限价差因子。

        term_spread = near - far

        Args:
            near_price: 近月合约价格序列
            far_price: 远月合约价格序列

        Returns:
            期限价差序列
        """
        near = np.asarray(near_price, dtype=float)
        far = np.asarray(far_price, dtype=float)

        min_len = min(len(near), len(far))
        return near[-min_len:] - far[-min_len:]

    def compute_roll_yield(
        self, near_price: np.ndarray, far_price: np.ndarray
    ) -> np.ndarray:
        """
        计算展期收益因子。

        roll_yield = far - near

        正值 = 远月升水（contango），持有空头展期获利
        负值 = 远月贴水（backwardation），持有多头展期获利

        Args:
            near_price: 近月合约价格序列
            far_price: 远月合约价格序列

        Returns:
            展期收益序列
        """
        near = np.asarray(near_price, dtype=float)
        far = np.asarray(far_price, dtype=float)

        min_len = min(len(near), len(far))
        raw = far[-min_len:] - near[-min_len:]

        # 平滑处理
        w = self.config.roll_yield_smooth_window
        if len(raw) >= w:
            smoothed = np.convolve(raw, np.ones(w) / w, mode="same")
            # 边界修正
            for i in range(w // 2):
                smoothed[i] = np.mean(raw[: i + w // 2 + 1])
                smoothed[-(i + 1)] = np.mean(raw[-(i + w // 2 + 1):])
            return smoothed

        return raw

    def compute_all(
        self, near_price: np.ndarray, far_price: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        计算所有期限结构因子。

        Args:
            near_price: 近月合约价格序列
            far_price: 远月合约价格序列

        Returns:
            {因子名: 因子值序列}
        """
        return {
            "basis_rate": self.compute_basis_rate(near_price, far_price),
            "term_spread": self.compute_term_spread(near_price, far_price),
            "roll_yield": self.compute_roll_yield(near_price, far_price),
        }
