"""
H_05: 三重共振因子（加权和版本）。

高阶复合因子，期限结构、持仓量趋势、价格突破方向三者按权重相加。

2026-06-11 重写（消除 OOS 过拟合）：
- 旧版（乘积形式）：Z(CARRY) * Z(ΔOI) * SIGN(ΔCLOSE)
  - 失效原因：OOS 震荡市中三者频繁异号，乘积=0 损失全部信号
  - 极端情况：三者两异一同，乘积=-1 反向
- 新版（加权和）：0.40*Z(CARRY) + 0.30*Z(ΔOI) + 0.30*SIGN(ΔCLOSE)
  - 优点：三者异号时仍保留部分信号
  - 权重分配：carry 主导（40%）+ ΔOI 与方向 30%/30% 辅助
  - 输出范围：连续，不再受 -1/0/+1 三值限制
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delta, sign, zscore, winsorize


@register_factor
class H_05(BaseFactor):
    """H_05: 三重共振因子（加权和版本）。"""

    name = "H_05"
    category = "高阶复合"
    formula = "0.40*Z(CARRY) + 0.30*Z(ΔOI) + 0.30*SIGN(ΔCLOSE)"
    dependencies = ["carry", "oi_safe", "close"]

    # 2026-06-11 重写：加权和系数（合计=1.0）
    weight_carry: float = 0.40
    weight_delta_oi: float = 0.30
    weight_price_dir: float = 0.30

    # P1整改：硬编码窗口改为类属性
    oi_delta_window: int = 5
    close_delta_window: int = 5

    def compute(
        self,
        carry: np.ndarray,
        oi_safe: np.ndarray,
        close: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        计算 H_05 因子（加权和版本）。

        Args:
            carry: Carry 期限结构因子
            oi_safe: 安全持仓量序列
            close: 收盘价序列
            **kwargs: zscore_window（引擎注入），>0 时为扩张窗口

        Returns:
            H_05 因子值序列（连续值，不再是 ±1/0 三值）
        """
        # P0整改：zscore_window 从 kwargs 接收（引擎已注入）
        zscore_window = kwargs.get("zscore_window", 0)
        if zscore_window is None or zscore_window <= 0:
            window_arg = None
        else:
            window_arg = zscore_window
        z_carry = zscore(carry, window=window_arg)
        z_delta_oi = zscore(delta(oi_safe, self.oi_delta_window), window=window_arg)
        price_dir = sign(delta(close, self.close_delta_window))
        # 加权和（替代原乘积）：消除"全部异号=全损"风险
        return (
            self.weight_carry * z_carry
            + self.weight_delta_oi * z_delta_oi
            + self.weight_price_dir * price_dir
        )

    def post_process(self, values: np.ndarray) -> np.ndarray:
        """
        后处理：1%和99%缩尾去除极端值。
        """
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)
