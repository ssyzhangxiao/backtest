"""
H_04: 价格加速度与相对持仓排名复合。

高阶复合因子，二阶价格变化(加速度)+短期资金面爆发→趋势启动极初期。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import delta, mean, safe_div, tsrank, zscore


@register_factor
class H_04(BaseFactor):
    """H_04: 价格加速度与相对持仓排名复合。"""

    name = "H_04"
    category = "高阶复合"
    formula = "(-1*ZSCORE(TSRANK(CLOSE,10),w)) * ZSCORE(DELTA(DELTA(CLOSE,1),1),w) * ZSCORE(TSRANK(OI_SAFE/MEAN(OI_SAFE,20),5),w)"
    dependencies = ["close", "oi_safe"]

    # P1整改：硬编码窗口改为类属性
    oi_ma_window: int = 20
    close_tsrank_window: int = 10
    oi_tsrank_window: int = 5

    def compute(
        self,
        close: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        # P0整改：zscore_window 从 kwargs 接收，oi_mean_20 内部计算
        zscore_window = kwargs.get("zscore_window", None)
        oi_mean_20 = mean(oi_safe, self.oi_ma_window)
        z_ts_rank = zscore(tsrank(close, self.close_tsrank_window), window=zscore_window)
        delta_close_1 = delta(close, 1)
        z_accel = zscore(delta(delta_close_1, 1), window=zscore_window)
        rel_oi = safe_div(oi_safe, oi_mean_20)
        z_oi_rank = zscore(tsrank(rel_oi, self.oi_tsrank_window), window=zscore_window)
        return (-1 * z_ts_rank) * z_accel * z_oi_rank

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)