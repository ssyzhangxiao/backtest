"""
H_02: 7日价格变化与持仓衰减线性排名复合因子。

高阶复合因子，衰减加权强调近期持仓异动，结合长期收益排名。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import decay_linear, delta, delay, mean, safe_div, sum_rolling, zscore


@register_factor
class H_02(BaseFactor):
    """H_02: 7日价格变化与持仓衰减线性排名复合因子。"""

    name = "H_02"
    category = "高阶复合"
    formula = "-1*ZSCORE(DELTA(CLOSE,7)*(1-ZSCORE(DECAYLINEAR(OI_SAFE/MEAN(OI_SAFE,20),9),w)),w) * (1+ZSCORE(SUM(RET,250),w))"
    dependencies = ["close", "oi_safe"]

    # P1整改：硬编码窗口改为类属性
    oi_ma_window: int = 20
    decay_window: int = 9
    close_delta_window: int = 7
    long_ret_window: int = 250

    def compute(
        self,
        close: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        # P0整改：zscore_window 从 kwargs 接收（引擎已注入），
        # oi_mean_20 内部计算，避免依赖未声明的预计算字段
        zscore_window = kwargs.get("zscore_window", None)
        oi_mean_20 = mean(oi_safe, self.oi_ma_window)
        rel_oi = safe_div(oi_safe, oi_mean_20)
        decay_oi = decay_linear(rel_oi, self.decay_window)
        z_decay = zscore(decay_oi, window=zscore_window)
        delta_close_7 = delta(close, self.close_delta_window)
        inner = delta_close_7 * (1 - z_decay)
        ret = safe_div(delta(close, 1), delay(close, 1))
        sum_ret_250 = sum_rolling(ret, self.long_ret_window)
        z_long_ret = zscore(sum_ret_250, window=zscore_window)
        return -1 * zscore(inner, window=zscore_window) * (1 + z_long_ret)

    def post_process(self, values: np.ndarray) -> np.ndarray:
        from ...operators import winsorize
        return winsorize(values, lower_pct=0.01, upper_pct=0.99)