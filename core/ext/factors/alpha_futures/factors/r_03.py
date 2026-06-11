"""
R_03: 收益率变化与平滑开盘增仓相关性乘积。

回归因子，捕捉动量反转+资金流向判断多空翻转点信号。
"""

import numpy as np

from ..base_factor import BaseFactor
from ..factor_registry import register_factor
from ...operators import corr, delay, delta, safe_div, zscore


@register_factor
class R_03(BaseFactor):
    """R_03: 收益率变化与平滑开盘增仓相关性乘积。"""

    name = "R_03"
    category = "回归"
    formula = "(-1*ZSCORE(DELTA(RET,3),w)) * CORR(OPEN_ADJ, DELTA(OI_SAFE,1), 10)"
    dependencies = ["close", "open_adj", "oi_safe"]

    # P1整改：硬编码窗口改为类属性
    ret_delta_window: int = 3
    corr_window: int = 10

    def compute(
        self,
        close: np.ndarray,
        open_adj: np.ndarray,
        oi_safe: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        计算 R_03 因子。

        Args:
            close: 收盘价序列
            open_adj: 平滑开盘价序列
            oi_safe: 安全持仓量序列
            **kwargs: 其他参数（未使用）

        Returns:
            R_03 因子值序列

        注意：
            zscore(window=None) 是扩张窗口（expanding zscore），
            即使用截至当前 bar 的全部历史数据计算均值/标准差，
            严格无前瞻性（no lookahead bias）。
        """
        ret = safe_div(delta(close, 1), delay(close, 1))
        # P2整改：明确注释：zscore(..., window=None) = 扩张窗口，无前瞻性
        z_delta_ret = zscore(delta(ret, self.ret_delta_window), window=None)
        corr_val = corr(open_adj, delta(oi_safe, 1), self.corr_window)
        return (-1 * z_delta_ret) * corr_val

    def post_process(self, values: np.ndarray) -> np.ndarray:
        """
        后处理：1%和99%缩尾去除极端值。

        Args:
            values: 原始因子值序列

        Returns:
            后处理后的因子值序列
        """
        from ...operators import winsorize

        return winsorize(values, lower_pct=0.01, upper_pct=0.99)
