"""
资金流因子。

基于持仓量和成交量的资金流分析，包含三个子因子：
  - position_change_rate：持仓量变化率（多空力量对比）
  - capital_net_flow：资金净流入/流出（成交额方向加权）
  - oi_price_divergence：持仓量-价格背离信号

数据源：日频持仓量+成交量数据（现有CSV可支持）。
验证标准：单因子IC > 0.02，持仓变化率因子需通过Granger因果检验。
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CapitalFlowConfig:
    """资金流因子配置。"""

    # 持仓量变化率窗口（交易日）
    oi_change_window: int = 5

    # 资金流计算窗口（交易日）
    flow_window: int = 20

    # 背离检测阈值：持仓量变化率 > 此值视为显著
    oi_significance_threshold: float = 0.03

    # 价格变化阈值：价格变化率 > 此值视为显著
    price_significance_threshold: float = 0.005


class CapitalFlowFactor:
    """
    资金流因子计算器。

    基于持仓量和成交量数据计算资金流相关因子。
    与 BaseStrategy._compute_oi_change / _compute_oi_divergence 互补，
    此模块提供更完整的因子输出和批量计算能力。

    用法:
        factor = CapitalFlowFactor()
        scores = factor.compute_all(close, volume, open_interest)
    """

    def __init__(self, config: Optional[CapitalFlowConfig] = None):
        self.config = config or CapitalFlowConfig()

    def compute_position_change_rate(
        self, open_interest: np.ndarray
    ) -> np.ndarray:
        """
        计算持仓量变化率因子。

        持仓量变化率 = (OI_t - MA(OI, N)) / MA(OI, N)

        正值=增仓（资金流入），负值=减仓（资金流出）。
        期货持仓量变化领先于价格：
          - 增仓+价格上涨 = 多头主动
          - 增仓+价格下跌 = 空头主动
          - 减仓 = 资金撤离

        Args:
            open_interest: 持仓量序列

        Returns:
            变化率序列（与输入等长，前N-1个为NaN）
        """
        oi = np.asarray(open_interest, dtype=float)
        n = self.config.oi_change_window
        result = np.full_like(oi, np.nan, dtype=float)

        if len(oi) < n:
            return result

        # 滚动均值
        oi_ma = pd.Series(oi).rolling(window=n, min_periods=n).mean().values

        # 变化率
        valid = oi_ma > 0
        result[valid] = (oi[valid] - oi_ma[valid]) / oi_ma[valid]

        return result

    def compute_capital_net_flow(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        open_interest: np.ndarray,
    ) -> np.ndarray:
        """
        计算资金净流入/流出因子。

        资金净流入 = Σ(成交额 * 方向权重)
        方向权重：价格上涨日=+1，下跌日=-1

        标准化：除以N日均值，得到相对资金流强度。

        Args:
            close: 收盘价序列
            volume: 成交量序列
            open_interest: 持仓量序列

        Returns:
            资金净流入因子序列
        """
        c = np.asarray(close, dtype=float)
        v = np.asarray(volume, dtype=float)
        oi = np.asarray(open_interest, dtype=float)

        n = len(c)
        result = np.full(n, np.nan, dtype=float)
        window = self.config.flow_window

        if n < window + 1:
            return result

        # 日收益率方向
        returns = np.diff(c) / c[:-1]
        returns = np.insert(returns, 0, 0.0)

        # 日成交额
        turnover = c * v

        # 方向加权成交额
        direction = np.sign(returns)
        weighted_flow = turnover * direction

        # 滚动累计
        flow_series = pd.Series(weighted_flow)
        rolling_sum = flow_series.rolling(window=window, min_periods=window).sum()
        rolling_mean = flow_series.rolling(window=window, min_periods=window).mean()

        # 标准化：除以均值绝对值
        mean_abs = rolling_mean.abs()
        valid = mean_abs > 1e-10
        result_values = rolling_sum.values.copy()
        result_values[valid] = result_values[valid] / mean_abs[valid]
        result_values[~valid] = 0.0

        # 前window个为NaN
        result_values[:window] = np.nan

        return result_values

    def compute_oi_price_divergence(
        self,
        close: np.ndarray,
        open_interest: np.ndarray,
    ) -> np.ndarray:
        """
        计算持仓量-价格背离信号因子。

        背离信号：
          - 量增价涨（同向多）：+1 → 买盘强劲
          - 量增价跌（同向空）：-1 → 卖盘强劲
          - 量减价涨（背离空）：-1 → 虚涨无力
          - 量减价跌（背离多）：+1 → 虚跌见底
          - 无显著变化：0

        连续化处理：输出为 [-1, +1] 的连续值而非离散信号。

        Args:
            close: 收盘价序列
            open_interest: 持仓量序列

        Returns:
            背离信号因子序列
        """
        c = np.asarray(close, dtype=float)
        oi = np.asarray(open_interest, dtype=float)
        n = len(c)
        result = np.full(n, np.nan, dtype=float)

        window = self.config.oi_change_window
        if n < window + 1:
            return result

        # 价格变化率
        price_change = np.full(n, 0.0)
        price_change[1:] = (c[1:] - c[:-1]) / c[:-1]

        # 持仓量变化率
        oi_change = np.full(n, 0.0)
        oi_ma = pd.Series(oi).rolling(window=window, min_periods=window).mean().values
        valid_oi = oi_ma > 0
        oi_change[valid_oi] = (oi[valid_oi] - oi_ma[valid_oi]) / oi_ma[valid_oi]

        # 连续化背离信号
        oi_thresh = self.config.oi_significance_threshold
        px_thresh = self.config.price_significance_threshold

        for i in range(window, n):
            oi_c = oi_change[i]
            px_c = price_change[i]

            # 连续化：用变化幅度缩放信号强度
            if abs(oi_c) < oi_thresh and abs(px_c) < px_thresh:
                result[i] = 0.0
                continue

            # 量增价涨 → 多头信号
            if oi_c > oi_thresh and px_c > px_thresh:
                result[i] = min(1.0, (oi_c + px_c) / (oi_thresh + px_thresh))
            # 量增价跌 → 空头信号
            elif oi_c > oi_thresh and px_c < -px_thresh:
                result[i] = -min(1.0, (oi_c + abs(px_c)) / (oi_thresh + px_thresh))
            # 量减价涨 → 虚涨偏空
            elif oi_c < -oi_thresh and px_c > px_thresh:
                result[i] = -min(1.0, (abs(oi_c) + px_c) / (oi_thresh + px_thresh)) * 0.5
            # 量减价跌 → 虚跌偏多
            elif oi_c < -oi_thresh and px_c < -px_thresh:
                result[i] = min(1.0, (abs(oi_c) + abs(px_c)) / (oi_thresh + px_thresh)) * 0.5
            else:
                result[i] = 0.0

        return result

    def compute_all(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        open_interest: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        计算所有资金流因子。

        Args:
            close: 收盘价序列
            volume: 成交量序列
            open_interest: 持仓量序列

        Returns:
            {因子名: 因子值序列}
        """
        return {
            "position_change_rate": self.compute_position_change_rate(open_interest),
            "capital_net_flow": self.compute_capital_net_flow(close, volume, open_interest),
            "oi_price_divergence": self.compute_oi_price_divergence(close, open_interest),
        }
