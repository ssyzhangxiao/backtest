"""
组合层工具：风险平价融合（E4 整改 2026-06-07）。

提取自 runner/backtest/experiments/e1_e5.py 的私有工具函数，
统一为公共 API（重复#P1：避免 e1_e5.py 内重复实现）。
"""

from typing import Dict

import pandas as pd

_EPSILON = 1e-10


def calculate_rolling_volatility(returns: pd.Series, window: int = 60) -> pd.Series:
    """
    计算滚动波动率（年化前可用 std）。

    Args:
        returns: 收益率序列
        window: 滚动窗口

    Returns:
        滚动波动率序列
    """
    return returns.rolling(window=window, min_periods=window // 2).std()


def calculate_risk_parity_weights(
    strategy_returns: Dict[str, pd.Series], window: int = 60
) -> pd.DataFrame:
    """
    计算风险平价权重（向量化版本）。

    公式：inv_vol_i = 1 / (vol_i + eps)；weight_i = inv_vol_i / sum(inv_vol)

    Args:
        strategy_returns: {策略名: 收益率序列}
        window: 滚动波动率窗口

    Returns:
        权重 DataFrame，索引为时间，每列一个策略
    """
    df_returns = pd.DataFrame(strategy_returns).fillna(0.0)
    df_vol = pd.DataFrame(
        {name: calculate_rolling_volatility(df_returns[name], window) for name in df_returns.columns}
    )
    # 初始期波动率 NaN 用前后填充
    df_vol = df_vol.ffill().bfill()

    inv_vol = 1.0 / (df_vol + _EPSILON)
    sum_inv_vol = inv_vol.sum(axis=1)
    return inv_vol.div(sum_inv_vol, axis=0)


def calculate_risk_parity_fusion(
    strategy_returns: Dict[str, pd.Series], window: int = 60
) -> Dict[str, float]:
    """
    计算风险平价融合的最终权重（E4 顶层调用入口）。

    与 PortfolioManager.allocate_weights 的区别：
        - **本函数**（runner/common/portfolio_utils）：基于**多策略历史收益率时序**，
          计算滚动波动率并用 1/vol 倒数分配权重，返回**平均权重**。属于离线分析工具，
          用于回测期策略权重融合，输出时间窗口内稳定的权重字典。
        - **PortfolioManager.allocate_weights**（core/portfolio）：基于**单时点信号字典**
          （{品种: 综合得分}），按指定 method（equal_weight / risk_parity / score_proportional
          等）做**实时权重分配**，支持 top_n 截取、total_allocation 约束。属于组合管理
          运行时 API，输出当前调仓周期的目标权重。

    P1 整改：原 e1_e5.py 中先调用 _calculate_risk_parity_weights 获取时间序列，
    再用 .mean() 取平均权重。本函数直接返回平均权重字典，避免外部两步调用。

    Args:
        strategy_returns: {策略名: 收益率序列}
        window: 滚动波动率窗口

    Returns:
        {策略名: 归一化权重}，所有权重之和为 1.0
    """
    df_weights = calculate_risk_parity_weights(strategy_returns, window=window)
    if df_weights.empty:
        return {}
    avg = df_weights.mean()
    total = float(avg.sum())
    if total <= 0:
        return {}
    return {name: float(v / total) for name, v in avg.items()}
