"""
商品期货Alpha因子库配置。

配置项说明：
  - gap_weight: 跳空缺口修复权重（0.5为默认，实际按品种自适应）
  - carry_oi_threshold: Carry因子远月流动性阈值（支持按品种动态设置）
  - delivery_exclude_days: 交割月剔除天数
  - limit_move_threshold: 涨跌停板过滤阈值
  - momentum_orth_window: 动量正交化窗口
  - zscore_window: ZSCORE滚动窗口
  - symbol: 当前品种代码（用于按品种动态阈值）
"""

from dataclasses import dataclass
from typing import Callable, Dict, Union

# 流动性阈值类型：int=统一阈值, Dict[str,int]=按品种, Callable[[str],int]=动态函数
OIThresholdType = Union[int, Dict[str, int], Callable[[str], int]]


@dataclass
class AlphaFuturesConfig:
    """商品期货Alpha因子库配置。"""

    # 跳空缺口修复权重（0.5为默认，实际按品种自适应）
    gap_weight: float = 0.5

    # 跳空权重自适应范围
    gap_weight_min: float = 0.2
    gap_weight_max: float = 0.8

    # Carry因子远月流动性阈值（支持按品种动态设置）
    # int=统一阈值, Dict[str,int]=按品种, Callable[[str],int]=动态函数
    carry_oi_threshold: OIThresholdType = 10000

    # 交割月剔除天数（进入交割月前N天数据剔除）
    delivery_exclude_days: int = 5

    # 涨跌停板过滤：开盘涨跌幅绝对值超过此阈值时INTRADAY_RET置零
    limit_move_threshold: float = 0.06

    # 动量正交化窗口（用于Carry因子剥离动量效应，0=不进行正交化）
    momentum_orth_window: int = 20

    # ZSCORE滚动窗口（用于因子标准化，0=扩张窗口，不可用全序列）
    zscore_window: int = 0

    # 当前品种代码（用于按品种动态阈值）
    symbol: str = ""
