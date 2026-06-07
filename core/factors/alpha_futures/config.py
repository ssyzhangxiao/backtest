"""
Alpha Futures 因子库配置。

本模块定义唯一的 `AlphaFuturesConfig`（与 `core/factors/alpha_futures_24.py` 合并后的权威源）。
旧 `alpha_futures_24.py:AlphaFuturesConfig` 已改为 re-export 形式，调用方可继续使用原路径。
"""
from dataclasses import dataclass
from typing import Callable, Dict, Union

# 远月 OI 阈值类型：int（统一阈值）/ Dict[symbol, int]（品种差异化）/ Callable 动态计算
OIThresholdType = Union[int, Dict[str, int], Callable[[str], int]]


@dataclass
class AlphaFuturesConfig:
    """Alpha Futures 因子库配置（权威源）。

    字段说明：
      - gap_weight: 跳空修复默认权重（数据不足或未自适应时回落值）
      - gap_weight_window: 自适应权重滚动窗口（交易日）
      - limit_move_threshold: 涨跌停板阈值（默认 0.06 = 6%，A 股商品 10% 的一半）
      - carry_oi_threshold: 远月 OI 阈值，可统一阈值 / 品种映射 / 动态函数
      - momentum_orth_window: Carry 正交化的动量窗口
      - zscore_window: Z-Score 滚动窗口，0 表示扩张窗口（无前瞻）
      - symbol: 当前品种代码（用于 per-symbol 动态阈值）

    注意：删除 `gap_weight_min/max` 与 `delivery_exclude_days`（曾定义但未在任何代码路径使用）。
    """

    # 基础配置
    symbol: str = ""
    zscore_window: int = 0
    gap_weight: float = 0.5
    gap_weight_window: int = 20
    limit_move_threshold: float = 0.06
    carry_oi_threshold: OIThresholdType = 10000
    momentum_orth_window: int = 20

    @staticmethod
    def from_backtest_config(bt_config):
        """
        从 BacktestConfig 转换。

        使用 getattr + 默认值，与 BacktestConfig 解耦：
          - 若 BacktestConfig 后续新增对应字段，自动透传
          - 若无对应字段，使用本类默认值（保证因子库自洽）
        """
        return AlphaFuturesConfig(
            symbol=(
                bt_config.symbols[0]
                if getattr(bt_config, "symbols", None)
                else ""
            ),
            zscore_window=getattr(bt_config, "zscore_window", 0),
            gap_weight=getattr(bt_config, "gap_weight", 0.5),
            gap_weight_window=getattr(bt_config, "gap_weight_window", 20),
            limit_move_threshold=getattr(bt_config, "limit_move_threshold", 0.06),
            carry_oi_threshold=getattr(bt_config, "carry_oi_threshold", 10000),
            momentum_orth_window=getattr(bt_config, "momentum_orth_window", 20),
        )
