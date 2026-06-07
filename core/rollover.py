"""
展期执行包装与成本统计模块。

2026-06-07 精简（P0-3）：
  - 删除 annotate_rollover_signals / _annotate_liquidity_rollover /
    _annotate_time_rollover / _annotate_spread_rollover 标注方法
  - 展期信号改由 DataLoader.rollover_flag 提供
  - RolloverManager 仅提供执行包装（create_rollover_exec_fn）
    和成本统计（adjust_equity_for_rollover / get_rollover_stats）

在 PyBroker 策略中实现期货合约展期（主动平仓换月）。
展期在 PyBroker 中的实现方式：
  由于 PyBroker 对每个 symbol 分别调用 execute 函数，
  展期逻辑通过检查自定义列（is_dominant, rollover_signal, rollover_from, rollover_to）
  来判断是否需要展期，并在 execute 中执行平仓操作。

  展期分为两步：
  1. 旧合约：检测到展期信号时平仓
  2. 新合约：由策略信号决定是否在新主力上开仓

  展期成本（价差+手续费）在交易记录中体现。

使用 RolloverManager 与 PyBroker 集成的步骤：
  1. 使用 DataLoader 加载数据并构建连续序列（DataLoader 已计算 rollover_flag）
  2. 将 DataFrame 传给 PyBroker
  3. 必须通过 pybroker.register_columns() 注册以下自定义列：
     'is_dominant', 'rollover_signal', 'rollover_from', 'rollover_to',
     'rollover_cost', 'dominant_symbol', 'prev_dominant_symbol', 'product'
  4. 使用 create_rollover_exec_fn() 包装策略执行函数
  5. 回测后调用 adjust_equity_for_rollover() 调整权益以反映展期成本
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from enum import Enum


class RolloverMode(Enum):
    """
    展期触发模式（保留枚举以兼容旧 import，不再被 RolloverManager 内部使用）。

    TIME: 时间触发 - 在合约到期前N天展期
    LIQUIDITY: 流动性触发 - 当新合约持仓量超过旧合约时展期
    SPREAD: 价差触发 - 当新旧合约价差低于阈值时展期
    """
    TIME = "time"
    LIQUIDITY = "liquidity"
    SPREAD = "spread"


class RolloverManager:
    """
    展期管理器（执行包装 + 成本统计）。

    2026-06-07 精简（P0-3）：
      - 删除标注方法（annotate_rollover_signals / _annotate_*）
      - 展期信号改由 DataLoader.rollover_flag 提供
      - 本类仅提供：
          * create_rollover_exec_fn：包装策略执行函数
          * adjust_equity_for_rollover：回测后扣除展期成本
          * get_rollover_stats：展期统计

    Attributes:
        mode: 展期触发模式（保留向后兼容，不再被内部使用）
        days_before_expiry: 时间触发模式下，到期前N天展期
        liquidity_ratio: 流动性触发模式下，新合约持仓量/旧合约持仓量 阈值
        spread_threshold: 价差触发模式下，价差阈值（元）
        cost_tolerance: 展期成本容忍度（最大允许价差，超过则不展期）
        commission_per_lot: 每手手续费
        max_rollover_delay: 价差触发模式下，最大展期延迟天数（超过则强制展期）
    """

    def __init__(self, mode: RolloverMode = RolloverMode.LIQUIDITY,
                 days_before_expiry: int = 5,
                 liquidity_ratio: float = 1.5,
                 spread_threshold: float = 20.0,
                 cost_tolerance: float = 50.0,
                 commission_per_lot: float = 5.0,
                 max_rollover_delay: int = 10):
        self.mode = mode
        self.days_before_expiry = days_before_expiry
        self.liquidity_ratio = liquidity_ratio
        self.spread_threshold = spread_threshold
        self.cost_tolerance = cost_tolerance
        self.commission_per_lot = commission_per_lot
        self.max_rollover_delay = max_rollover_delay

    def create_rollover_exec_fn(self, strategy_fn: Callable) -> Callable:
        """
        创建一个包装了展期逻辑的执行函数。

        该函数在调用原始策略逻辑之前，先检查并执行展期操作。
        这是将展期逻辑集成到 PyBroker 策略中的推荐方式。

        要求：在使用前，必须将 'is_dominant', 'rollover_signal',
        'rollover_from', 'rollover_to' 等列注册到 PyBroker 数据中
        （通过 pybroker.register_columns() 或 add_execution 的 indicators）。

        这些列会作为 ctx 的属性存在（PyBroker 机制），
        若属性不存在则回退到 ctx.data[ctx.idx] 访问。

        Args:
            strategy_fn: 原始策略执行函数，签名为 fn(ctx: ExecContext) -> None

        Returns:
            包装后的执行函数
        """
        def _get_ctx_attr(ctx, name, default=None):
            val = getattr(ctx, name, None)
            if val is not None:
                return val
            try:
                if hasattr(ctx, 'data') and hasattr(ctx, 'idx'):
                    data_col = ctx.data.get(name)
                    if data_col is not None:
                        return data_col[ctx.idx]
            except (KeyError, IndexError, TypeError):
                pass
            return default

        def wrapped_execute(ctx):
            rollover_signal = _get_ctx_attr(ctx, 'rollover_signal', False)
            rollover_from = _get_ctx_attr(ctx, 'rollover_from', None)

            if rollover_signal:
                if rollover_from is not None and ctx.symbol != rollover_from:
                    pass
                else:
                    long_pos = ctx.long_pos()
                    if long_pos:
                        ctx.sell_shares = long_pos.shares
                        if 'rollover_trades' not in ctx.session:
                            ctx.session['rollover_trades'] = []
                        rollover_to = _get_ctx_attr(ctx, 'rollover_to', None)
                        rollover_cost = _get_ctx_attr(ctx, 'rollover_cost', 0.0)
                        ctx.session['rollover_trades'].append({
                            'date': str(ctx.dt),
                            'symbol': ctx.symbol,
                            'action': 'close_long',
                            'shares': long_pos.shares,
                            'price': ctx.close[-1],
                            'rollover_to': rollover_to,
                            'rollover_cost': rollover_cost,
                        })
                        return

                    short_pos = ctx.short_pos()
                    if short_pos:
                        ctx.buy_shares = short_pos.shares
                        if 'rollover_trades' not in ctx.session:
                            ctx.session['rollover_trades'] = []
                        rollover_to = _get_ctx_attr(ctx, 'rollover_to', None)
                        rollover_cost = _get_ctx_attr(ctx, 'rollover_cost', 0.0)
                        ctx.session['rollover_trades'].append({
                            'date': str(ctx.dt),
                            'symbol': ctx.symbol,
                            'action': 'close_short',
                            'shares': short_pos.shares,
                            'price': ctx.close[-1],
                            'rollover_to': rollover_to,
                            'rollover_cost': rollover_cost,
                        })
                        return

            strategy_fn(ctx)

        return wrapped_execute

    def adjust_equity_for_rollover(self, equity_series: pd.Series,
                                   rollover_trades: List[Dict]) -> pd.Series:
        """
        根据展期交易记录调整权益曲线，扣除展期成本。

        PyBroker 不支持直接修改成交价，因此展期成本（价差+手续费）
        需要在回测后通过此方法从权益曲线中扣除。

        每笔展期交易的 rollover_cost 从对应日期的权益中扣除，
        后续日期的权益也相应调整。

        Args:
            equity_series: 原始权益曲线，索引为日期
            rollover_trades: 展期交易记录列表，每条包含 date, rollover_cost 等

        Returns:
            调整后的权益曲线
        """
        adjusted = equity_series.copy()

        for trade in rollover_trades:
            trade_date = pd.Timestamp(trade['date'])
            cost = trade.get('rollover_cost', 0.0)
            mask = adjusted.index >= trade_date
            adjusted.loc[mask] -= cost

        return adjusted

    def get_actual_rollover_trades(self, session: Dict) -> List[Dict]:
        """
        从 PyBroker session 中收集实际发生的展期交易。

        使用实际交易数据进行统计，比基于 rollover_signal 标注的统计更准确。

        Args:
            session: PyBroker 的 session 字典，包含 'rollover_trades' 键

        Returns:
            展期交易记录列表
        """
        return session.get('rollover_trades', [])

    def get_rollover_stats(self, df: pd.DataFrame,  # noqa: ARG002  (保留入参以兼容旧调用)
                           session: Optional[Dict] = None) -> Dict:
        """
        计算展期统计信息。

        仅使用 session 中的实际交易数据（rollover_trades）。
        不再回退到基于 rollover_signal 的统计（标注方法已于 2026-06-07 删除）。

        Args:
            df: 保留入参以兼容旧调用，忽略
            session: 可选，PyBroker 的 session 字典

        Returns:
            展期统计字典，包含：
            - total_rollovers: 展期次数
            - rollover_dates: 展期日期列表
            - avg_rollover_cost: 平均展期成本
            - max_rollover_cost: 最大展期成本
            - total_rollover_cost: 总展期成本
            - actual_trades: 实际展期交易记录（如有）
        """
        actual_trades: List[Dict] = []
        if session is not None:
            actual_trades = self.get_actual_rollover_trades(session)

        if not actual_trades:
            return {
                "total_rollovers": 0,
                "rollover_dates": [],
                "avg_rollover_cost": 0.0,
                "max_rollover_cost": 0.0,
                "total_rollover_cost": 0.0,
                "actual_trades": [],
            }

        costs = [t.get('rollover_cost', 0.0) for t in actual_trades]
        dates = [t.get('date', '') for t in actual_trades]
        return {
            "total_rollovers": len(actual_trades),
            "rollover_dates": dates,
            "avg_rollover_cost": float(np.mean(costs)) if costs else 0.0,
            "max_rollover_cost": max(costs) if costs else 0.0,
            "total_rollover_cost": sum(costs),
            "actual_trades": actual_trades,
        }
