"""
展期处理逻辑模块。

在 PyBroker 策略中实现期货合约展期（主动平仓换月）。
支持三种展期模式：时间触发、流动性触发、价差触发。

展期在 PyBroker 中的实现方式：
  由于 PyBroker 对每个 symbol 分别调用 execute 函数，
  展期逻辑通过检查自定义列（is_dominant, rollover_flag, dominant_symbol）
  来判断是否需要展期，并在 execute 中执行平仓操作。

  展期分为两步：
  1. 旧合约：检测到非主力时平仓
  2. 新合约：由策略信号决定是否在新主力上开仓

  展期成本（价差+手续费）在交易记录中体现。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum


class RolloverMode(Enum):
    """
    展期触发模式。

    TIME: 时间触发 - 在合约到期前N天展期
    LIQUIDITY: 流动性触发 - 当新合约持仓量超过旧合约时展期
    SPREAD: 价差触发 - 当新旧合约价差低于阈值时展期
    """
    TIME = "time"
    LIQUIDITY = "liquidity"
    SPREAD = "spread"


class RolloverManager:
    """
    展期管理器。

    负责在回测数据上标注展期信号，并提供展期执行逻辑。
    展期信号通过自定义列注册到 PyBroker 数据中，
    在策略的 execute 函数中检查并执行。

    Attributes:
        mode: 展期触发模式
        days_before_expiry: 时间触发模式下，到期前N天展期
        liquidity_ratio: 流动性触发模式下，新合约持仓量/旧合约持仓量 阈值
        spread_threshold: 价差触发模式下，价差阈值（元）
        cost_tolerance: 展期成本容忍度（最大允许价差，超过则不展期）
        commission_per_lot: 每手手续费
    """

    def __init__(self, mode: RolloverMode = RolloverMode.LIQUIDITY,
                 days_before_expiry: int = 5,
                 liquidity_ratio: float = 1.5,
                 spread_threshold: float = 20.0,
                 cost_tolerance: float = 50.0,
                 commission_per_lot: float = 5.0):
        self.mode = mode
        self.days_before_expiry = days_before_expiry
        self.liquidity_ratio = liquidity_ratio
        self.spread_threshold = spread_threshold
        self.cost_tolerance = cost_tolerance
        self.commission_per_lot = commission_per_lot

    def annotate_rollover_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        在 DataFrame 上标注展期信号。

        根据选择的展期模式，计算每日是否应触发展期，
        并在 DataFrame 上添加辅助列。

        新增列：
        - rollover_signal: bool，当日是否应展期
        - rollover_from: str，展期源合约（旧主力）
        - rollover_to: str，展期目标合约（新主力）
        - rollover_cost: float，预估展期成本

        Args:
            df: DataLoader 输出的完整 DataFrame

        Returns:
            带有展期信号列的 DataFrame
        """
        result = df.copy()

        if self.mode == RolloverMode.LIQUIDITY:
            result = self._annotate_liquidity_rollover(result)
        elif self.mode == RolloverMode.TIME:
            result = self._annotate_time_rollover(result)
        elif self.mode == RolloverMode.SPREAD:
            result = self._annotate_spread_rollover(result)

        return result

    def _annotate_liquidity_rollover(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        流动性触发展期：当新合约持仓量超过旧合约一定比例时展期。

        基于 DataLoader 已计算的 dominant_symbol 和 rollover_flag，
        进一步检查流动性条件。
        """
        df['rollover_signal'] = False
        df['rollover_from'] = np.nan
        df['rollover_to'] = np.nan
        df['rollover_cost'] = 0.0

        if 'rollover_flag' not in df.columns:
            return df

        rollover_dates = df[df['rollover_flag']]['date'].unique()

        for date in rollover_dates:
            day_data = df[df['date'] == date]
            prev_dominant = day_data['prev_dominant_symbol'].iloc[0]
            new_dominant = day_data['dominant_symbol'].iloc[0]

            if pd.isna(prev_dominant):
                continue

            prev_oi = day_data[day_data['symbol'] == prev_dominant]['open_interest'].values
            new_oi = day_data[day_data['symbol'] == new_dominant]['open_interest'].values

            if len(prev_oi) == 0 or len(new_oi) == 0:
                continue

            if new_oi[0] >= prev_oi[0] * self.liquidity_ratio:
                mask = df['date'] == date
                df.loc[mask, 'rollover_signal'] = True
                df.loc[mask, 'rollover_from'] = prev_dominant
                df.loc[mask, 'rollover_to'] = new_dominant

                prev_close = day_data[day_data['symbol'] == prev_dominant]['close'].values
                new_close = day_data[day_data['symbol'] == new_dominant]['close'].values
                if len(prev_close) > 0 and len(new_close) > 0:
                    cost = abs(prev_close[0] - new_close[0]) + self.commission_per_lot * 2
                    df.loc[mask, 'rollover_cost'] = cost

        return df

    def _annotate_time_rollover(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        时间触发展期：在合约到期前N天展期。

        通过合约代码中的月份信息推断到期日，
        在到期前 days_before_expiry 天触发展期。

        合约代码格式假设：RB2401 表示2024年1月到期的螺纹钢合约。
        """
        df['rollover_signal'] = False
        df['rollover_from'] = np.nan
        df['rollover_to'] = np.nan
        df['rollover_cost'] = 0.0

        if 'rollover_flag' not in df.columns:
            return df

        symbols = df['symbol'].unique()
        expiry_map = {}
        for sym in symbols:
            try:
                year_month = sym[-4:]
                year = 2000 + int(year_month[:2])
                month = int(year_month[2:])
                expiry = pd.Timestamp(year=year, month=month, day=1) - pd.Timedelta(days=1)
                expiry_map[sym] = expiry
            except (ValueError, IndexError):
                continue

        for date_val in df['date'].unique():
            day_data = df[df['date'] == date_val]
            dominant = day_data['dominant_symbol'].iloc[0]

            if dominant not in expiry_map:
                continue

            days_to_expiry = (expiry_map[dominant] - pd.Timestamp(date_val)).days

            if 0 < days_to_expiry <= self.days_before_expiry:
                next_dominant = self._find_next_dominant(day_data, dominant)
                if next_dominant:
                    mask = df['date'] == date_val
                    df.loc[mask, 'rollover_signal'] = True
                    df.loc[mask, 'rollover_from'] = dominant
                    df.loc[mask, 'rollover_to'] = next_dominant

                    prev_close = day_data[day_data['symbol'] == dominant]['close'].values
                    new_close = day_data[day_data['symbol'] == next_dominant]['close'].values
                    if len(prev_close) > 0 and len(new_close) > 0:
                        cost = abs(prev_close[0] - new_close[0]) + self.commission_per_lot * 2
                        df.loc[mask, 'rollover_cost'] = cost

        return df

    def _annotate_spread_rollover(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        价差触发展期：当新旧合约价差低于阈值时展期。

        仅在价差可接受（低于 cost_tolerance）时才执行展期。
        """
        df['rollover_signal'] = False
        df['rollover_from'] = np.nan
        df['rollover_to'] = np.nan
        df['rollover_cost'] = 0.0

        if 'rollover_flag' not in df.columns:
            return df

        rollover_dates = df[df['rollover_flag']]['date'].unique()

        for date in rollover_dates:
            day_data = df[df['date'] == date]
            prev_dominant = day_data['prev_dominant_symbol'].iloc[0]
            new_dominant = day_data['dominant_symbol'].iloc[0]

            if pd.isna(prev_dominant):
                continue

            prev_close = day_data[day_data['symbol'] == prev_dominant]['close'].values
            new_close = day_data[day_data['symbol'] == new_dominant]['close'].values

            if len(prev_close) == 0 or len(new_close) == 0:
                continue

            spread = abs(prev_close[0] - new_close[0])
            cost = spread + self.commission_per_lot * 2

            if spread <= self.spread_threshold and cost <= self.cost_tolerance:
                mask = df['date'] == date
                df.loc[mask, 'rollover_signal'] = True
                df.loc[mask, 'rollover_from'] = prev_dominant
                df.loc[mask, 'rollover_to'] = new_dominant
                df.loc[mask, 'rollover_cost'] = cost

        return df

    @staticmethod
    def _find_next_dominant(day_data: pd.DataFrame, current_dominant: str) -> Optional[str]:
        """
        在当日数据中找到下一个主力合约候选。

        选择持仓量第二大的合约作为候选。

        Args:
            day_data: 当日所有合约数据
            current_dominant: 当前主力合约代码

        Returns:
            下一个主力合约代码，或 None
        """
        others = day_data[day_data['symbol'] != current_dominant]
        if others.empty:
            return None
        idx = others['open_interest'].idxmax()
        return others.loc[idx, 'symbol']

    def create_rollover_exec_fn(self, strategy_fn):
        """
        创建一个包装了展期逻辑的执行函数。

        该函数在调用原始策略逻辑之前，先检查并执行展期操作。
        这是将展期逻辑集成到 PyBroker 策略中的推荐方式。

        Args:
            strategy_fn: 原始策略执行函数，签名为 fn(ctx: ExecContext) -> None

        Returns:
            包装后的执行函数
        """
        manager = self

        def wrapped_execute(ctx):
            is_dominant = getattr(ctx, 'is_dominant', True)
            rollover_signal = getattr(ctx, 'rollover_signal', False)

            if rollover_signal and not is_dominant:
                long_pos = ctx.long_pos()
                if long_pos:
                    ctx.sell_shares = long_pos.shares
                    if 'rollover_trades' not in ctx.session:
                        ctx.session['rollover_trades'] = []
                    ctx.session['rollover_trades'].append({
                        'date': str(ctx.dt),
                        'symbol': ctx.symbol,
                        'action': 'close_long',
                        'shares': long_pos.shares,
                        'price': ctx.close[-1]
                    })
                    return

                short_pos = ctx.short_pos()
                if short_pos:
                    ctx.buy_shares = short_pos.shares
                    if 'rollover_trades' not in ctx.session:
                        ctx.session['rollover_trades'] = []
                    ctx.session['rollover_trades'].append({
                        'date': str(ctx.dt),
                        'symbol': ctx.symbol,
                        'action': 'close_short',
                        'shares': short_pos.shares,
                        'price': ctx.close[-1]
                    })
                    return

            strategy_fn(ctx)

        return wrapped_execute

    def get_rollover_stats(self, df: pd.DataFrame) -> Dict:
        """
        计算展期统计信息。

        Args:
            df: 带有展期信号的 DataFrame

        Returns:
            展期统计字典
        """
        if 'rollover_signal' not in df.columns:
            return {"total_rollovers": 0}

        rollover_days = df[df['rollover_signal']].drop_duplicates(subset=['date'])
        stats = {
            "total_rollovers": len(rollover_days),
            "rollover_dates": rollover_days['date'].dt.strftime('%Y-%m-%d').tolist() if len(rollover_days) > 0 else [],
            "avg_rollover_cost": rollover_days['rollover_cost'].mean() if len(rollover_days) > 0 else 0,
            "max_rollover_cost": rollover_days['rollover_cost'].max() if len(rollover_days) > 0 else 0,
            "total_rollover_cost": rollover_days['rollover_cost'].sum() if len(rollover_days) > 0 else 0,
        }
        return stats
