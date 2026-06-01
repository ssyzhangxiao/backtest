"""
展期处理逻辑模块。

在 PyBroker 策略中实现期货合约展期（主动平仓换月）。
支持三种展期模式：时间触发、流动性触发、价差触发。

展期在 PyBroker 中的实现方式：
  由于 PyBroker 对每个 symbol 分别调用 execute 函数，
  展期逻辑通过检查自定义列（is_dominant, rollover_signal, rollover_from, rollover_to）
  来判断是否需要展期，并在 execute 中执行平仓操作。

  展期分为两步：
  1. 旧合约：检测到展期信号时平仓
  2. 新合约：由策略信号决定是否在新主力上开仓

  展期成本（价差+手续费）在交易记录中体现。

使用 RolloverManager 与 PyBroker 集成的步骤：
  1. 使用 DataLoader 加载数据并构建连续序列
  2. 创建 RolloverManager 实例，调用 annotate_rollover_signals() 标注展期信号
  3. 将标注后的 DataFrame 传给 PyBroker
  4. 必须通过 pybroker.register_columns() 注册以下自定义列：
     'is_dominant', 'rollover_signal', 'rollover_from', 'rollover_to',
     'rollover_cost', 'dominant_symbol', 'prev_dominant_symbol', 'product'
  5. 使用 create_rollover_exec_fn() 包装策略执行函数
  6. 回测后调用 adjust_equity_for_rollover() 调整权益以反映展期成本
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
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

    时间触发和价差触发模式不依赖 DataLoader 的 rollover_flag，
    而是通过合约代码推断到期日，按到期日顺序确定下一个候选合约。
    流动性触发模式仍使用 rollover_flag 和 prev_dominant_symbol。

    Attributes:
        mode: 展期触发模式
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

    @staticmethod
    def _parse_expiry_date(symbol: str) -> Optional[pd.Timestamp]:
        """
        从合约代码推断到期日。

        合约代码格式假设：RB2401 表示2024年1月到期的螺纹钢合约。
        到期日取该月最后一个交易日（简化为该月最后一天）。

        Args:
            symbol: 合约代码

        Returns:
            到期日 Timestamp，无法解析时返回 None
        """
        try:
            year_month = symbol[-4:]
            year = 2000 + int(year_month[:2])
            month = int(year_month[2:])
            expiry = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
            return expiry
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _build_expiry_map(symbols: np.ndarray) -> Dict[str, pd.Timestamp]:
        """
        构建 symbol -> expiry_date 映射。

        Args:
            symbols: 所有合约代码数组

        Returns:
            字典 {合约代码: 到期日}
        """
        expiry_map: Dict[str, pd.Timestamp] = {}
        for sym in symbols:
            exp = RolloverManager._parse_expiry_date(sym)
            if exp is not None:
                expiry_map[sym] = exp
        return expiry_map

    @staticmethod
    def _find_next_contract_by_expiry(product_symbols: List[str],
                                      current_symbol: str,
                                      expiry_map: Dict[str, pd.Timestamp]) -> Optional[str]:
        """
        按到期日顺序找到当前合约的下一个合约。

        在同一品种的合约中，找到到期日晚于当前合约的最近一个合约。
        不使用持仓量次大，纯粹按到期日排序。

        Args:
            product_symbols: 同一品种的所有合约代码列表
            current_symbol: 当前主力合约代码
            expiry_map: 合约代码 -> 到期日映射

        Returns:
            下一个合约代码，若无则返回 None
        """
        current_expiry = expiry_map.get(current_symbol)
        if current_expiry is None:
            return None

        later_symbols = []
        for sym in product_symbols:
            sym_expiry = expiry_map.get(sym)
            if sym_expiry is not None and sym_expiry > current_expiry:
                later_symbols.append((sym, sym_expiry))

        if not later_symbols:
            return None

        later_symbols.sort(key=lambda x: x[1])
        return later_symbols[0][0]

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

        注意：展期信号仅在旧主力合约行上标记，
        非旧主力合约行的 rollover_signal 始终为 False。

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

    def _init_rollover_columns(self, df: pd.DataFrame) -> None:
        df['rollover_signal'] = False
        df['rollover_from'] = pd.Series([None] * len(df), index=df.index, dtype=object)
        df['rollover_to'] = pd.Series([None] * len(df), index=df.index, dtype=object)
        df['rollover_cost'] = 0.0

    def _annotate_liquidity_rollover(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        流动性触发展期：当新合约持仓量超过旧合约一定比例时展期。

        基于 DataLoader 已计算的 dominant_symbol 和 rollover_flag，
        进一步检查流动性条件。
        展期信号仅在旧主力合约行上标记。
        """
        self._init_rollover_columns(df)

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
                mask_old = (df['date'] == date) & (df['symbol'] == prev_dominant)
                df.loc[mask_old, 'rollover_signal'] = True
                df.loc[mask_old, 'rollover_from'] = prev_dominant
                df.loc[mask_old, 'rollover_to'] = new_dominant

                prev_close = day_data[day_data['symbol'] == prev_dominant]['close'].values
                new_close = day_data[day_data['symbol'] == new_dominant]['close'].values
                if len(prev_close) > 0 and len(new_close) > 0:
                    cost = abs(prev_close[0] - new_close[0]) + self.commission_per_lot * 2
                    df.loc[mask_old, 'rollover_cost'] = cost

        return df

    def _annotate_time_rollover(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        时间触发展期：在合约到期前N天展期。

        不依赖 rollover_flag，直接根据合约代码推断到期日。
        预先建立 symbol -> expiry_date 映射。
        对于每个日期，获取当前主力合约 dominant_symbol，
        若距离到期日 <= days_before_expiry，则找出该品种的下一个合约（按到期日排序）。
        若无下一个合约（最后一个合约），则不展期。
        展期信号仅在旧主力合约行上标记。

        性能优化：预计算每个合约的到期日，仅对临近到期的日期进行检查。
        """
        self._init_rollover_columns(df)

        symbols = df['symbol'].unique()
        expiry_map = self._build_expiry_map(symbols)

        product_groups = df.groupby('product')['symbol'].unique().to_dict()

        dominant_per_date = df.groupby('date')['dominant_symbol'].first()

        check_window = self.days_before_expiry + 30
        for date_val, dominant in dominant_per_date.items():
            if dominant not in expiry_map:
                continue

            days_to_expiry = (expiry_map[dominant] - pd.Timestamp(date_val)).days

            if days_to_expiry < 0 or days_to_expiry > check_window:
                continue

            if 0 < days_to_expiry <= self.days_before_expiry:
                product = df.loc[df['symbol'] == dominant, 'product'].iloc[0]
                product_symbols = product_groups.get(product, [])
                next_contract = self._find_next_contract_by_expiry(
                    product_symbols.tolist(), dominant, expiry_map
                )

                if next_contract is None:
                    continue

                day_data = df[df['date'] == date_val]
                mask_old = (df['date'] == date_val) & (df['symbol'] == dominant)
                df.loc[mask_old, 'rollover_signal'] = True
                df.loc[mask_old, 'rollover_from'] = dominant
                df.loc[mask_old, 'rollover_to'] = next_contract

                prev_close = day_data[day_data['symbol'] == dominant]['close'].values
                new_close = day_data[day_data['symbol'] == next_contract]['close'].values
                if len(prev_close) > 0 and len(new_close) > 0:
                    cost = abs(prev_close[0] - new_close[0]) + self.commission_per_lot * 2
                    df.loc[mask_old, 'rollover_cost'] = cost

        return df

    def _annotate_spread_rollover(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        价差触发展期：当新旧合约价差低于阈值时展期。

        不依赖 rollover_flag。
        对每个日期，获取当前主力合约 dominant_symbol，
        确定下一个候选主力（按到期日顺序，同时间模式）。
        计算新旧合约的收盘价价差，若 spread <= spread_threshold
        且总成本 <= cost_tolerance，则标记展期。
        如果价差条件长期不满足，在距离到期日 <= max_rollover_delay 时强制展期。
        展期信号仅在旧主力合约行上标记。
        """
        self._init_rollover_columns(df)

        symbols = df['symbol'].unique()
        expiry_map = self._build_expiry_map(symbols)

        product_groups = df.groupby('product')['symbol'].unique().to_dict()

        dominant_per_date = df.groupby('date')['dominant_symbol'].first()

        for date_val, dominant in dominant_per_date.items():
            if dominant not in expiry_map:
                continue

            days_to_expiry = (expiry_map[dominant] - pd.Timestamp(date_val)).days

            if days_to_expiry <= 0:
                continue

            product = df.loc[df['symbol'] == dominant, 'product'].iloc[0]
            product_symbols = product_groups.get(product, [])
            next_contract = self._find_next_contract_by_expiry(
                product_symbols.tolist(), dominant, expiry_map
            )

            if next_contract is None:
                continue

            day_data = df[df['date'] == date_val]
            prev_close = day_data[day_data['symbol'] == dominant]['close'].values
            new_close = day_data[day_data['symbol'] == next_contract]['close'].values

            if len(prev_close) == 0 or len(new_close) == 0:
                continue

            spread = abs(prev_close[0] - new_close[0])
            cost = spread + self.commission_per_lot * 2

            should_rollover = False

            if spread <= self.spread_threshold and cost <= self.cost_tolerance:
                should_rollover = True
            elif days_to_expiry <= self.max_rollover_delay:
                should_rollover = True

            if should_rollover:
                mask_old = (df['date'] == date_val) & (df['symbol'] == dominant)
                df.loc[mask_old, 'rollover_signal'] = True
                df.loc[mask_old, 'rollover_from'] = dominant
                df.loc[mask_old, 'rollover_to'] = next_contract
                df.loc[mask_old, 'rollover_cost'] = cost

        return df

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
        manager = self

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
        cumulative_deduction = 0.0

        for trade in rollover_trades:
            trade_date = pd.Timestamp(trade['date'])
            cost = trade.get('rollover_cost', 0.0)
            cumulative_deduction += cost
            mask = adjusted.index >= trade_date
            adjusted.loc[mask] -= cost

        return adjusted

    def get_actual_rollover_trades(self, session: Dict) -> List[Dict]:
        """
        从 PyBroker session 中收集实际发生的展期交易。

        替代基于 rollover_signal 的统计，使用实际交易数据更准确。

        Args:
            session: PyBroker 的 session 字典，包含 'rollover_trades' 键

        Returns:
            展期交易记录列表
        """
        return session.get('rollover_trades', [])

    def get_rollover_stats(self, df: pd.DataFrame,
                           session: Optional[Dict] = None) -> Dict:
        """
        计算展期统计信息。

        优先使用实际交易数据（从 session 中的 rollover_trades），
        若不可用则回退到基于 rollover_signal 的统计。

        Args:
            df: 带有展期信号的 DataFrame
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

        if actual_trades:
            costs = [t.get('rollover_cost', 0.0) for t in actual_trades]
            dates = [t.get('date', '') for t in actual_trades]
            stats: Dict = {
                "total_rollovers": len(actual_trades),
                "rollover_dates": dates,
                "avg_rollover_cost": np.mean(costs) if costs else 0.0,
                "max_rollover_cost": max(costs) if costs else 0.0,
                "total_rollover_cost": sum(costs),
                "actual_trades": actual_trades,
            }
            return stats

        if 'rollover_signal' not in df.columns:
            return {
                "total_rollovers": 0,
                "rollover_dates": [],
                "avg_rollover_cost": 0.0,
                "max_rollover_cost": 0.0,
                "total_rollover_cost": 0.0,
                "actual_trades": [],
            }

        rollover_days = df[df['rollover_signal']].drop_duplicates(subset=['date'])
        stats = {
            "total_rollovers": len(rollover_days),
            "rollover_dates": rollover_days['date'].dt.strftime('%Y-%m-%d').tolist() if len(rollover_days) > 0 else [],
            "avg_rollover_cost": rollover_days['rollover_cost'].mean() if len(rollover_days) > 0 else 0.0,
            "max_rollover_cost": rollover_days['rollover_cost'].max() if len(rollover_days) > 0 else 0.0,
            "total_rollover_cost": rollover_days['rollover_cost'].sum() if len(rollover_days) > 0 else 0.0,
            "actual_trades": [],
        }
        return stats
