"""
策略执行器 — PyBroker 策略执行函数生成 + 风控适配。

位置: core/engine/strategy_executor.py

提供:
  - RiskManagerAdapter: PyBroker ExecContext 风控适配
  - StrategyExecutorFactory: 策略执行函数工厂
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.config import BacktestConfig
from core.strategy_registry import StrategyLibrary
from core.engine.switch_engine import FactorScoringEngine
from core.risk_controller import RiskController, RiskConfig

logger = logging.getLogger(__name__)

try:
    from pybroker import ExecContext
    PYBROKER_AVAILABLE = True
except ImportError:
    PYBROKER_AVAILABLE = False
    ExecContext = Any


class RiskManagerAdapter:
    """
    风控适配器：将 RiskController 的纯逻辑适配到 PyBroker ExecContext。

    核心风控逻辑委托给 RiskController，本类仅负责：
      1. 从 ExecContext 提取持仓数据
      2. 调用 RiskController 进行风控判断
      3. 将判断结果转换为 PyBroker 操作
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.02,
        max_position_pct: float = 0.2,
        max_total_position_pct: float = 0.4,
        daily_loss_limit: float = 0.03,
        use_atr_stop: bool = False,
        atr_multiplier: float = 2.0,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.max_position_pct = max_position_pct
        self.max_total_position_pct = max_total_position_pct
        self.daily_loss_limit = daily_loss_limit
        self.use_atr_stop = use_atr_stop
        self.atr_multiplier = atr_multiplier
        self._prev_equity: Optional[float] = None
        self._current_date: Optional[str] = None
        self._controller = RiskController(RiskConfig(
            stop_loss_pct=stop_loss_pct,
            max_position_pct=max_position_pct,
            max_total_position_pct=max_total_position_pct,
        ))

    @property
    def controller(self) -> RiskController:
        return self._controller

    def check_stop_loss(self, ctx: ExecContext) -> bool:
        """检查是否触发止损（含 ATR 动态止损）。"""
        triggered = False

        long_pos = ctx.long_pos()
        if long_pos:
            pnl_pct = self._compute_pnl_pct_long(long_pos)
            effective_stop = self._get_effective_stop(ctx)
            if -pnl_pct / 100 > effective_stop:
                ctx.sell_shares = int(long_pos.shares)
                triggered = True
                logger.info("多头止损触发: 亏损=%.2f%%, 阈值=%.2f%%", pnl_pct, effective_stop * 100)

        short_pos = ctx.short_pos()
        if short_pos:
            pnl_pct = self._compute_pnl_pct_short(short_pos)
            effective_stop = self._get_effective_stop(ctx)
            if -pnl_pct / 100 > effective_stop:
                ctx.buy_shares = int(short_pos.shares)
                triggered = True
                logger.info("空头止损触发: 亏损=%.2f%%, 阈值=%.2f%%", pnl_pct, effective_stop * 100)

        return triggered

    def _get_effective_stop(self, ctx: ExecContext) -> float:
        """计算有效止损阈值（ATR动态 + 固定止损取大值）。"""
        effective_stop = self.stop_loss_pct
        if self.use_atr_stop:
            atr_val = _get_indicator(ctx, "atr_14")
            current_close = _get_close(ctx)
            if atr_val is not None and current_close is not None and current_close > 0:
                atr_stop_pct = self.atr_multiplier * float(atr_val) / current_close
                effective_stop = max(self.stop_loss_pct, atr_stop_pct)
        return effective_stop

    def apply_position_limit(self, ctx: ExecContext, intended_shares: int) -> int:
        """检查单合约仓位限制，返回实际可下单数量。"""
        if intended_shares <= 0:
            return 0
        max_shares = ctx.calc_target_shares(self.max_position_pct)
        actual = min(intended_shares, int(max_shares))
        return max(actual, 0)

    def is_total_position_exceeded(self, ctx: ExecContext) -> bool:
        """检查总仓位是否超过上限，返回 True 表示超限。"""
        total_equity = float(ctx.total_equity)
        if total_equity <= 0:
            return True
        positions_value = 0.0
        long_pos = ctx.long_pos()
        if long_pos:
            positions_value += float(long_pos.market_value)
        short_pos = ctx.short_pos()
        if short_pos:
            positions_value += abs(float(short_pos.market_value))
        return positions_value / total_equity >= self.max_total_position_pct

    def check_daily_loss_limit(self, ctx: ExecContext) -> bool:
        """检查当日亏损是否超过限制。"""
        current_date = str(ctx.dt)
        if self._current_date != current_date:
            self._current_date = current_date
            self._prev_equity = float(ctx.total_equity)
            return False
        if self._prev_equity is None or self._prev_equity <= 0:
            return False
        current_equity = float(ctx.total_equity)
        daily_loss_pct = (self._prev_equity - current_equity) / self._prev_equity
        return daily_loss_pct > self.daily_loss_limit

    def wrap_with_risk_control(self, strategy_fn: Callable) -> Callable:
        """创建带有风控检查的策略执行函数。"""
        rm = self

        def wrapped_execute(ctx: ExecContext) -> None:
            if rm.check_stop_loss(ctx):
                return

            long_before = ctx.long_pos()
            short_before = ctx.short_pos()
            had_long = long_before is not None and long_before.shares > 0
            had_short = short_before is not None and short_before.shares > 0

            strategy_fn(ctx)

            if ctx.buy_shares and ctx.buy_shares > 0:
                if had_short and short_before:
                    ctx.buy_shares = min(ctx.buy_shares, int(short_before.shares))
                else:
                    over_daily = rm.check_daily_loss_limit(ctx)
                    over_total = rm.is_total_position_exceeded(ctx)
                    if over_daily or over_total:
                        ctx.buy_shares = 0
                    else:
                        ctx.buy_shares = rm.apply_position_limit(ctx, ctx.buy_shares)

            if ctx.sell_shares and ctx.sell_shares > 0:
                if had_long and long_before:
                    ctx.sell_shares = min(ctx.sell_shares, int(long_before.shares))
                else:
                    over_daily = rm.check_daily_loss_limit(ctx)
                    over_total = rm.is_total_position_exceeded(ctx)
                    if over_daily or over_total:
                        ctx.sell_shares = 0
                    else:
                        ctx.sell_shares = rm.apply_position_limit(ctx, ctx.sell_shares)

        return wrapped_execute

    @staticmethod
    def _compute_pnl_pct_long(pos) -> float:
        """计算多头持仓盈亏百分比。"""
        pnl = float(pos.pnl)
        cost_basis = float(pos.equity) - pnl
        if cost_basis <= 0:
            return 0.0
        return pnl / cost_basis * 100

    @staticmethod
    def _compute_pnl_pct_short(pos) -> float:
        """计算空头持仓盈亏百分比（使用入场价×手数作为成本基准）。"""
        pnl = float(pos.pnl)
        cost_basis = 0.0
        try:
            entry_price = float(pos.entry_price) if pos.entry_price is not None else 0.0
            shares = float(pos.shares) if pos.shares is not None else 0.0
            if entry_price > 0 and shares > 0:
                cost_basis = entry_price * shares
        except (ValueError, TypeError, AttributeError):
            pass
        if cost_basis <= 0:
            return 0.0
        return pnl / cost_basis * 100


def _get_indicator(ctx, name: str) -> Optional[float]:
    """从 PyBroker ctx 获取指标值（辅助函数），缺失时记录日志并返回 None。"""
    try:
        val = ctx.indicator(name)
        if val is not None and hasattr(val, '__len__') and len(val) > 0:
            return val[-1]
        if val is not None:
            return val
        logger.debug("指标 %s 为空", name)
        return None
    except (ValueError, KeyError) as e:
        logger.debug("指标 %s 不存在: %s", name, e)
        return None
    except Exception as e:
        logger.warning("获取指标 %s 异常: %s", name, e)
        return None


def _get_close(ctx) -> Optional[float]:
    """安全获取当前收盘价。"""
    try:
        close = ctx.close
        if hasattr(close, "__getitem__") and len(close) > 0:
            return float(close[-1])
        return float(close) if close is not None else None
    except Exception as e:
        logger.warning("获取收盘价异常: %s", e)
        return None


class StrategyExecutorFactory:
    """
    根据策略名称生成 PyBroker 策略执行函数（fn(ctx: ExecContext)）。

    因子打分调仓模式：
      1. 计算各因子得分（从 PyBroker 指标读取）
      2. 加权合成综合得分
      3. 调仓日根据得分决定方向和仓位
      4. 非调仓日不执行调仓（风控由 RiskManagerAdapter 外层处理）
    """

    def __init__(
        self,
        library: Optional[StrategyLibrary] = None,
        switch_engine: Optional[FactorScoringEngine] = None,
        config: Optional[BacktestConfig] = None,
    ) -> None:
        self.library = library or StrategyLibrary()
        self.switch_engine = switch_engine or FactorScoringEngine(self.library)
        self.config = config or BacktestConfig()
        self._position_size = self.config.max_position_pct
        # 滚动IC引擎（回测运行时注入）
        self._rolling_ic_engine: Optional[Any] = None
        # 前瞻收益缓存（用于更新IC）
        self._prev_close: Dict[str, float] = {}
        self._prev_factor_scores: Dict[str, Dict[str, float]] = {}
        # 总品种数（用于判断横截面数据是否收集完毕）
        self._total_symbols: int = 0

    def create_executor(
        self,
        strategy_name: str,
        enable_switching: bool = True,
        all_strategy_names: Optional[List[str]] = None,
        custom_params: Optional[Dict[str, Dict[str, any]]] = None,
    ) -> Callable:
        """
        创建因子打分调仓的 PyBroker 执行函数。

        Args:
            strategy_name: 主策略名称
            enable_switching: 兼容旧接口，打分模式下忽略
            all_strategy_names: 所有注册策略名称列表
            custom_params: 自定义策略参数

        Returns:
            可传入 pybroker.Strategy.add_execution() 的执行函数。
        """
        profile = self.library.get_profile(strategy_name)
        if profile is None:
            raise ValueError(f"未知策略: {strategy_name}")

        params = dict(profile.default_params)
        if custom_params and strategy_name in custom_params:
            params.update(custom_params[strategy_name])
        position_size = self._position_size
        scoring_engine = self.switch_engine

        factor_names = all_strategy_names or [strategy_name]
        strategy_weights = self._compute_strategy_weights(factor_names)
        strategy_params: Dict[str, Dict] = {}
        for sname in factor_names:
            sp = self.library.get_profile(sname)
            sparams = dict(sp.default_params) if sp else dict(params)
            if custom_params and sname in custom_params:
                sparams.update(custom_params[sname])
            strategy_params[sname] = sparams

        single_factor_mode = len(factor_names) == 1
        single_factor_name = factor_names[0] if single_factor_mode else None

        bar_counter: List[int] = [0]
        # 横截面数据收集状态
        cross_section_collected: List[bool] = [False]
        cross_section_finalized: List[bool] = [False]

        def executor_fn(ctx: ExecContext) -> None:
            """PyBroker 因子打分调仓执行函数。"""
            nonlocal params

            bar_counter[0] += 1
            trading_day_idx = bar_counter[0]

            symbol = ctx.symbol
            current_close = _get_close(ctx)
            
            # 1. 先更新滚动IC引擎（用上一bar的因子得分和当前bar的前瞻收益）
            if (self._rolling_ic_engine is not None
                    and self.config.use_rolling_ic
                    and symbol in self._prev_close
                    and symbol in self._prev_factor_scores):
                prev_close = self._prev_close[symbol]
                if prev_close > 0 and current_close is not None and current_close > 0:
                    forward_ret = (current_close - prev_close) / prev_close
                    self._rolling_ic_engine.update(
                        self._prev_factor_scores[symbol], forward_ret, symbol
                    )
                    dynamic_weights = self._rolling_ic_engine.get_dynamic_weights()
                    from core.engine.switch_engine import _shared_ic_weights
                    _shared_ic_weights.clear()
                    _shared_ic_weights.update(dynamic_weights)
                    # 同步到switch_engine的_ic_weights
                    scoring_engine.set_ic_weights(dynamic_weights)

            # 2. 再提取当前bar的因子得分（用于下一bar的IC更新）
            factor_scores = scoring_engine.extract_factor_scores(ctx, strategy_params)
            
            # 3. 缓存当前因子得分和收盘价
            if factor_scores:
                self._prev_factor_scores[symbol] = dict(factor_scores)
            if current_close is not None:
                self._prev_close[symbol] = current_close

            # 4. 收集横截面数据
            is_rebalance = scoring_engine.is_rebalance_day(
                trading_day_index=trading_day_idx,
                dt=ctx.dt,
            )

            if is_rebalance:
                if factor_scores:
                    scoring_engine.update_cross_section(symbol, factor_scores, dt=ctx.dt)
                else:
                    scoring_engine.update_cross_section(symbol, {}, dt=ctx.dt)
                cross_section_collected[0] = True

            if not is_rebalance:
                return

            # 5. 品种轮动检查：未入选的品种平仓
            if not scoring_engine.is_symbol_selected(symbol):
                has_long = ctx.pos(ctx.symbol, "long") is not None
                has_short = ctx.pos(ctx.symbol, "short") is not None
                if has_long or has_short:
                    self._close_all_positions(ctx)
                return

            # 6. 计算综合得分（使用最新的IC权重）
            if single_factor_mode and single_factor_name:
                composite_score = factor_scores.get(single_factor_name, 0.0)
            else:
                composite_score = scoring_engine.compute_composite_score(factor_scores)

            roll_yield_ma_val = scoring_engine.extract_indicator(ctx, "roll_yield_ma")
            spread_pct: Optional[float] = None
            if (roll_yield_ma_val is not None and current_close is not None
                    and roll_yield_ma_val > 0):
                spread_pct = (current_close - roll_yield_ma_val) / roll_yield_ma_val * 100

            direction, score_pct = scoring_engine.score_to_position(composite_score)

            # 趋势过滤：只顺趋势方向交易
            if scoring_engine.config.use_trend_filter and direction != 0:
                sma_20 = _get_indicator(ctx, "sma_20")
                if sma_20 is not None and current_close is not None:
                    is_uptrend = current_close > sma_20
                    if direction == 1 and not is_uptrend:
                        self._close_all_positions(ctx)
                        return
                    if direction == -1 and is_uptrend:
                        self._close_all_positions(ctx)
                        return

            atr_vol = _get_indicator(ctx, "atr_14")
            vol_scale = 1.0
            if atr_vol is not None and current_close is not None and current_close > 0:
                current_vol = float(atr_vol) / current_close
                target_vol = params.get("target_vol", 0.015)
                vol_scale = min(1.0, target_vol / max(current_vol, 0.0001))

            if (strategy_name == "roll_yield" and spread_pct is not None
                    and "roll_yield" in strategy_params):
                exit_thr = strategy_params["roll_yield"].get("exit_threshold", 0.5)
                if abs(spread_pct) < exit_thr:
                    self._close_all_positions(ctx)
                    return

            has_long = ctx.pos(ctx.symbol, "long") is not None
            has_short = ctx.pos(ctx.symbol, "short") is not None

            effective_size = self._compute_position_size(
                position_size, score_pct, vol_scale,
                min_position_pct=params.get("min_position_pct", self.config.min_position_pct),
            )

            self._execute_rebalance(ctx, direction, effective_size, has_long, has_short)

        executor_fn.__name__ = f"executor_{strategy_name}"

        risk_manager = RiskManagerAdapter(
            stop_loss_pct=self.config.stop_loss_pct,
            max_position_pct=self.config.max_position_pct,
            max_total_position_pct=self.config.max_total_position_pct,
            use_atr_stop=True,
        )
        return risk_manager.wrap_with_risk_control(executor_fn)

    @staticmethod
    def _compute_position_size(
        position_size: float,
        score_pct: float,
        vol_scale: float,
        min_position_pct: float = 0.0,
    ) -> float:
        """
        计算实际仓位比例。

        Args:
            position_size: 配置的最大仓位比例
            score_pct: 得分绝对值（0~1）
            vol_scale: 波动率缩放系数（0~1）
            min_position_pct: 最小仓位比例

        Returns:
            有效仓位比例，确保不超过 position_size
        """
        effective = position_size * score_pct * vol_scale
        effective = max(min_position_pct, effective)
        return min(effective, position_size)

    @staticmethod
    def _execute_rebalance(
        ctx: ExecContext,
        direction: int,
        effective_size: float,
        has_long: bool,
        has_short: bool,
    ) -> None:
        """执行调仓操作：根据方向和仓位调整持仓。"""
        if direction == 1 and not has_long:
            if has_short:
                ctx.cover_all_shares()
            ctx.buy_shares = ctx.calc_target_shares(effective_size)
        elif direction == -1 and not has_short:
            if has_long:
                ctx.sell_all_shares()
            ctx.sell_shares = ctx.calc_target_shares(effective_size)
        elif direction == 0:
            if has_long:
                ctx.sell_all_shares()
            if has_short:
                ctx.cover_all_shares()

    @staticmethod
    def _close_all_positions(ctx: ExecContext) -> None:
        """平掉当前品种所有持仓。"""
        if ctx.pos(ctx.symbol, "long") is not None:
            ctx.sell_all_shares()
        if ctx.pos(ctx.symbol, "short") is not None:
            ctx.cover_all_shares()

    def _compute_strategy_weights(self, strategy_names: List[str]) -> Dict[str, float]:
        """计算策略权重（风险倒数加权，止损比例作为风险代理）。"""
        weights: Dict[str, float] = {}
        total_inv_risk = 0.0

        enabled_names: List[str] = []
        for name in strategy_names:
            profile = self.library.get_profile(name)
            if profile is None or not profile.enabled:
                continue
            enabled_names.append(name)
            risk = profile.stop_loss_pct if profile.stop_loss_pct > 0 else 0.05
            inv_risk = 1.0 / risk
            weights[name] = inv_risk
            total_inv_risk += inv_risk

        if total_inv_risk > 0 and len(weights) > 0:
            weights = {k: v / total_inv_risk for k, v in weights.items()}
        elif enabled_names:
            n = len(enabled_names)
            weights = {name: 1.0 / n for name in enabled_names}

        return weights

    def create_fusion_executor(
        self,
        strategy_instances: Dict[str, Any],
        risk_manager: Any = None,
        use_weighted_fusion: bool = True,
        use_regime_filter: bool = False,
        signal_threshold: float = 0.5,
    ) -> Tuple[Callable, List]:
        """
        创建基于策略类 execute 方法的信号融合执行器。

        各策略输出信号 → 加权合成 → 调仓日执行。
        执行结果统一包装风控检查。

        Args:
            strategy_instances: {策略名: 策略实例}
            risk_manager: 风控适配器，如果提供则包装风控
            use_weighted_fusion: 是否使用风险倒数加权
            use_regime_filter: 兼容参数，当前忽略
            signal_threshold: 信号强度阈值

        Returns:
            (执行函数, 指标列表)
        """
        all_indicators: List = []
        for name, strategy in strategy_instances.items():
            try:
                all_indicators.extend(strategy.register_indicators())
            except Exception as e:
                logger.warning("策略 %s 注册指标失败: %s", name, e)

        strategy_names = list(strategy_instances.keys())
        if use_weighted_fusion and self.library is not None:
            weights = self._compute_strategy_weights(strategy_names)
        else:
            weights = {name: 1.0 / max(len(strategy_names), 1)
                       for name in strategy_names}

        position_size = self._position_size
        n_strategies = max(len(strategy_names), 1)
        strategy_fns = {name: strat.execute for name, strat in strategy_instances.items()}
        scoring_engine = self.switch_engine

        def fusion_fn(ctx: ExecContext) -> None:
            if not scoring_engine.is_rebalance_day(dt=ctx.dt):
                return

            signals: List[float] = []
            total_weight = 0.0

            for name, fn in strategy_fns.items():
                original_buy = ctx.buy_shares
                original_sell = ctx.sell_shares
                try:
                    fn(ctx)
                    w = weights.get(name, 0.0)
                    if ctx.buy_shares is not None:
                        signals.append(1.0 * w)
                        total_weight += w
                    elif ctx.sell_shares is not None:
                        signals.append(-1.0 * w)
                        total_weight += w
                except Exception as e:
                    logger.warning("融合执行: 策略 %s 执行异常: %s", name, e)
                finally:
                    ctx.buy_shares = original_buy
                    ctx.sell_shares = original_sell

            if signals and total_weight > 0:
                weighted_signal = sum(signals) / total_weight
                if weighted_signal > signal_threshold:
                    size = position_size / n_strategies
                    ctx.buy_shares = ctx.calc_target_shares(size)
                elif weighted_signal < -signal_threshold:
                    has_long = False
                    try:
                        pos = ctx.long_pos()
                        has_long = pos is not None and pos.shares > 0
                    except Exception as e:
                        logger.warning("融合执行器检查多头持仓失败: %s", e)
                    if has_long:
                        ctx.sell_all_shares()

        if risk_manager is not None:
            fusion_fn = risk_manager.wrap_with_risk_control(fusion_fn)

        return fusion_fn, all_indicators