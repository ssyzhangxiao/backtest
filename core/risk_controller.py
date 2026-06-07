"""
独立风控控制器。

纯风控逻辑类，无任何框架依赖。
BrokerAdapter 不再包含风控决策逻辑，仅负责执行。

风控规则：
  - 止损：单品种亏损超过阈值时强制平仓
  - ATR 动态止损：止损阈值取 max(固定止损, 2*ATR/Close)
  - 持仓上限：总持仓不超过设定比例
  - 单品种仓位上限：单品种不超过设定比例
  - 复合止损（追踪 + 时间 + 固定）：P0整改后整合为内部组件

P0整改（2026-06-07）：
  - **CompositeStopManager 整合为内部组件**
  - 对外只暴露统一风控接口（check_stop_loss / check_composite_stop / check_position_limit）
  - 旧的 pnl-based check_stop_loss 已标记 @deprecated，新代码请用 check_composite_stop
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import logging

from core.risk.composite_stop import CompositeStopManager, CompositeStopResult

_logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """风控配置。"""

    stop_loss_pct: float = 0.05
    max_position_pct: float = 0.2
    max_total_position_pct: float = 0.6
    stop_loss_cooldown: int = 1

    # 复合止损参数（P0整改：合并到 RiskConfig）
    use_composite_stop: bool = True
    fixed_stop_pct: float = 0.05
    trailing_mode: str = "pct"          # "pct" 或 "atr"
    trailing_pct: float = 0.03
    trailing_atr_mult: float = 2.0
    max_holding_days: int = 10
    time_target_return: float = 0.01
    stop_loss_verbose: bool = False     # 默认 False 避免回测刷屏


class RiskController:
    """
    风控控制器。

    职责：
      1. 检查单品种是否触发止损
      2. 检查持仓是否超限
      3. 计算有效止损阈值（ATR 动态止损）
      4. 维护止损冷却期
      5. **P0整改**：整合 CompositeStopManager（追踪+时间+固定止损）

    不负责：
      - 执行交易（由 BrokerAdapter 负责）
      - 因子计算（由 FactorScoringEngine 负责）

    用法:
        # 兼容旧接口（pnl-based 止损）
        triggered = controller.check_stop_loss(symbol, mv, pnl, close, atr, day_idx)

        # 新接口（完整复合止损）
        result = controller.check_composite_stop(
            symbol="rb2401",
            direction="long",
            entry_price=3800,
            current_price=3700,
            highest_since_entry=3900,
            entry_day=0, current_day=12,
            atr_value=50.0,
        )
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        """
        初始化风控控制器。

        Args:
            config: 风控配置（None 时使用默认配置）
        """
        self.config = config or RiskConfig()
        self._cooldown_until: Dict[str, int] = {}

        # P0整改：CompositeStopManager 作为内部组件初始化
        if self.config.use_composite_stop:
            self._composite_stop = CompositeStopManager(
                fixed_stop_pct=self.config.fixed_stop_pct,
                trailing_mode=self.config.trailing_mode,
                trailing_pct=self.config.trailing_pct,
                trailing_atr_mult=self.config.trailing_atr_mult,
                max_holding_days=self.config.max_holding_days,
                time_target_return=self.config.time_target_return,
                verbose=self.config.stop_loss_verbose,
            )
        else:
            self._composite_stop = None

    @property
    def composite_stop(self) -> Optional[CompositeStopManager]:
        """获取复合止损管理器（供高级用法访问，P0整改后整合为内部组件）。"""
        return self._composite_stop

    # -----------------------------------------------------------------------
    # 旧接口（已废弃，保留仅用于向后兼容）：pnl-based 止损
    # -----------------------------------------------------------------------
    def check_stop_loss(
        self,
        symbol: str,
        position_market_value: float,
        position_pnl: float,
        current_close: float,
        atr_val: Optional[float] = None,
        trading_day_index: int = 0,
    ) -> bool:
        """
        检查是否触发止损（pnl-based 兼容接口）。

        .. deprecated:: 2026-06-07
            此方法已废弃，将于 v4.0 移除。
            新代码请使用 :meth:`check_composite_stop` 走完整的复合止损流程
            （固定止损 + 追踪止损 + 时间止损的优先级叠加）。

        Args:
            symbol: 品种代码
            position_market_value: 持仓市值
            position_pnl: 持仓盈亏
            current_close: 当前收盘价
            atr_val: ATR 值（可选，用于动态止损）
            trading_day_index: 当前交易日序号

        Returns:
            True 表示触发止损
        """
        warnings.warn(
            "RiskController.check_stop_loss 已废弃，请改用 check_composite_stop",
            DeprecationWarning,
            stacklevel=2,
        )
        if position_market_value <= 0:
            return False

        # 冷却期检查
        cooldown_until = self._cooldown_until.get(symbol, 0)
        if trading_day_index < cooldown_until:
            return False

        # 计算有效止损阈值
        effective_stop = self.config.stop_loss_pct
        if atr_val is not None and current_close > 0:
            atr_stop_pct = 2.0 * float(atr_val) / current_close
            effective_stop = max(self.config.stop_loss_pct, atr_stop_pct)

        loss_ratio = -float(position_pnl) / float(position_market_value)
        if loss_ratio > effective_stop:
            self._cooldown_until[symbol] = (
                trading_day_index + self.config.stop_loss_cooldown
            )
            _logger.info(
                "品种 %s 触发止损: 亏损率=%.2f%%, 阈值=%.2f%%",
                symbol, loss_ratio * 100, effective_stop * 100,
            )
            return True

        return False

    # -----------------------------------------------------------------------
    # 新接口（P0整改）：完整的复合止损（追踪+时间+固定）
    # -----------------------------------------------------------------------
    def check_composite_stop(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        current_price: float,
        highest_since_entry: float,
        lowest_since_entry: float,
        entry_day: int,
        current_day: int,
        atr_value: Optional[float] = None,
        auto_register_entry: bool = False,
    ) -> CompositeStopResult:
        """
        完整的复合止损检查（P0整改后的统一接口）。

        内部委托给 CompositeStopManager。
        优先级：固定止损 > 追踪止损 > 时间止损。

        Args:
            symbol: 品种代码
            direction: 持仓方向 "long" 或 "short"
            entry_price: 入场价
            current_price: 当前价
            highest_since_entry: 入场以来最高价（多头使用）
            lowest_since_entry: 入场以来最低价（空头使用）
            entry_day: 入场日索引
            current_day: 当前日索引
            atr_value: ATR 值
            auto_register_entry: 是否在首次调用时自动注册入场价/固定止损价
                - True：自动调用 set_entry 记录固定止损价
                - False（默认）：调用方需自行 set_entry

        Returns:
            CompositeStopResult 复合止损结果。
            未启用复合止损时（use_composite_stop=False），返回未触发的空结果。
        """
        if self._composite_stop is None:
            return CompositeStopResult(triggered=False, direction=direction)

        if auto_register_entry:
            # 自动注册入场价（计算固定止损价）
            self._composite_stop.set_entry(
                symbol=symbol,
                entry_price=entry_price,
                direction=direction,
            )

        if direction == "long":
            return self._composite_stop.check_long(
                symbol=symbol,
                entry_price=entry_price,
                current_price=current_price,
                highest_since_entry=highest_since_entry,
                entry_day=entry_day,
                current_day=current_day,
                atr_value=atr_value,
            )
        else:
            return self._composite_stop.check_short(
                symbol=symbol,
                entry_price=entry_price,
                current_price=current_price,
                lowest_since_entry=lowest_since_entry,
                entry_day=entry_day,
                current_day=current_day,
                atr_value=atr_value,
            )

    def set_position_entry(
        self,
        symbol: str,
        entry_price: float,
        direction: str,
    ) -> None:
        """
        注册持仓入场价（P0整改：暴露给执行器用于固定止损初始化）。

        Args:
            symbol: 品种代码
            entry_price: 入场价
            direction: 持仓方向
        """
        if self._composite_stop is not None:
            self._composite_stop.set_entry(
                symbol=symbol,
                entry_price=entry_price,
                direction=direction,
            )

    def clear_position(self, symbol: str, direction: Optional[str] = None) -> None:
        """
        清除品种的止损状态（持仓平仓后调用）。

        Args:
            symbol: 品种代码
            direction: 持仓方向，None 表示清除所有方向
        """
        if self._composite_stop is not None:
            self._composite_stop.clear(symbol, direction)

    # -----------------------------------------------------------------------
    # 持仓限制
    # -----------------------------------------------------------------------
    def check_position_limit(
        self,
        symbol: str,
        current_position_pct: float,
        total_position_pct: float,
    ) -> bool:
        """
        检查是否超过持仓限制。

        Args:
            symbol: 品种代码
            current_position_pct: 当前品种持仓占总权益比例
            total_position_pct: 总持仓占总权益比例

        Returns:
            True 表示超限
        """
        if current_position_pct > self.config.max_position_pct:
            _logger.warning(
                "品种 %s 仓位超限: %.2f%% > %.2f%%",
                symbol, current_position_pct * 100, self.config.max_position_pct * 100,
            )
            return True

        if total_position_pct > self.config.max_total_position_pct:
            _logger.warning(
                "总仓位超限: %.2f%% > %.2f%%",
                total_position_pct * 100, self.config.max_total_position_pct * 100,
            )
            return True

        return False

    def is_in_cooldown(self, symbol: str, trading_day_index: int) -> bool:
        """检查品种是否在止损冷却期内。"""
        cooldown_until = self._cooldown_until.get(symbol, 0)
        return trading_day_index < cooldown_until

    def clear_cooldown(self, symbol: str):
        """清除品种的冷却期。"""
        self._cooldown_until.pop(symbol, None)

    # -----------------------------------------------------------------------
    # P1-任务10整改：多品种整体风控
    # -----------------------------------------------------------------------
    def check_portfolio_risk(
        self,
        positions: Dict[str, Dict[str, float]],
        trading_day_index: int = 0,
    ) -> Dict[str, bool]:
        """
        批量检查多品种风控。

        P1-任务10整改：单品种风控是基础，多品种整体风控通过
        组合层面的相关性敞口、总杠杆、行业集中度等指标控制。

        Args:
            positions: {symbol: {"position_pct": float, "pnl": float, "market_value": float,
                                  "current_close": float, "atr": Optional[float]}}
            trading_day_index: 当前交易日序号

        Returns:
            {symbol: 是否触发风控止损/超限}
        """
        result: Dict[str, bool] = {}
        total_position_pct = 0.0
        # 第一轮：累计总仓位
        for sym, info in positions.items():
            total_position_pct += float(info.get("position_pct", 0.0))
        # 第二轮：单品种风控 + 总仓位校验
        for sym, info in positions.items():
            triggered = False
            position_market_value = float(info.get("market_value", 0.0))
            position_pnl = float(info.get("pnl", 0.0))
            current_close = float(info.get("current_close", 0.0))
            atr_val = info.get("atr")
            current_position_pct = float(info.get("position_pct", 0.0))

            # 1) 单品种止损
            if self.check_stop_loss(
                symbol=sym,
                position_market_value=position_market_value,
                position_pnl=position_pnl,
                current_close=current_close,
                atr_val=atr_val,
                trading_day_index=trading_day_index,
            ):
                triggered = True

            # 2) 单品种/总仓位超限
            if self.check_position_limit(
                symbol=sym,
                current_position_pct=current_position_pct,
                total_position_pct=total_position_pct,
            ):
                triggered = True

            result[sym] = triggered
        return result

    def check_concentration(
        self,
        positions: Dict[str, float],
        max_concentration: float = 0.4,
    ) -> List[str]:
        """
        检查品种集中度：单品种持仓占比不得超过 max_concentration。

        P1-任务10整改：新增多品种整体风控方法，
        防止单品种过度暴露。

        Args:
            positions: {symbol: 当前持仓市值}
            max_concentration: 单品种最大集中度（0-1）

        Returns:
            超过集中度阈值的品种列表
        """
        total = sum(abs(v) for v in positions.values())
        if total <= 0:
            return []
        over_concentrated: List[str] = []
        for sym, mv in positions.items():
            if abs(mv) / total > max_concentration:
                over_concentrated.append(sym)
        if over_concentrated:
            _logger.warning(
                "集中度超限品种: %s (阈值 %.2f%%)",
                over_concentrated, max_concentration * 100,
            )
        return over_concentrated

    def check_total_drawdown(
        self,
        current_equity: float,
        peak_equity: float,
        max_drawdown: float = 0.15,
    ) -> bool:
        """
        检查组合级别最大回撤。

        P1-任务10整改：组合层面风控，
        超过阈值时触发整体降仓/清仓。

        Args:
            current_equity: 当前权益
            peak_equity: 历史峰值权益
            max_drawdown: 最大回撤阈值（0-1）

        Returns:
            True 表示触发整体风控
        """
        if peak_equity <= 0 or current_equity <= 0:
            return False
        drawdown = (peak_equity - current_equity) / peak_equity
        if drawdown > max_drawdown:
            _logger.warning(
                "组合回撤超限: %.2f%% > %.2f%%, 触发整体风控",
                drawdown * 100, max_drawdown * 100,
            )
            return True
        return False
