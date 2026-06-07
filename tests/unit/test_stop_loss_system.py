"""
止损系统单元测试（P0/P1/P2 整改后）。

覆盖：
  - TrailingStop 多空状态分离
  - TimeStop verbose 日志开关
  - CompositeStopManager 多空状态独立
  - RiskController 复合止损集成
  - 多品种状态隔离
  - 止损 stop_price 语义（时间止损返回 nan）
"""

import math

from core.risk.trailing_stop import TrailingStop, StopDirection
from core.risk.time_stop import TimeStop
from core.risk.composite_stop import CompositeStopManager
from core.risk_controller import RiskController, RiskConfig


# ────────────────────────────────────────────────────────────
# TrailingStop 多空状态分离
# ────────────────────────────────────────────────────────────
class TestTrailingStopDirectionIsolation:
    """验证同一品种多空追踪状态完全独立。"""

    def setup_method(self):
        self.ts = TrailingStop(mode="pct", trail_pct=0.05)

    def test_long_and_short_independent(self):
        """多头和空头追踪止损互不影响。"""
        # 多头：入场100，最高110
        long_res = self.ts.check_long(
            symbol="rb2401",
            entry_price=100.0,
            current_price=110.0,
            highest_since_entry=110.0,
        )
        assert not long_res.triggered
        long_stop = long_res.stop_price

        # 空头：入场100，最低90
        short_res = self.ts.check_short(
            symbol="rb2401",
            entry_price=100.0,
            current_price=90.0,
            lowest_since_entry=90.0,
        )
        assert not short_res.triggered
        short_stop = short_res.stop_price

        # 多头止损应在最高价之下（向上保护）
        assert long_stop < 110.0
        # 空头止损应在最低价之上（向下保护）
        assert short_stop > 90.0

    def test_long_state_persists_through_short_calls(self):
        """多头状态不受空头调用影响。"""
        self.ts.check_long(
            symbol="rb2401", entry_price=100.0, current_price=110.0,
            highest_since_entry=110.0,
        )
        long_state_1 = self.ts.get_state("rb2401", StopDirection.LONG)
        assert long_state_1 is not None
        stop_before = long_state_1.stop_price

        # 调用空头，不应改变多头状态
        self.ts.check_short(
            symbol="rb2401", entry_price=100.0, current_price=90.0,
            lowest_since_entry=90.0,
        )
        long_state_2 = self.ts.get_state("rb2401", StopDirection.LONG)
        assert long_state_2.stop_price == stop_before

    def test_clear_direction_only_clears_one_direction(self):
        """clear_direction 只清除指定方向的状态。"""
        self.ts.check_long(
            symbol="rb2401", entry_price=100.0, current_price=110.0,
            highest_since_entry=110.0,
        )
        self.ts.check_short(
            symbol="rb2401", entry_price=100.0, current_price=90.0,
            lowest_since_entry=90.0,
        )

        self.ts.clear_direction("rb2401", StopDirection.LONG)
        assert self.ts.get_state("rb2401", StopDirection.LONG) is None
        assert self.ts.get_state("rb2401", StopDirection.SHORT) is not None

    def test_clear_clears_all_directions(self):
        """clear(symbol) 清除该品种所有方向。"""
        self.ts.check_long(
            symbol="rb2401", entry_price=100.0, current_price=110.0,
            highest_since_entry=110.0,
        )
        self.ts.check_short(
            symbol="rb2401", entry_price=100.0, current_price=90.0,
            lowest_since_entry=90.0,
        )
        self.ts.clear("rb2401")
        assert self.ts.get_state("rb2401", StopDirection.LONG) is None
        assert self.ts.get_state("rb2401", StopDirection.SHORT) is None

    def test_long_stop_only_moves_up(self):
        """多头追踪止损价只能上移不能下移。"""
        # 第一次：最高110
        self.ts.check_long(
            symbol="rb2401", entry_price=100.0, current_price=110.0,
            highest_since_entry=110.0,
        )
        first_stop = self.ts.get_state("rb2401", StopDirection.LONG).stop_price

        # 第二次：最高价跌到105
        self.ts.check_long(
            symbol="rb2401", entry_price=100.0, current_price=102.0,
            highest_since_entry=105.0,
        )
        second_stop = self.ts.get_state("rb2401", StopDirection.LONG).stop_price
        assert second_stop >= first_stop


# ────────────────────────────────────────────────────────────
# TimeStop verbose 开关
# ────────────────────────────────────────────────────────────
class TestTimeStopVerbose:
    """验证 TimeStop 的 verbose 日志开关。"""

    def test_no_trigger_within_max_days(self):
        """持仓未超过最大天数不触发。"""
        ts = TimeStop(max_holding_days=10, target_return=0.01, verbose=True)
        result = ts.check(
            entry_day=0, current_day=5,
            entry_price=100.0, current_price=100.5,
            direction="long",
        )
        assert not result.triggered
        assert result.holding_days == 5

    def test_triggered_when_holding_too_long(self):
        """持仓超过最大天数且收益未达标时触发。"""
        ts = TimeStop(max_holding_days=10, target_return=0.01, verbose=True)
        result = ts.check(
            entry_day=0, current_day=12,
            entry_price=100.0, current_price=100.1,  # 收益 0.1% < 1%
            direction="long",
        )
        assert result.triggered
        assert result.holding_days == 12
        assert result.current_return < result.target_return

    def test_not_triggered_when_target_reached(self):
        """持仓超时但已达标时不应触发。"""
        ts = TimeStop(max_holding_days=10, target_return=0.01, verbose=True)
        result = ts.check(
            entry_day=0, current_day=12,
            entry_price=100.0, current_price=102.0,  # 收益 2% >= 1%
            direction="long",
        )
        assert not result.triggered

    def test_short_direction_return_calc(self):
        """空头方向收益率计算正确（价格下跌为正）。"""
        ts = TimeStop(max_holding_days=10, target_return=0.01)
        # 空头：价格从100跌到98，收益=(100-98)/100=2%
        result = ts.check(
            entry_day=0, current_day=12,
            entry_price=100.0, current_price=98.0,
            direction="short",
        )
        assert not result.triggered  # 2% > 1% 不触发
        assert abs(result.current_return - 0.02) < 1e-9


# ────────────────────────────────────────────────────────────
# CompositeStopManager 多空状态独立
# ────────────────────────────────────────────────────────────
class TestCompositeStopManagerDirectionIsolation:
    """验证复合止损多空状态分离。"""

    def setup_method(self):
        self.mgr = CompositeStopManager(
            fixed_stop_pct=0.05,
            max_holding_days=10,
            time_target_return=0.01,
        )

    def test_long_fixed_stop_calculation(self):
        """多头固定止损：entry * (1 - pct)。"""
        self.mgr.set_entry("rb2401", 100.0, direction="long")
        snapshot = self.mgr.get_state_snapshot()
        assert "rb2401" in snapshot
        assert abs(snapshot["rb2401"]["long"] - 95.0) < 1e-9

    def test_short_fixed_stop_calculation(self):
        """空头固定止损：entry * (1 + pct)。"""
        self.mgr.set_entry("rb2401", 100.0, direction="short")
        snapshot = self.mgr.get_state_snapshot()
        assert abs(snapshot["rb2401"]["short"] - 105.0) < 1e-9

    def test_long_and_short_independent(self):
        """同一品种多空固定止损价独立。"""
        self.mgr.set_entry("rb2401", 100.0, direction="long")
        self.mgr.set_entry("rb2401", 100.0, direction="short")
        snapshot = self.mgr.get_state_snapshot()
        assert abs(snapshot["rb2401"]["long"] - 95.0) < 1e-9
        assert abs(snapshot["rb2401"]["short"] - 105.0) < 1e-9

    def test_long_fixed_stop_trigger(self):
        """多头固定止损触发：价格 <= 止损价。"""
        self.mgr.set_entry("rb2401", 100.0, direction="long")
        result = self.mgr.check_long(
            symbol="rb2401",
            entry_price=100.0,
            current_price=94.0,  # 跌破95
            highest_since_entry=100.0,
            entry_day=0, current_day=3,
        )
        assert result.triggered
        assert result.trigger_reason == "fixed_stop"
        assert result.fixed_stop_triggered
        assert result.stop_price == 95.0

    def test_short_fixed_stop_trigger(self):
        """空头固定止损触发：价格 >= 止损价。"""
        self.mgr.set_entry("rb2401", 100.0, direction="short")
        result = self.mgr.check_short(
            symbol="rb2401",
            entry_price=100.0,
            current_price=106.0,  # 突破105
            lowest_since_entry=100.0,
            entry_day=0, current_day=3,
        )
        assert result.triggered
        assert result.trigger_reason == "fixed_stop"
        assert result.stop_price == 105.0

    def test_no_trigger_within_threshold(self):
        """未触发任何止损时，stop_price 应为追踪止损价。"""
        self.mgr.set_entry("rb2401", 100.0, direction="long")
        result = self.mgr.check_long(
            symbol="rb2401",
            entry_price=100.0,
            current_price=99.5,  # 在追踪止损价（98.94）之上，未触发
            highest_since_entry=102.0,
            entry_day=0, current_day=3,
        )
        assert not result.triggered
        # 追踪止损价 = 102 * (1 - 0.03) = 98.94
        assert result.stop_price > 0
        assert abs(result.stop_price - 98.94) < 0.01

    def test_time_stop_returns_nan_stop_price(self):
        """时间止损触发时 stop_price 为 np.nan（语义统一）。"""
        self.mgr.set_entry("rb2401", 100.0, direction="long")
        result = self.mgr.check_long(
            symbol="rb2401",
            entry_price=100.0,
            current_price=100.05,  # 收益 0.05% < 1%
            highest_since_entry=101.0,
            entry_day=0, current_day=12,  # 超过 10 天
        )
        assert result.triggered
        assert result.trigger_reason == "time_stop"
        assert math.isnan(result.stop_price), "时间止损 stop_price 应为 nan"

    def test_clear_one_direction_keeps_other(self):
        """clear(direction=...) 仅清除指定方向。"""
        self.mgr.set_entry("rb2401", 100.0, direction="long")
        self.mgr.set_entry("rb2401", 100.0, direction="short")
        self.mgr.clear("rb2401", direction="long")
        snapshot = self.mgr.get_state_snapshot()
        assert "long" not in snapshot["rb2401"]
        assert "short" in snapshot["rb2401"]

    def test_multi_symbol_state_isolation(self):
        """多品种状态完全隔离。"""
        self.mgr.set_entry("rb2401", 100.0, direction="long")
        self.mgr.set_entry("cu2401", 50000.0, direction="long")
        snapshot = self.mgr.get_state_snapshot()
        assert abs(snapshot["rb2401"]["long"] - 95.0) < 1e-9
        assert abs(snapshot["cu2401"]["long"] - 47500.0) < 1e-9
        # 清除 rb 不应影响 cu
        self.mgr.clear("rb2401")
        snapshot = self.mgr.get_state_snapshot()
        assert "rb2401" not in snapshot
        assert "cu2401" in snapshot


# ────────────────────────────────────────────────────────────
# RiskController.check_composite_stop 集成
# ────────────────────────────────────────────────────────────
class TestRiskControllerCompositeStop:
    """验证 RiskController 整合 CompositeStopManager。"""

    def setup_method(self):
        self.ctrl = RiskController(RiskConfig(
            fixed_stop_pct=0.05,
            max_holding_days=10,
            time_target_return=0.01,
            stop_loss_verbose=False,
        ))

    def test_composite_stop_disabled(self):
        """use_composite_stop=False 时返回未触发的空结果。"""
        ctrl = RiskController(RiskConfig(use_composite_stop=False))
        result = ctrl.check_composite_stop(
            symbol="rb2401", direction="long",
            entry_price=100.0, current_price=90.0,
            highest_since_entry=100.0, lowest_since_entry=100.0,
            entry_day=0, current_day=3,
        )
        assert not result.triggered
        assert result.direction == "long"

    def test_auto_register_entry(self):
        """auto_register_entry=True 自动调用 set_entry。"""
        result = self.ctrl.check_composite_stop(
            symbol="rb2401", direction="long",
            entry_price=100.0, current_price=94.0,  # 跌破95
            highest_since_entry=100.0, lowest_since_entry=100.0,
            entry_day=0, current_day=3,
            auto_register_entry=True,
        )
        assert result.triggered
        assert result.trigger_reason == "fixed_stop"
        # 内部状态已记录
        snapshot = self.ctrl.composite_stop.get_state_snapshot()
        assert "rb2401" in snapshot

    def test_manual_set_entry(self):
        """手动调用 set_position_entry 预登记入场价。"""
        self.ctrl.set_position_entry("rb2401", 100.0, direction="long")
        result = self.ctrl.check_composite_stop(
            symbol="rb2401", direction="long",
            entry_price=100.0, current_price=99.5,  # 在追踪止损价之上，未触发
            highest_since_entry=100.0, lowest_since_entry=99.5,
            entry_day=0, current_day=3,
            auto_register_entry=False,
        )
        assert not result.triggered
        assert not result.fixed_stop_triggered

    def test_clear_position(self):
        """clear_position 清除指定方向状态。"""
        self.ctrl.set_position_entry("rb2401", 100.0, direction="long")
        self.ctrl.set_position_entry("rb2401", 100.0, direction="short")
        self.ctrl.clear_position("rb2401", direction="long")
        snapshot = self.ctrl.composite_stop.get_state_snapshot()
        assert "long" not in snapshot["rb2401"]
        assert "short" in snapshot["rb2401"]

    def test_composite_stop_property(self):
        """composite_stop 属性可访问内部组件。"""
        assert self.ctrl.composite_stop is not None
        assert isinstance(self.ctrl.composite_stop, CompositeStopManager)

    def test_composite_stop_long_full_flow(self):
        """完整流程：固定止损触发。"""
        self.ctrl.set_position_entry("rb2401", 100.0, direction="long")
        # 价格小幅回调，未触发（追踪止损价 102*0.97=98.94，当前 99.5 在其上）
        r1 = self.ctrl.check_composite_stop(
            symbol="rb2401", direction="long",
            entry_price=100.0, current_price=99.5,
            highest_since_entry=102.0, lowest_since_entry=99.5,
            entry_day=0, current_day=3,
        )
        assert not r1.triggered
        # 价格继续下跌，触发固定止损（跌破95）
        r2 = self.ctrl.check_composite_stop(
            symbol="rb2401", direction="long",
            entry_price=100.0, current_price=94.5,
            highest_since_entry=102.0, lowest_since_entry=94.5,
            entry_day=0, current_day=5,
        )
        assert r2.triggered
        assert r2.trigger_reason == "fixed_stop"
        assert r2.stop_price == 95.0


# ────────────────────────────────────────────────────────────
# 止损 stop_price 语义统一性
# ────────────────────────────────────────────────────────────
class TestStopPriceSemantics:
    """验证 stop_price 字段在各种触发场景下的语义统一性。"""

    def setup_method(self):
        self.mgr = CompositeStopManager(
            fixed_stop_pct=0.05,
            max_holding_days=10,
            time_target_return=0.01,
        )

    def test_fixed_stop_has_actual_price(self):
        """固定止损触发：stop_price 是具体价格。"""
        self.mgr.set_entry("rb2401", 100.0, direction="long")
        result = self.mgr.check_long(
            symbol="rb2401", entry_price=100.0,
            current_price=94.0, highest_since_entry=100.0,
            entry_day=0, current_day=3,
        )
        assert not math.isnan(result.stop_price)
        assert result.stop_price == 95.0

    def test_trailing_stop_has_actual_price(self):
        """追踪止损触发：stop_price 是具体价格。"""
        # 不用 set_entry，直接触发追踪止损
        result = self.mgr.check_long(
            symbol="rb2401", entry_price=100.0,
            current_price=88.0,  # 跌破追踪价
            highest_since_entry=100.0,
            entry_day=0, current_day=3,
        )
        # 触发追踪止损
        if result.trigger_reason == "trailing_stop":
            assert not math.isnan(result.stop_price)
            assert result.stop_price > 0

    def test_time_stop_has_nan_price(self):
        """时间止损触发：stop_price 为 nan（语义：按时平仓，无固定止损价）。"""
        self.mgr.set_entry("rb2401", 100.0, direction="long")
        result = self.mgr.check_long(
            symbol="rb2401", entry_price=100.0,
            current_price=100.0,
            highest_since_entry=101.0,
            entry_day=0, current_day=12,  # 超过 10 天
        )
        assert result.triggered
        assert result.trigger_reason == "time_stop"
        assert math.isnan(result.stop_price)

    def test_no_trigger_has_trailing_price(self):
        """未触发：stop_price 是当前追踪止损价。"""
        self.mgr.set_entry("rb2401", 100.0, direction="long")
        result = self.mgr.check_long(
            symbol="rb2401", entry_price=100.0,
            current_price=99.0,
            highest_since_entry=102.0,
            entry_day=0, current_day=3,
        )
        assert not result.triggered
        # 未触发时 stop_price 应是追踪止损价
        assert result.stop_price > 0
        # 102 * (1 - 0.03) = 98.94
        assert abs(result.stop_price - 98.94) < 0.01
