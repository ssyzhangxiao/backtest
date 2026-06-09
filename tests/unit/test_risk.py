"""
风控模块单元测试。

规则6要求：新增风控规则必须有触发条件测试。
覆盖：追踪止损（pct/atr模式）、时间止损、复合止损优先级。
"""

import unittest
import numpy as np

from core.risk.trailing_stop import TrailingStop, TrailingStopResult
from core.risk.time_stop import TimeStop, TimeStopResult
from core.risk.composite_stop import CompositeStopManager, CompositeStopResult


class TestTrailingStop(unittest.TestCase):
    """追踪止损测试。"""

    def test_long_pct_mode_not_triggered(self):
        """多头百分比模式：价格在止损线上方，不应触发。"""
        ts = TrailingStop(mode="pct", trail_pct=0.03)
        result = ts.check_long(
            symbol="rb2401",
            entry_price=3800,
            current_price=3900,
            highest_since_entry=3950,
        )
        self.assertFalse(result.triggered)
        # 止损价 = 3950 * (1 - 0.03) = 3831.5
        self.assertAlmostEqual(result.stop_price, 3950 * 0.97, places=1)

    def test_long_pct_mode_triggered(self):
        """多头百分比模式：价格跌破止损线，应触发。"""
        ts = TrailingStop(mode="pct", trail_pct=0.03)
        result = ts.check_long(
            symbol="rb2401",
            entry_price=3800,
            current_price=3700,
            highest_since_entry=3950,
        )
        # 止损价 = 3950 * 0.97 = 3831.5，当前3700 < 3831.5
        self.assertTrue(result.triggered)

    def test_long_atr_mode_triggered(self):
        """多头ATR模式：价格跌破最高价-N*ATR，应触发。"""
        ts = TrailingStop(mode="atr", atr_multiplier=2.0)
        result = ts.check_long(
            symbol="rb2401",
            entry_price=3800,
            current_price=3840,
            highest_since_entry=3950,
            atr_value=50.0,
        )
        # 止损价 = 3950 - 2*50 = 3850，当前3840 < 3850
        self.assertTrue(result.triggered)

    def test_long_atr_mode_not_triggered(self):
        """多头ATR模式：价格在止损线上方，不应触发。"""
        ts = TrailingStop(mode="atr", atr_multiplier=2.0)
        result = ts.check_long(
            symbol="rb2401",
            entry_price=3800,
            current_price=3860,
            highest_since_entry=3950,
            atr_value=50.0,
        )
        # 止损价 = 3950 - 100 = 3850，当前3860 > 3850
        self.assertFalse(result.triggered)

    def test_short_pct_mode_triggered(self):
        """空头百分比模式：价格涨破止损线，应触发。"""
        ts = TrailingStop(mode="pct", trail_pct=0.03)
        result = ts.check_short(
            symbol="rb2401",
            entry_price=3900,
            current_price=4000,
            lowest_since_entry=3750,
        )
        # 止损价 = 3750 * (1 + 0.03) = 3862.5，当前4000 > 3862.5
        self.assertTrue(result.triggered)

    def test_zero_trail_distance(self):
        """追踪距离为0时，止损价等于最高/最低价。"""
        ts = TrailingStop(mode="pct", trail_pct=0.0)
        result = ts.check_long(
            symbol="test",
            entry_price=100,
            current_price=99,
            highest_since_entry=105,
        )
        # 止损价 = 105 * 1.0 = 105，当前99 < 105
        self.assertTrue(result.triggered)


class TestTimeStop(unittest.TestCase):
    """时间止损测试。"""

    def test_not_triggered_within_limit(self):
        """持仓天数未超限，不应触发。"""
        ts = TimeStop(max_holding_days=10, target_return=0.01)
        result = ts.check(
            entry_day=100,
            current_day=105,
            entry_price=100,
            current_price=100.5,
        )
        self.assertFalse(result.triggered)
        self.assertEqual(result.holding_days, 5)

    def test_triggered_exceeds_limit(self):
        """持仓天数超限且收益未达标，应触发。"""
        ts = TimeStop(max_holding_days=10, target_return=0.01)
        result = ts.check(
            entry_day=100,
            current_day=115,
            entry_price=100,
            current_price=100.5,
        )
        # 持仓15天 > 10天，收益0.5% < 1%
        self.assertTrue(result.triggered)

    def test_not_triggered_target_reached(self):
        """持仓天数超限但收益达标，不应触发。"""
        ts = TimeStop(max_holding_days=10, target_return=0.01)
        result = ts.check(
            entry_day=100,
            current_day=115,
            entry_price=100,
            current_price=102,
        )
        # 持仓15天 > 10天，但收益2% > 1%
        self.assertFalse(result.triggered)

    def test_short_direction_time_stop(self):
        """空头方向的时间止损。"""
        ts = TimeStop(max_holding_days=10, target_return=0.01)
        result = ts.check(
            entry_day=100,
            current_day=115,
            entry_price=100,
            current_price=98,
            direction="short",
        )
        # 空头收益2% > 1%，不应触发
        self.assertFalse(result.triggered)

    def test_max_holding_days_clamped(self):
        """最大持仓天数应限制在3~20范围内。"""
        ts1 = TimeStop(max_holding_days=1)
        self.assertEqual(ts1.max_holding_days, 3)
        ts2 = TimeStop(max_holding_days=50)
        self.assertEqual(ts2.max_holding_days, 20)


class TestCompositeStopManager(unittest.TestCase):
    """复合止损管理器测试。"""

    def setUp(self):
        self.manager = CompositeStopManager(
            fixed_stop_pct=0.05,
            trailing_mode="pct",
            trailing_pct=0.03,
            max_holding_days=10,
            time_target_return=0.01,
        )

    def test_no_stop_triggered(self):
        """所有止损条件均未满足。"""
        self.manager.set_entry("rb2401", 3800)
        result = self.manager.check_long(
            symbol="rb2401",
            entry_price=3800,
            current_price=3900,
            highest_since_entry=3950,
            entry_day=0,
            current_day=5,
        )
        self.assertFalse(result.triggered)

    def test_fixed_stop_triggered_first(self):
        """固定止损优先级最高，应最先触发。"""
        self.manager.set_entry("rb2401", 3800)
        result = self.manager.check_long(
            symbol="rb2401",
            entry_price=3800,
            current_price=3590,  # 跌幅5.5% > 5%
            highest_since_entry=3950,
            entry_day=0,
            current_day=5,
        )
        self.assertTrue(result.triggered)
        self.assertTrue(result.fixed_stop_triggered)

    def test_trailing_stop_triggered(self):
        """追踪止损触发（固定止损未触发时）。"""
        self.manager.set_entry("rb2401", 3800)
        result = self.manager.check_long(
            symbol="rb2401",
            entry_price=3800,
            current_price=3820,  # 跌幅0.5% < 5%，固定止损不触发
            highest_since_entry=3950,
            entry_day=0,
            current_day=5,
        )
        # 追踪止损价 = 3950 * 0.97 = 3831.5，当前3820 < 3831.5
        self.assertTrue(result.triggered)
        self.assertFalse(result.fixed_stop_triggered)

    def test_time_stop_triggered(self):
        """时间止损触发（固定和追踪止损均未触发时）。"""
        self.manager.set_entry("rb2401", 3800)
        result = self.manager.check_long(
            symbol="rb2401",
            entry_price=3800,
            current_price=3805,  # 微利，追踪止损不触发
            highest_since_entry=3810,
            entry_day=0,
            current_day=15,  # 持仓15天 > 10天
        )
        # 追踪止损价 = 3810 * 0.97 = 3695.7，当前3805 > 3695.7
        # 时间止损：15天 > 10天，收益0.13% < 1%
        self.assertTrue(result.triggered)

    def test_priority_fixed_over_trailing(self):
        """固定止损优先级高于追踪止损。"""
        self.manager.set_entry("rb2401", 3800)
        # 同时满足固定止损和追踪止损条件
        result = self.manager.check_long(
            symbol="rb2401",
            entry_price=3800,
            current_price=3550,  # 跌幅6.6% > 5%
            highest_since_entry=3900,
            entry_day=0,
            current_day=5,
        )
        self.assertTrue(result.triggered)
        # 固定止损应被标记
        self.assertTrue(result.fixed_stop_triggered)


if __name__ == "__main__":
    unittest.main()
