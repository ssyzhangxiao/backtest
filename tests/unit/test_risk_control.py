"""
RiskController 单元测试。

验证止损、持仓限制、冷却期等风控逻辑。
"""

import pytest

from core.risk_controller import RiskController, RiskConfig


class TestStopLoss:
    """验证止损逻辑。"""

    def setup_method(self):
        self.ctrl = RiskController(RiskConfig(stop_loss_pct=0.05, stop_loss_cooldown=1))

    def test_triggers_stop_loss(self):
        """亏损超过阈值触发止损。"""
        result = self.ctrl.check_stop_loss(
            symbol="RB",
            position_market_value=100000,
            position_pnl=-6000,
            current_close=3500,
            trading_day_index=10,
        )
        assert result is True

    def test_no_stop_loss_within_threshold(self):
        """亏损未超阈值不触发止损。"""
        result = self.ctrl.check_stop_loss(
            symbol="RB",
            position_market_value=100000,
            position_pnl=-3000,
            current_close=3500,
            trading_day_index=10,
        )
        assert result is False

    def test_atr_dynamic_stop(self):
        """ATR 动态止损：2*ATR/Close > 固定止损时使用 ATR 止损。"""
        result = self.ctrl.check_stop_loss(
            symbol="RB",
            position_market_value=100000,
            position_pnl=-4000,
            current_close=3500,
            atr_val=100,
            trading_day_index=10,
        )
        # ATR止损 = 2*100/3500 ≈ 5.7%，大于固定5%
        # 亏损4% < 5.7%，不触发
        assert result is False

    def test_zero_market_value(self):
        """市值为0时不触发止损。"""
        result = self.ctrl.check_stop_loss(
            symbol="RB",
            position_market_value=0,
            position_pnl=0,
            current_close=3500,
            trading_day_index=10,
        )
        assert result is False


class TestCooldown:
    """验证冷却期逻辑。"""

    def setup_method(self):
        self.ctrl = RiskController(RiskConfig(stop_loss_pct=0.05, stop_loss_cooldown=3))

    def test_cooldown_after_stop_loss(self):
        """止损后进入冷却期。"""
        self.ctrl.check_stop_loss(
            symbol="RB",
            position_market_value=100000,
            position_pnl=-6000,
            current_close=3500,
            trading_day_index=10,
        )
        # cooldown_until = 10 + 3 = 13，trading_day_index < 13 时在冷却期
        assert self.ctrl.is_in_cooldown("RB", 11) is True
        assert self.ctrl.is_in_cooldown("RB", 12) is True
        assert self.ctrl.is_in_cooldown("RB", 13) is False

    def test_no_cooldown_without_stop_loss(self):
        """未触发止损时无冷却期。"""
        assert self.ctrl.is_in_cooldown("RB", 10) is False

    def test_clear_cooldown(self):
        """手动清除冷却期。"""
        self.ctrl.check_stop_loss(
            symbol="RB",
            position_market_value=100000,
            position_pnl=-6000,
            current_close=3500,
            trading_day_index=10,
        )
        self.ctrl.clear_cooldown("RB")
        assert self.ctrl.is_in_cooldown("RB", 11) is False


class TestPositionLimit:
    """验证持仓限制逻辑。"""

    def setup_method(self):
        self.ctrl = RiskController(RiskConfig(
            max_position_pct=0.2,
            max_total_position_pct=0.6,
        ))

    def test_single_position_over_limit(self):
        """单品种仓位超限。"""
        result = self.ctrl.check_position_limit(
            symbol="RB",
            current_position_pct=0.25,
            total_position_pct=0.3,
        )
        assert result is True

    def test_total_position_over_limit(self):
        """总仓位超限。"""
        result = self.ctrl.check_position_limit(
            symbol="RB",
            current_position_pct=0.15,
            total_position_pct=0.65,
        )
        assert result is True

    def test_within_limits(self):
        """仓位在限制内。"""
        result = self.ctrl.check_position_limit(
            symbol="RB",
            current_position_pct=0.15,
            total_position_pct=0.5,
        )
        assert result is False
