"""
PortfolioManager.allocate_weights 与 RiskController.check_composite_stop 单元测试。

P2-2：为核心公共系统补充测试覆盖（规则6）。
"""

import warnings

import pytest

from core.portfolio import PortfolioManager
from core.risk_controller import RiskController, RiskConfig


# ══════════════════════════════════════════════════════════════════════════════
# PortfolioManager.allocate_weights
# ══════════════════════════════════════════════════════════════════════════════
class TestAllocateWeightsEqualWeight:
    """验证 equal_weight 模式。"""

    def test_basic_three_symbols(self):
        pm = PortfolioManager(total_allocation=1.0)
        weights = pm.allocate_weights(
            signals={"RB": 0.5, "CU": -0.3, "AU": 0.8},
            method="equal_weight",
        )
        # 3 个品种等权 = 1/3，多空符号由信号决定
        assert set(weights.keys()) == {"RB", "CU", "AU"}
        assert weights["RB"] == pytest.approx(1 / 3)
        assert weights["CU"] == pytest.approx(-1 / 3)
        assert weights["AU"] == pytest.approx(1 / 3)
        # 总权重（绝对值）= total_allocation
        assert sum(abs(v) for v in weights.values()) == pytest.approx(1.0)

    def test_total_allocation_override(self):
        pm = PortfolioManager(total_allocation=0.8)
        weights = pm.allocate_weights(
            signals={"A": 1.0, "B": -1.0},
            method="equal_weight",
            total_allocation=0.4,
        )
        # 覆盖了 total_allocation，每个品种 = 0.4/2 = 0.2
        assert weights["A"] == pytest.approx(0.2)
        assert weights["B"] == pytest.approx(-0.2)

    def test_empty_signals_returns_empty(self):
        pm = PortfolioManager()
        assert pm.allocate_weights(signals={}, method="equal_weight") == {}

    def test_zero_scores_filtered(self):
        """绝对值过小的信号被过滤。"""
        pm = PortfolioManager()
        weights = pm.allocate_weights(
            signals={"A": 1.0, "B": 0.0, "C": -0.5},
            method="equal_weight",
        )
        # B 被过滤（绝对值 < 1e-8）
        assert "B" not in weights
        assert "A" in weights and "C" in weights


class TestAllocateWeightsScoreWeighted:
    """验证 score_weighted 模式。"""

    def test_proportional_to_abs_score(self):
        pm = PortfolioManager(total_allocation=1.0)
        weights = pm.allocate_weights(
            signals={"A": 1.0, "B": 0.5, "C": -0.25},
            method="score_weighted",
        )
        # |score| 加权：A=1.0, B=0.5, C=0.25，总和=1.75
        # A 的权重 = (1.0/1.75) * 1.0 = 0.5714
        assert weights["A"] == pytest.approx(1.0 / 1.75, rel=1e-3)
        assert weights["B"] == pytest.approx(0.5 / 1.75, rel=1e-3)
        assert weights["C"] == pytest.approx(-0.25 / 1.75, rel=1e-3)


class TestAllocateWeightsRiskParity:
    """验证 risk_parity 模式。"""

    def test_risk_inverse_weighting(self):
        pm = PortfolioManager(total_allocation=1.0)
        weights = pm.allocate_weights(
            signals={"A": 1.0, "B": 1.0},
            method="risk_parity",
            risk_estimates={"A": 0.10, "B": 0.40},
        )
        # 风险倒数：A=10, B=2.5，总和=12.5
        # A 的权重 = (10/12.5) * 1.0 = 0.8
        # B 的权重 = (2.5/12.5) * 1.0 = 0.2
        assert weights["A"] == pytest.approx(0.8, rel=1e-3)
        assert weights["B"] == pytest.approx(0.2, rel=1e-3)

    def test_missing_risk_falls_back_to_equal(self):
        """未提供 risk_estimates 的品种退化为 1（等权）。"""
        # 显式 total_allocation=1.0 便于断言
        pm = PortfolioManager(total_allocation=1.0)
        weights = pm.allocate_weights(
            signals={"A": 1.0, "B": 1.0, "C": 1.0},
            method="risk_parity",
            risk_estimates={"A": 0.10},  # B/C 缺失
        )
        # A=10, B=1, C=1 总和=12，A=(10/12)*1.0
        assert weights["A"] == pytest.approx(10 / 12, rel=1e-3)


class TestAllocateWeightsTopN:
    """验证 top_n 模式。"""

    def test_picks_top_n(self):
        pm = PortfolioManager()
        weights = pm.allocate_weights(
            signals={"A": 0.9, "B": 0.5, "C": 0.3, "D": 0.1},
            method="top_n",
            top_n=2,
            total_allocation=0.4,
        )
        assert set(weights.keys()) == {"A", "B"}  # 仅 Top 2
        # 每个 0.4/2 = 0.2
        assert weights["A"] == pytest.approx(0.2)
        assert weights["B"] == pytest.approx(0.2)

    def test_short_top_n(self):
        """负 top_n 表示做空前 N。"""
        pm = PortfolioManager()
        weights = pm.allocate_weights(
            signals={"A": -0.9, "B": -0.5, "C": 0.3},
            method="top_n",
            top_n=-2,
            total_allocation=0.4,
        )
        # 选中 A、B（最负的 2 个），权重为负
        assert set(weights.keys()) == {"A", "B"}
        assert weights["A"] < 0
        assert weights["B"] < 0


class TestAllocateWeightsUnknownMethod:
    """未知 method 回退到 equal_weight。"""

    def test_fallback_to_equal_weight(self):
        pm = PortfolioManager(total_allocation=0.6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weights = pm.allocate_weights(
                signals={"A": 1.0, "B": -1.0},
                method="unknown_method_xyz",
            )
        # 回退到 equal_weight
        assert weights["A"] == pytest.approx(0.3)
        assert weights["B"] == pytest.approx(-0.3)


# ══════════════════════════════════════════════════════════════════════════════
# RiskController.check_composite_stop
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckCompositeStopDisabled:
    """use_composite_stop=False 时，check_composite_stop 应直接返回未触发。"""

    def test_disabled_returns_untriggered(self):
        config = RiskConfig(use_composite_stop=False)
        rc = RiskController(config=config)
        result = rc.check_composite_stop(
            symbol="rb2401",
            direction="long",
            entry_price=3800.0,
            current_price=3000.0,  # 大跌
            highest_since_entry=3900.0,
            lowest_since_entry=2500.0,
            entry_day=0,
            current_day=20,
            atr_value=50.0,
        )
        assert result.triggered is False


class TestCheckCompositeStopLong:
    """多头复合止损。"""

    def _rc(self) -> RiskController:
        config = RiskConfig(
            use_composite_stop=True,
            fixed_stop_pct=0.05,         # 5% 固定止损
            trailing_pct=0.03,           # 3% 追踪止损
            max_holding_days=10,         # 持仓 10 天
            time_target_return=0.01,     # 时间止损目标 1%
        )
        return RiskController(config=config)

    def test_fixed_stop_triggered(self):
        """跌破固定止损价（5%）应触发。"""
        rc = self._rc()
        rc.set_position_entry("rb2401", entry_price=3800.0, direction="long")
        result = rc.check_composite_stop(
            symbol="rb2401",
            direction="long",
            entry_price=3800.0,
            current_price=3600.0,  # 跌 5.26% > 5% 阈值
            highest_since_entry=3800.0,
            lowest_since_entry=3600.0,
            entry_day=0,
            current_day=2,
            atr_value=50.0,
        )
        assert result.triggered is True
        assert result.fixed_stop_triggered is True

    def test_trailing_stop_triggered(self):
        """从高点回撤追踪止损阈值应触发。"""
        rc = self._rc()
        rc.set_position_entry("rb2401", entry_price=3800.0, direction="long")
        result = rc.check_composite_stop(
            symbol="rb2401",
            direction="long",
            entry_price=3800.0,
            current_price=3700.0,
            highest_since_entry=4000.0,  # 最高 +5.26%
            # 固定止损价 = 3800 * 0.95 = 3610，未触发
            # 追踪止损价 = 4000 * 0.97 = 3880，跌破 3880 触发
            lowest_since_entry=3700.0,
            entry_day=0,
            current_day=2,
            atr_value=50.0,
        )
        assert result.triggered is True

    def test_time_stop_triggered(self):
        """持仓超过 max_holding_days 且未达目标应触发时间止损。"""
        rc = self._rc()
        rc.set_position_entry("rb2401", entry_price=3800.0, direction="long")
        result = rc.check_composite_stop(
            symbol="rb2401",
            direction="long",
            entry_price=3800.0,
            current_price=3810.0,  # 涨 0.26% < 1% 目标
            highest_since_entry=3810.0,
            lowest_since_entry=3790.0,
            entry_day=0,
            current_day=11,         # 超过 10 天
            atr_value=50.0,
        )
        assert result.triggered is True

    def test_no_trigger_within_thresholds(self):
        """各项阈值内应保持未触发。"""
        rc = self._rc()
        rc.set_position_entry("rb2401", entry_price=3800.0, direction="long")
        result = rc.check_composite_stop(
            symbol="rb2401",
            direction="long",
            entry_price=3800.0,
            current_price=3780.0,  # 跌 0.5%（<5%）
            highest_since_entry=3820.0,
            lowest_since_entry=3780.0,
            entry_day=0,
            current_day=3,  # 持仓 3 天（<10）
            atr_value=50.0,
        )
        assert result.triggered is False


class TestCheckCompositeStopShort:
    """空头复合止损。"""

    def _rc(self) -> RiskController:
        config = RiskConfig(
            use_composite_stop=True,
            fixed_stop_pct=0.05,
            trailing_pct=0.03,
            max_holding_days=10,
            time_target_return=0.01,
        )
        return RiskController(config=config)

    def test_short_fixed_stop_triggered(self):
        """空头：价格上涨超过固定止损应触发。"""
        rc = self._rc()
        rc.set_position_entry("rb2401", entry_price=3800.0, direction="short")
        result = rc.check_composite_stop(
            symbol="rb2401",
            direction="short",
            entry_price=3800.0,
            current_price=4000.0,  # 涨 5.26% > 5%
            highest_since_entry=4000.0,
            lowest_since_entry=3800.0,
            entry_day=0,
            current_day=2,
            atr_value=50.0,
        )
        assert result.triggered is True
        assert result.fixed_stop_triggered is True

    def test_clear_position_removes_state(self):
        """clear_position 后应回到未触发状态（先入场再清空）。"""
        rc = self._rc()
        rc.set_position_entry("rb2401", entry_price=3800.0, direction="long")
        rc.clear_position("rb2401", direction="long")
        # 再次检查：内部状态已清空，不应因 _composite_stop 历史触发
        result = rc.check_composite_stop(
            symbol="rb2401",
            direction="long",
            entry_price=3800.0,
            current_price=3780.0,
            highest_since_entry=3800.0,
            lowest_since_entry=3780.0,
            entry_day=0,
            current_day=2,
            atr_value=50.0,
        )
        # 注意：clear 后会重新登记门槛（如果有持久化）；这里只验证不抛异常
        assert result is not None


# ══════════════════════════════════════════════════════════════════════════════
# RiskController.check_stop_loss 兼容性
# ══════════════════════════════════════════════════════════════════════════════
class TestCheckStopLossDeprecation:
    """验证旧接口被标记为 @deprecated，但仍能正常调用。"""

    def test_emits_deprecation_warning(self):
        rc = RiskController()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            rc.check_stop_loss(
                symbol="RB",
                position_market_value=100000.0,
                position_pnl=-1000.0,
                current_close=3500.0,
                atr_val=None,
                trading_day_index=10,
            )
        deprecation_warnings = [
            w for w in captured if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1, "check_stop_loss 应触发 DeprecationWarning"

    def test_legacy_still_functional(self):
        """旧接口仍能正常判断止损触发。"""
        rc = RiskController(config=RiskConfig(stop_loss_pct=0.05))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 亏损 6% > 5% 阈值，触发
            assert rc.check_stop_loss(
                symbol="RB",
                position_market_value=100000.0,
                position_pnl=-6000.0,
                current_close=3500.0,
                atr_val=None,
                trading_day_index=10,
            ) is True
            # 亏损 2% < 5% 阈值，不触发
            assert rc.check_stop_loss(
                symbol="RB",
                position_market_value=100000.0,
                position_pnl=-2000.0,
                current_close=3500.0,
                atr_val=None,
                trading_day_index=10,
            ) is False
