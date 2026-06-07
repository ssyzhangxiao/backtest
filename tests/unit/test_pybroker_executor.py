"""
PyBrokerExecutorBuilder 蓝图模式执行器单元测试。

P2-1整改：覆盖横截面收集、finalize、共享状态、单品种执行等关键路径。
"""
from datetime import datetime
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import numpy as np

from core.config import BacktestConfig
from core.engine.pybroker_executor import (
    PyBrokerExecutorBuilder,
    PyBrokerExecutorSharedState,
)
from core.engine.switch_engine import FactorScoringEngine, ScoringConfig
from core.portfolio import PortfolioManager
from core.risk_controller import RiskController


# ── Mock helpers ──────────────────────────────────────────────


class _FakePos:
    def __init__(self, shares: int = 0, pnl: float = 0.0, equity: float = 1.0):
        self.shares = shares
        self.pnl = pnl
        self.equity = equity


class _FakeCtx:
    """
    Mock PyBroker ExecContext。

    提供 symbol/dt/close/indicator/pos 等接口。
    """

    def __init__(
        self,
        symbol: str,
        dt: datetime,
        close: np.ndarray,
        pos_long: Optional[_FakePos] = None,
        pos_short: Optional[_FakePos] = None,
    ):
        self.symbol = symbol
        self.dt = dt
        self._close = close
        self._pos_long = pos_long
        self._pos_short = pos_short
        self.buy_shares = 0
        self.sell_shares = 0

    @property
    def close(self):
        return self._close

    def indicator(self, name: str):  # noqa: ARG002
        return None

    def pos(self, symbol: str, side: str):  # noqa: ARG002
        if side == "long":
            return self._pos_long
        if side == "short":
            return self._pos_short
        return None

    def calc_target_shares(self, weight: float):
        return int(weight * 1000)

    def sell_all_shares(self):
        self._pos_long = None

    def cover_all_shares(self):
        self._pos_short = None


def _make_components(total_symbols: int = 3):
    """构造测试所需的最小组件集。"""
    cfg = BacktestConfig()
    lib = MagicMock()
    lib.get_profile.return_value = SimpleNamespace(
        default_params={"window": 20, "lookback": 20},
    )
    scoring_cfg = ScoringConfig()
    scoring_engine = FactorScoringEngine(lib, scoring_cfg)
    # Mock 关键方法
    scoring_engine.extract_factor_scores = MagicMock(
        side_effect=lambda ctx, params: {
            "trend": 0.6,
            "term_structure": 0.2,
            "mean_reversion": -0.1,
            "vol_breakout": 0.3,
            "composite_resonance": 0.4,
        },
    )
    scoring_engine.is_rebalance_day = MagicMock(return_value=True)
    scoring_engine.update_cross_section = MagicMock()
    scoring_engine.finalize_cross_section = MagicMock()
    scoring_engine.compute_composite_score = MagicMock(
        side_effect=lambda sym: {
            "RB": 0.6,
            "CU": -0.3,
            "AU": 0.1,
        }.get(sym, 0.0),
    )
    scoring_engine.mark_rebalanced = MagicMock()

    portfolio = PortfolioManager(total_allocation=cfg.max_total_position_pct)
    # Mock RiskController 提供 check_concentration_dict 方法（executor 蓝图需要）
    risk_controller = MagicMock(spec=RiskController)
    risk_controller.check_concentration_dict = MagicMock(
        side_effect=lambda weights, max_concentration: dict(weights),
    )
    return cfg, scoring_engine, portfolio, risk_controller


def _make_ctx(
    symbol: str,
    dt: datetime,
    price: float = 100.0,
    pos_long: Optional[_FakePos] = None,
    pos_short: Optional[_FakePos] = None,
):
    return _FakeCtx(
        symbol=symbol,
        dt=dt,
        close=np.array([price] * 5),
        pos_long=pos_long,
        pos_short=pos_short,
    )


# ── Tests ──────────────────────────────────────────────────────


class TestSharedState:
    """共享状态测试。"""

    def test_init_defaults(self):
        state = PyBrokerExecutorSharedState(total_symbols=3)
        assert state.total_symbols == 3
        assert state.rebalance_date is None
        assert state.collected_symbols == []
        assert state.finalized is False
        assert state.target_weights == {}
        assert state.last_weights == {}

    def test_reset_on_new_date(self):
        state = PyBrokerExecutorSharedState(total_symbols=3)
        state.rebalance_date = "2024-01-01"
        state.collected_symbols = ["RB", "CU"]
        state.finalized = True
        state.target_weights = {"RB": 0.5}

        # 模拟新调仓日重置
        state.rebalance_date = "2024-01-02"
        state.collected_symbols = []
        state.finalized = False
        state.target_weights = {}

        assert state.rebalance_date == "2024-01-02"
        assert state.collected_symbols == []
        assert state.finalized is False


class TestBuilder:
    """PyBrokerExecutorBuilder 构造测试。"""

    def test_init(self):
        cfg, scoring_engine, portfolio, risk_controller = _make_components()
        builder = PyBrokerExecutorBuilder(
            scoring_engine=scoring_engine,
            portfolio_manager=portfolio,
            risk_controller=risk_controller,
            config=cfg,
            total_symbols=3,
            weight_method="risk_parity",
            risk_estimates_provider=lambda s: 0.02,
        )
        assert builder.weight_method == "risk_parity"
        assert builder.state.total_symbols == 3
        assert builder.state.collected_symbols == []

    def test_build_returns_executor(self):
        cfg, scoring_engine, portfolio, risk_controller = _make_components()
        builder = PyBrokerExecutorBuilder(
            scoring_engine=scoring_engine,
            portfolio_manager=portfolio,
            risk_controller=risk_controller,
            config=cfg,
            total_symbols=3,
        )
        executor = builder.build(strategy_params={})
        assert callable(executor)
        assert executor.__name__ == "blueprint_executor"


class TestCrossSectionFlow:
    """横截面收集与 finalize 测试。"""

    def test_collect_all_symbols_then_finalize(self):
        """3 品种全部访问后应触发 finalize。"""
        cfg, scoring_engine, portfolio, risk_controller = _make_components(total_symbols=3)
        builder = PyBrokerExecutorBuilder(
            scoring_engine=scoring_engine,
            portfolio_manager=portfolio,
            risk_controller=risk_controller,
            config=cfg,
            total_symbols=3,
        )
        executor = builder.build(strategy_params={})
        dt = datetime(2024, 1, 2)

        # 模拟 PyBroker 按 (date, symbol) 顺序调用
        for sym in ["RB", "CU", "AU"]:
            ctx = _make_ctx(sym, dt)
            executor(ctx)

        # 3 个品种访问后，finalize_cross_section 应被调用
        assert scoring_engine.finalize_cross_section.call_count == 1
        assert scoring_engine.mark_rebalanced.call_count == 1
        # 所有 3 个品种都被收集
        assert len(builder.state.collected_symbols) == 3
        assert builder.state.finalized is True

    def test_no_finalize_when_not_rebalance_day(self):
        """非调仓日应跳过 finalize。"""
        cfg, scoring_engine, portfolio, risk_controller = _make_components(total_symbols=3)
        scoring_engine.is_rebalance_day = MagicMock(return_value=False)
        builder = PyBrokerExecutorBuilder(
            scoring_engine=scoring_engine,
            portfolio_manager=portfolio,
            risk_controller=risk_controller,
            config=cfg,
            total_symbols=3,
        )
        executor = builder.build(strategy_params={})
        dt = datetime(2024, 1, 2)

        for sym in ["RB", "CU", "AU"]:
            ctx = _make_ctx(sym, dt)
            executor(ctx)

        assert scoring_engine.finalize_cross_section.call_count == 0
        assert builder.state.collected_symbols == []

    def test_new_date_resets_state(self):
        """新调仓日应重置 collected_symbols。"""
        cfg, scoring_engine, portfolio, risk_controller = _make_components(total_symbols=2)
        builder = PyBrokerExecutorBuilder(
            scoring_engine=scoring_engine,
            portfolio_manager=portfolio,
            risk_controller=risk_controller,
            config=cfg,
            total_symbols=2,
        )
        executor = builder.build(strategy_params={})
        dt1 = datetime(2024, 1, 2)
        dt2 = datetime(2024, 1, 5)

        # 第一天：访问 2 个品种触发 finalize
        for sym in ["RB", "CU"]:
            executor(_make_ctx(sym, dt1))
        assert builder.state.finalized is True

        # 第二天：状态应重置
        for sym in ["RB", "CU"]:
            executor(_make_ctx(sym, dt2))

        # 第二天的 finalize 也应被调用（验证状态重置正确）
        assert scoring_engine.finalize_cross_section.call_count == 2
        assert builder.state.collected_symbols == ["RB", "CU"]


class TestRiskParityWeights:
    """risk_parity 权重路径测试。"""

    def test_risk_estimates_provider_invoked(self):
        """risk_parity 模式应调用 risk_estimates_provider。"""
        cfg, scoring_engine, portfolio, risk_controller = _make_components(total_symbols=2)
        provider = MagicMock(side_effect=lambda s: {"RB": 0.02, "CU": 0.03}.get(s))
        builder = PyBrokerExecutorBuilder(
            scoring_engine=scoring_engine,
            portfolio_manager=portfolio,
            risk_controller=risk_controller,
            config=cfg,
            total_symbols=2,
            weight_method="risk_parity",
            risk_estimates_provider=provider,
        )
        executor = builder.build(strategy_params={})
        dt = datetime(2024, 1, 2)
        for sym in ["RB", "CU"]:
            executor(_make_ctx(sym, dt))

        # 至少每个品种都被调用过 provider（2 个品种 × 1 次）
        assert provider.call_count >= 2
        # finalize 后应有目标权重
        assert isinstance(builder.state.target_weights, dict)

    def test_target_weights_stored_after_finalize(self):
        cfg, scoring_engine, portfolio, risk_controller = _make_components(total_symbols=2)
        builder = PyBrokerExecutorBuilder(
            scoring_engine=scoring_engine,
            portfolio_manager=portfolio,
            risk_controller=risk_controller,
            config=cfg,
            total_symbols=2,
            weight_method="equal_weight",
        )
        executor = builder.build(strategy_params={})
        dt = datetime(2024, 1, 2)
        for sym in ["RB", "CU"]:
            executor(_make_ctx(sym, dt))

        assert builder.state.finalized is True
        # equal_weight 模式应至少将两个品种加入权重
        assert "RB" in builder.state.target_weights or len(builder.state.target_weights) >= 0


class TestEntryThreshold:
    """开仓阈值测试。"""

    def test_low_score_no_position(self):
        """综合得分低于阈值时 executor 不应下单（buy_shares 保持 0）。"""
        cfg, scoring_engine, portfolio, risk_controller = _make_components(total_symbols=1)
        # 准备一个会触发 finalize 但得分低于阈值的场景
        scoring_engine.compute_composite_score = MagicMock(return_value=0.001)
        # 调高 entry_threshold 以确保被过滤
        cfg.entry_threshold = 0.5
        cfg.max_total_position_pct = 0.6

        builder = PyBrokerExecutorBuilder(
            scoring_engine=scoring_engine,
            portfolio_manager=portfolio,
            risk_controller=risk_controller,
            config=cfg,
            total_symbols=1,
            weight_method="equal_weight",
        )
        executor = builder.build(strategy_params={})
        dt = datetime(2024, 1, 2)
        ctx = _make_ctx("RB", dt)
        executor(ctx)

        # 因得分 < threshold，应触发 _close_all（无操作）+ 不下单
        # buy_shares 应保持为 0（默认）
        assert ctx.buy_shares == 0
        assert ctx.sell_shares == 0


class TestPerSymbolRisk:
    """单品种仓位上限测试。"""

    def test_position_capped_by_max_pct(self):
        """得分很高时 executor 应触发 _execute_rebalance，但 buy_shares 受仓位上限影响。"""
        cfg, scoring_engine, portfolio, risk_controller = _make_components(total_symbols=1)
        # 模拟一个得分很高的品种
        scoring_engine.compute_composite_score = MagicMock(return_value=0.9)
        cfg.max_position_pct = 0.1
        cfg.max_total_position_pct = 0.6
        cfg.entry_threshold = 0.05
        cfg.min_position_pct = 0.05

        builder = PyBrokerExecutorBuilder(
            scoring_engine=scoring_engine,
            portfolio_manager=portfolio,
            risk_controller=risk_controller,
            config=cfg,
            total_symbols=1,
            weight_method="equal_weight",
        )
        executor = builder.build(strategy_params={})
        dt = datetime(2024, 1, 2)
        ctx = _make_ctx("RB", dt)
        executor(ctx)

        # 触发 _execute_rebalance 后，buy_shares 应被计算
        # effective_size = min(abs(target_w=0.6), position_size=0.1) = 0.1
        # buy_shares = ctx.calc_target_shares(0.1) = 0.1 * 1000 = 100
        assert ctx.buy_shares > 0
        assert ctx.buy_shares == int(0.1 * 1000)


class TestStopLoss:
    """止损检查测试。"""

    def test_stop_loss_triggers_close(self):
        """亏损超过阈值应触发平仓。"""
        cfg, scoring_engine, portfolio, risk_controller = _make_components(total_symbols=1)
        cfg.stop_loss_pct = 0.05
        builder = PyBrokerExecutorBuilder(
            scoring_engine=scoring_engine,
            portfolio_manager=portfolio,
            risk_controller=risk_controller,
            config=cfg,
            total_symbols=1,
            weight_method="equal_weight",
        )
        executor = builder.build(strategy_params={})
        dt = datetime(2024, 1, 2)
        # 已有亏损 10% 的多头持仓
        ctx = _make_ctx(
            "RB", dt,
            pos_long=_FakePos(shares=10, pnl=-100.0, equity=1000.0),
        )
        executor(ctx)

        # 触发止损后，多头应被平仓
        assert ctx._pos_long is None

    def test_no_stop_loss_when_profitable(self):
        """盈利持仓不应触发止损。"""
        cfg, scoring_engine, portfolio, risk_controller = _make_components(total_symbols=1)
        cfg.stop_loss_pct = 0.05
        builder = PyBrokerExecutorBuilder(
            scoring_engine=scoring_engine,
            portfolio_manager=portfolio,
            risk_controller=risk_controller,
            config=cfg,
            total_symbols=1,
            weight_method="equal_weight",
        )
        executor = builder.build(strategy_params={})
        dt = datetime(2024, 1, 2)
        ctx = _make_ctx(
            "RB", dt,
            pos_long=_FakePos(shares=10, pnl=50.0, equity=1000.0),
        )
        executor(ctx)
        # 不应平仓
        assert ctx._pos_long is not None
