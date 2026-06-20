"""四因子 CTA 融合信号单元测试（规则6）。

覆盖：
  - get_four_factor_signal：四因子加权融合
  - get_four_factor_signal_dynamic：方向二仓位缩放
  - 缺失数据回退：basis_momentum / receipt_change 自动权重重分配
  - 权重裁剪：信号输出恒定在 [-1, 1]
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from core.execution.signal_abstraction import SignalAbstractionLayer


def _make_mock_pool(signals: dict) -> MagicMock:
    """构造一个 mock factor_pool，根据 (symbol, bar_idx) 返回 signals。"""
    pool = MagicMock()
    pool.compute_signals_for_bar = MagicMock(return_value=signals)
    return pool


def _make_ohlcv(n: int = 100, with_far: bool = True) -> pd.DataFrame:
    """构造测试用 OHLCV DataFrame。"""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "open": np.linspace(100, 110, n),
        "high": np.linspace(101, 111, n),
        "low": np.linspace(99, 109, n),
        "close": np.linspace(100, 110, n),
        "volume": np.full(n, 1000.0),
    })
    if with_far:
        df["far_close"] = np.linspace(105, 115, n)
    return df


class TestFourFactorSignal(unittest.TestCase):
    """get_four_factor_signal 测试。"""

    def setUp(self):
        self.pool = _make_mock_pool({})
        self.layer = SignalAbstractionLayer(
            factor_pool=self.pool,
            xs_position_base=0.25,
            xs_opposite_penalty=0.4,
        )

    def test_basic_weighted_sum(self):
        """四因子加权求和应符合预期。"""
        signals = {
            "donchian_breakout": 1.0,
            "carry": 0.5,
            "basis_momentum": -0.5,
            "receipt_change": 1.0,
        }
        self.pool.compute_signals_for_bar = MagicMock(return_value=signals)
        result = self.layer.get_four_factor_signal(
            "SHFE.RB", _make_ohlcv(), bar_idx=50,
        )
        # 权重 0.30*1.0 + 0.25*0.5 + 0.25*(-0.5) + 0.20*1.0
        # = 0.30 + 0.125 - 0.125 + 0.20 = 0.50
        expected = 0.30 * 1.0 + 0.25 * 0.5 + 0.25 * (-0.5) + 0.20 * 1.0
        self.assertAlmostEqual(result, expected, places=4)

    def test_all_zero_signals(self):
        """全 0 信号 → 输出 0。"""
        signals = {
            "donchian_breakout": 0.0,
            "carry": 0.0,
            "basis_momentum": 0.0,
            "receipt_change": 0.0,
        }
        self.pool.compute_signals_for_bar = MagicMock(return_value=signals)
        result = self.layer.get_four_factor_signal(
            "SHFE.RB", _make_ohlcv(), bar_idx=50,
        )
        self.assertEqual(result, 0.0)

    def test_missing_basis_momentum_fallback(self):
        """无 far_close 时，basis_momentum 权重重新分配。"""
        # 通过 compute_four_factor_weights 显式调整
        weights = SignalAbstractionLayer.compute_four_factor_weights(
            has_basis=False, has_receipt=True,
        )
        # 总和应为 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
        # basis_momentum 不在权重中
        self.assertNotIn("basis_momentum", weights)
        # 应有动量、期限、仓单
        self.assertIn("donchian_breakout", weights)
        self.assertIn("carry", weights)
        self.assertIn("receipt_change", weights)

    def test_missing_receipt_fallback(self):
        """无仓单时，receipt_change 权重重新分配。"""
        weights = SignalAbstractionLayer.compute_four_factor_weights(
            has_basis=True, has_receipt=False,
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
        self.assertNotIn("receipt_change", weights)
        self.assertIn("donchian_breakout", weights)
        self.assertIn("carry", weights)
        self.assertIn("basis_momentum", weights)

    def test_all_available_default_weights(self):
        """全可用时使用默认权重。"""
        weights = SignalAbstractionLayer.compute_four_factor_weights(
            has_basis=True, has_receipt=True,
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
        # 默认权重
        self.assertAlmostEqual(weights["donchian_breakout"], 0.30, places=4)
        self.assertAlmostEqual(weights["carry"], 0.25, places=4)
        self.assertAlmostEqual(weights["basis_momentum"], 0.25, places=4)
        self.assertAlmostEqual(weights["receipt_change"], 0.20, places=4)

    def test_only_donchian_and_carry(self):
        """仅动量+期限可用时，权重按原比例放大。"""
        weights = SignalAbstractionLayer.compute_four_factor_weights(
            has_basis=False, has_receipt=False,
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
        self.assertIn("donchian_breakout", weights)
        self.assertIn("carry", weights)
        # 比例 = 0.30 : 0.25 = 6:5
        self.assertAlmostEqual(weights["donchian_breakout"], 0.30 / 0.55, places=4)
        self.assertAlmostEqual(weights["carry"], 0.25 / 0.55, places=4)

    def test_clipping_to_range(self):
        """输出应在 [-1, 1]。"""
        signals = {
            "donchian_breakout": 1.0,
            "carry": 1.0,
            "basis_momentum": 1.0,
            "receipt_change": 1.0,
        }
        self.pool.compute_signals_for_bar = MagicMock(return_value=signals)
        result = self.layer.get_four_factor_signal(
            "SHFE.RB", _make_ohlcv(), bar_idx=50,
        )
        # 总和 = 0.30 + 0.25 + 0.25 + 0.20 = 1.0
        self.assertAlmostEqual(result, 1.0, places=4)

    def test_custom_weights(self):
        """自定义权重应被使用。"""
        signals = {
            "donchian_breakout": 1.0,
            "carry": 0.0,
            "basis_momentum": 0.0,
            "receipt_change": 0.0,
        }
        self.pool.compute_signals_for_bar = MagicMock(return_value=signals)
        result = self.layer.get_four_factor_signal(
            "SHFE.RB", _make_ohlcv(), bar_idx=50,
            weights={"donchian_breakout": 0.5, "carry": 0.2, "basis_momentum": 0.2, "receipt_change": 0.1},
        )
        # 只动量有信号 → 0.5
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_set_weights_runtime(self):
        """运行时 set_four_factor_weights 应覆盖默认。"""
        self.layer.set_four_factor_weights({
            "donchian_breakout": 0.5,
            "carry": 0.2,
            "basis_momentum": 0.2,
            "receipt_change": 0.1,
        })
        signals = {
            "donchian_breakout": 1.0,
            "carry": 0.0,
            "basis_momentum": 0.0,
            "receipt_change": 0.0,
        }
        self.pool.compute_signals_for_bar = MagicMock(return_value=signals)
        result = self.layer.get_four_factor_signal(
            "SHFE.RB", _make_ohlcv(), bar_idx=50,
        )
        self.assertAlmostEqual(result, 0.5, places=4)


class TestFourFactorSignalDynamic(unittest.TestCase):
    """get_four_factor_signal_dynamic 测试（叠加方向二）。"""

    def setUp(self):
        self.pool = _make_mock_pool({})
        # 方向二参数 b=0.25, p=0.4
        self.layer = SignalAbstractionLayer(
            factor_pool=self.pool,
            xs_position_base=0.25,
            xs_position_ceiling=1.0,
            xs_opposite_penalty=0.4,
        )

    def test_strong_xs_same_direction_full_position(self):
        """横截面强 + 与四因子同向 → 满仓。"""
        signals = {
            "donchian_breakout": 0.5,
            "carry": 0.5,
            "basis_momentum": 0.5,
            "receipt_change": 0.5,
        }
        self.pool.compute_signals_for_bar = MagicMock(return_value=signals)
        # 横截面 z=1.0（强）
        result = self.layer.get_four_factor_signal_dynamic(
            "SHFE.RB", _make_ohlcv(), bar_idx=50, cross_section_z=1.0,
        )
        # 满仓 position_scale = 0.25 + (1.0-0.25)*1.0 = 1.0
        # four_factor = 0.5（总权重 1.0）
        # final = 0.5 * 1.0 = 0.5
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_weak_xs_half_position(self):
        """横截面无信息 → 半仓。"""
        signals = {
            "donchian_breakout": 1.0,
            "carry": 0.0,
            "basis_momentum": 0.0,
            "receipt_change": 0.0,
        }
        self.pool.compute_signals_for_bar = MagicMock(return_value=signals)
        # 横截面 z=0 → 弱信号
        result = self.layer.get_four_factor_signal_dynamic(
            "SHFE.RB", _make_ohlcv(), bar_idx=50, cross_section_z=0.0,
        )
        # position_scale = 0.25 + 0 = 0.25
        # four_factor = 0.30（仅动量有信号，权重 0.30）
        # final = 0.30 * 0.25 = 0.075
        self.assertAlmostEqual(result, 0.30 * 0.25, places=4)

    def test_opposite_direction_penalty(self):
        """异号时 → 额外减仓。"""
        signals = {
            "donchian_breakout": 1.0,
            "carry": 0.0,
            "basis_momentum": 0.0,
            "receipt_change": 0.0,
        }
        self.pool.compute_signals_for_bar = MagicMock(return_value=signals)
        # 四因子为正，横截面为负（异号）
        result = self.layer.get_four_factor_signal_dynamic(
            "SHFE.RB", _make_ohlcv(), bar_idx=50, cross_section_z=-1.0,
        )
        # position_scale = 0.25 + (1.0-0.25)*1.0 = 1.0
        # 异号 → 1.0 * 0.4 = 0.4
        # four_factor = 0.30
        # final = 0.30 * 0.4 = 0.12
        self.assertAlmostEqual(result, 0.30 * 0.4, places=4)

    def test_clipping(self):
        """输出应在 [-1, 1]。"""
        signals = {
            "donchian_breakout": 1.0,
            "carry": 1.0,
            "basis_momentum": 1.0,
            "receipt_change": 1.0,
        }
        self.pool.compute_signals_for_bar = MagicMock(return_value=signals)
        result = self.layer.get_four_factor_signal_dynamic(
            "SHFE.RB", _make_ohlcv(), bar_idx=50, cross_section_z=2.0,
        )
        self.assertGreaterEqual(result, -1.0)
        self.assertLessEqual(result, 1.0)


if __name__ == "__main__":
    unittest.main()
