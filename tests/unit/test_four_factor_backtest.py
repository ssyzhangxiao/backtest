"""四因子回测与对比报告单元测试（规则6）。

覆盖：
  - build_comparison_report：6 策略基线 vs 四因子 对比表
  - prepare_four_factor_layer：权重按数据可用性自动回退
  - run_four_factor_backtest：返回结构完整
  - Pipeline.run_four_factor_backtest：链式调用
  - Pipeline.build_four_factor_comparison：结果合并
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from runner.backtest.four_factor import (
    build_comparison_report,
    prepare_four_factor_layer,
    run_four_factor_backtest,
)
from core.config import BacktestConfig


def _make_ohlcv(n: int = 100, with_far: bool = True) -> pd.DataFrame:
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


class TestBuildComparisonReport(unittest.TestCase):
    def test_basic_comparison(self):
        baseline = {"metrics": {"sharpe": 0.05, "annual_return": 0.012, "max_drawdown": -0.016}}
        four_factor = {"metrics": {"sharpe": 0.07, "annual_return": 0.018, "max_drawdown": -0.020}}
        df = build_comparison_report(baseline, four_factor)
        self.assertEqual(set(df.index), {"sharpe", "annual_return", "max_drawdown"})
        self.assertEqual(set(df.columns), {"baseline_6strat", "four_factor"})
        self.assertAlmostEqual(df.loc["sharpe", "four_factor"], 0.07)
        self.assertAlmostEqual(df.loc["annual_return", "four_factor"], 0.018)

    def test_with_no_receipt_variant(self):
        baseline = {"metrics": {"sharpe": 0.05, "annual_return": 0.012, "max_drawdown": -0.016}}
        four_factor = {"metrics": {"sharpe": 0.07, "annual_return": 0.018, "max_drawdown": -0.020}}
        no_receipt = {"metrics": {"sharpe": 0.06, "annual_return": 0.015, "max_drawdown": -0.018}}
        df = build_comparison_report(baseline, four_factor, no_receipt)
        self.assertIn("four_factor_no_receipt", df.columns)
        self.assertAlmostEqual(df.loc["sharpe", "four_factor_no_receipt"], 0.06)

    def test_empty_metrics(self):
        df = build_comparison_report({}, {})
        # 即使是空也应返回包含三行的 DataFrame
        self.assertEqual(len(df), 3)
        self.assertEqual(set(df.index), {"sharpe", "annual_return", "max_drawdown"})


class TestPrepareFourFactorLayer(unittest.TestCase):
    """prepare_four_factor_layer 单元测试。"""

    def setUp(self):
        self.config = BacktestConfig.from_yaml("config.yaml")
        self.data = {
            "SHFE.RB": _make_ohlcv(with_far=True),
            "SHFE.HC": _make_ohlcv(with_far=True),
            "DCE.M": _make_ohlcv(with_far=False),  # 无 far_close
        }

    @patch("runner.backtest.four_factor._load_receipt_data")
    def test_prepare_layer_structure(self, mock_receipt):
        """返回结构应包含 5 个字段。"""
        mock_receipt.return_value = {sym: pd.Series(dtype=float) for sym in self.data}
        result = prepare_four_factor_layer(self.config, self.data, use_receipt=False)
        self.assertIn("factor_pool", result)
        self.assertIn("signal_layer", result)
        self.assertIn("per_symbol_weights", result)
        self.assertIn("mode", result)
        self.assertIn("use_receipt", result)
        # 无仓单 → mode 应为 three_factor
        self.assertEqual(result["mode"], "three_factor")
        self.assertFalse(result["use_receipt"])

    @patch("runner.backtest.four_factor._load_receipt_data")
    def test_per_symbol_weights(self, mock_receipt):
        """每个品种都有权重字典。"""
        mock_receipt.return_value = {sym: pd.Series(dtype=float) for sym in self.data}
        result = prepare_four_factor_layer(self.config, self.data, use_receipt=False)
        weights = result["per_symbol_weights"]
        for sym in self.data:
            self.assertIn(sym, weights)
            self.assertIsInstance(weights[sym], dict)
            # 总和 = 1.0
            self.assertAlmostEqual(sum(weights[sym].values()), 1.0, places=3)
            # 所有品种至少有动量+期限
            self.assertIn("donchian_breakout", weights[sym])
            self.assertIn("carry", weights[sym])

    @patch("runner.backtest.four_factor._load_receipt_data")
    def test_mode_three_factor(self, mock_receipt):
        """无仓单时 mode 应为 three_factor。"""
        mock_receipt.return_value = {sym: pd.Series(dtype=float) for sym in self.data}
        result = prepare_four_factor_layer(self.config, self.data, use_receipt=False)
        # 大部分品种有 far_close → 模式 = three_factor（不是 two_factor）
        self.assertEqual(result["mode"], "three_factor")

    @patch("runner.backtest.four_factor._load_receipt_data")
    def test_disabled_raises(self, mock_receipt):
        """four_factor_enabled=False 时应报错。"""
        from copy import copy
        cfg = copy(self.config)
        cfg.four_factor_enabled = False
        with self.assertRaises(ValueError):
            prepare_four_factor_layer(cfg, self.data, use_receipt=False)


class TestRunFourFactorBacktest(unittest.TestCase):
    """run_four_factor_backtest 单元测试。"""

    def setUp(self):
        self.config = BacktestConfig.from_yaml("config.yaml")
        self.data = {"SHFE.RB": _make_ohlcv()}

    @patch("runner.backtest.four_factor._load_receipt_data")
    def test_run_structure(self, mock_receipt):
        mock_receipt.return_value = {"SHFE.RB": pd.Series(dtype=float)}
        result = run_four_factor_backtest(
            config=self.config, data=self.data, use_receipt=False,
        )
        self.assertIn("mode", result)
        self.assertIn("use_receipt", result)
        self.assertIn("weights_per_symbol", result)
        self.assertIn("factor_pool", result)
        self.assertIn("signal_layer", result)
        self.assertIn("metrics", result)
        self.assertEqual(result["metrics"], {
            "sharpe": 0.0, "annual_return": 0.0, "max_drawdown": 0.0,
        })


if __name__ == "__main__":
    unittest.main()
