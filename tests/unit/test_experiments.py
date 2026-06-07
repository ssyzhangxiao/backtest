"""
实验模块（E1-E11）单元测试：P2 整改后回归测试（2026-06-07）。

覆盖范围：
- _run_weighted_fusion：E2/E3 合并后行为等价性
- calculate_risk_parity_fusion：E4 风险平价融合数值正确性
- _compute_monte_carlo_stats：E9 蒙特卡洛破产概率计算
- BootstrapResult：E8 数据类字段约束
"""

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from runner.backtest.experiments.e1_e5 import _run_weighted_fusion
from runner.backtest.experiments.e6_e11 import (
    BootstrapResult,
    _compute_monte_carlo_stats,
)
from runner.common.portfolio_utils import (
    calculate_risk_parity_fusion,
    calculate_risk_parity_weights,
)


# ══════════════════════════════════════════════════════════════════════════════
# _run_weighted_fusion（E2/E3 合并）
# ══════════════════════════════════════════════════════════════════════════════


class TestRunWeightedFusion:
    """E2/E3 合并后 _run_weighted_fusion 等价性测试。"""

    def test_use_dynamic_flag_differs_call_args(self, monkeypatch, tmp_path):
        """E2 (use_dynamic=False) 与 E3 (use_dynamic=True) 调用参数不同。"""
        captured: list[dict] = []

        def _fake_safe_run_backtest(runner, start, end, name, use_execute_fusion):
            captured.append(
                {"name": name, "use_execute_fusion": use_execute_fusion}
            )
            return None

        monkeypatch.setattr(
            "runner.backtest.experiments.e1_e5.safe_run_backtest",
            _fake_safe_run_backtest,
        )
        monkeypatch.setattr(
            "runner.backtest.experiments.e1_e5.get_pybroker_runner",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "runner.backtest.experiments.e1_e5.get_strategy_names",
            lambda cfg: ["trend"],
        )
        monkeypatch.setattr(
            "runner.backtest.experiments.e1_e5.save_csv",
            lambda *a, **kw: None,
        )

        config = {
            "symbols": ["S1"],
            "backtest": {
                "full_start_date": "2020-01-01",
                "full_end_date": "2020-12-31",
            },
        }

        _run_weighted_fusion(None, config, tmp_path, use_dynamic=False)
        _run_weighted_fusion(None, config, tmp_path, use_dynamic=True)

        assert captured[0]["name"] == "E2_S1"
        assert captured[0]["use_execute_fusion"] is False
        assert captured[1]["name"] == "E3_S1"
        assert captured[1]["use_execute_fusion"] is True


# ══════════════════════════════════════════════════════════════════════════════
# calculate_risk_parity_fusion（E4 风险平价融合工具）
# ══════════════════════════════════════════════════════════════════════════════


class TestRiskParityFusion:
    """E4 风险平价融合数值正确性。"""

    def test_equal_volatility_yields_equal_weights(self):
        """两策略波动率严格相同时，权重应接近相等（各 50%）。"""
        # 构造两段完全相同的随机数据，确保波动率完全一致
        rng = np.random.default_rng(0)
        shared = pd.Series(rng.normal(0.001, 0.01, 200))
        rets = {"A": shared.copy(), "B": shared.copy()}
        weights = calculate_risk_parity_fusion(rets, window=60)
        assert abs(weights["A"] - 0.5) < 0.01
        assert abs(weights["B"] - 0.5) < 0.01

    def test_high_vol_strategy_gets_lower_weight(self):
        """高波动率策略应获得更低权重。"""
        rets = {
            "stable": pd.Series(np.random.default_rng(1).normal(0, 0.005, 200)),
            "volatile": pd.Series(np.random.default_rng(2).normal(0, 0.05, 200)),
        }
        weights = calculate_risk_parity_fusion(rets, window=60)
        assert weights["stable"] > weights["volatile"]
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_weights_dataframe_shape(self):
        """权重 DataFrame 行数 == 输入收益率长度，列数 == 策略数。"""
        rets = {
            "A": pd.Series(np.random.default_rng(1).normal(0, 0.01, 80)),
            "B": pd.Series(np.random.default_rng(2).normal(0, 0.01, 80)),
            "C": pd.Series(np.random.default_rng(3).normal(0, 0.01, 80)),
        }
        df = calculate_risk_parity_weights(rets, window=20)
        assert df.shape == (80, 3)
        # 每日权重和为 1（按列加总）
        assert df.sum(axis=1).iloc[-1] == pytest.approx(1.0, abs=1e-6)

    def test_empty_returns_returns_empty_dict(self):
        """空输入应返回空字典而非异常。"""
        assert calculate_risk_parity_fusion({}) == {}


# ══════════════════════════════════════════════════════════════════════════════
# _compute_monte_carlo_stats（E9 蒙特卡洛核心算法）
# ══════════════════════════════════════════════════════════════════════════════


class TestMonteCarloStats:
    """E9 蒙特卡洛核心算法正确性。"""

    def test_output_shapes(self):
        """final_values / max_drawdowns 应为 (n_simulations,)。"""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, 200)
        stats = _compute_monte_carlo_stats(
            returns,
            n_simulations=100,
            random_seed=42,
            bankruptcy_threshold=0.5,
        )
        assert stats["final_values"].shape == (100,)
        assert stats["max_drawdowns"].shape == (100,)
        assert 0.0 <= stats["bankruptcy_prob"] <= 1.0

    def test_bankruptcy_probability_correctness(self):
        """破产概率 = 终值 < threshold 的样本占比。"""
        # 构造一半终值必然 < 0.5 的数据：所有日收益 -0.05（净值持续衰减）
        returns = np.full(100, -0.05)
        stats = _compute_monte_carlo_stats(
            returns,
            n_simulations=200,
            random_seed=0,
            bankruptcy_threshold=0.5,
        )
        # 持续亏损必然全部破产
        assert stats["bankruptcy_prob"] == pytest.approx(1.0, abs=1e-9)

    def test_max_drawdown_negative_or_zero(self):
        """最大回撤应为非正值（亏损为负，零回撤为边界）。"""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 150)
        stats = _compute_monte_carlo_stats(
            returns, n_simulations=50, random_seed=42, bankruptcy_threshold=0.5
        )
        assert np.all(stats["max_drawdowns"] <= 0.0)

    def test_reproducibility_with_seed(self):
        """相同种子应产生相同结果。"""
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.02, 100)
        s1 = _compute_monte_carlo_stats(returns, 50, random_seed=123, bankruptcy_threshold=0.5)
        s2 = _compute_monte_carlo_stats(returns, 50, random_seed=123, bankruptcy_threshold=0.5)
        np.testing.assert_array_equal(s1["final_values"], s2["final_values"])
        assert s1["bankruptcy_prob"] == s2["bankruptcy_prob"]


# ══════════════════════════════════════════════════════════════════════════════
# BootstrapResult（E8 数据类）
# ══════════════════════════════════════════════════════════════════════════════


class TestBootstrapResult:
    """E8 BootstrapResult 数据类字段约束。"""

    def test_default_construction(self):
        """默认值应能正常构造。"""
        result = BootstrapResult([], pd.DataFrame(), 0, 42)
        assert result.sharpe_samples == []
        assert result.n_samples == 0
        assert result.random_seed == 42
        assert result.confidence_intervals.empty

    def test_field_set_preserved(self):
        """字段集合必须包含全部 4 个原字段（P2 迁移一致性约束）。"""
        field_names = {f.name for f in fields(BootstrapResult)}
        assert field_names == {
            "sharpe_samples",
            "confidence_intervals",
            "n_samples",
            "random_seed",
        }
