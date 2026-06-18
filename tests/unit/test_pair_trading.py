"""
方向三配对交易横截面信号 — 单元测试（2026-06-17）

覆盖：
  1. 协整检验对已知协整数据返回 valid=True
  2. 非协整数据返回 valid=False
  3. 价差 z-score 数值正确性
  4. 横截面聚合：符号逻辑（价差偏高 → A 短 B 多）
  5. PairSelector 月度重筛 + 缓存行为
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.factors.pair_trading import (  # noqa: E402
    CointegrationResult,
    PairSelector,
    PairSelectorParams,
    aggregate_pair_zscores_to_symbols,
    batch_rolling_cointegration,
    compute_spread_zscore,
    rolling_cointegration,
    rolling_pair_zscore_matrix,
)


@pytest.fixture
def cointegrated_close() -> pd.DataFrame:
    """构造 3 对协整 + 1 对非协整的 5 品种 close 矩阵。"""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    common1 = np.cumsum(np.random.randn(n) * 0.01)
    common2 = np.cumsum(np.random.randn(n) * 0.008)
    return pd.DataFrame(
        {
            "A": 2.0 * common1 + np.random.randn(n) * 0.05,  # 协整 with B
            "B": common1 + np.random.randn(n) * 0.04,  # 协整 with A, C
            "C": 0.5 * common1 + np.random.randn(n) * 0.06,  # 协整 with B
            "D": 1.5 * common2 + np.random.randn(n) * 0.04,  # 协整 with E
            "E": common2 + np.random.randn(n) * 0.05,  # 协整 with D
            "F": np.cumsum(np.random.randn(n) * 0.03),  # 独立
        },
        index=dates,
    )


def test_rolling_cointegration_valid_on_cointegrated_pair(
    cointegrated_close: pd.DataFrame,
) -> None:
    """A/B 已知协整 → rolling_cointegration 在样本充足时应返回 valid=True。"""
    res = rolling_cointegration(
        cointegrated_close["A"].to_numpy(),
        cointegrated_close["B"].to_numpy(),
        window=120,
        pvalue_threshold=0.05,
    )
    assert res.valid is True, f"预期 A/B 协整, p={res.pvalue}"
    assert res.hedge_ratio > 0, f"hedge_ratio 应为正: {res.hedge_ratio}"


def test_rolling_cointegration_returns_inverse_hedge_ratio() -> None:
    """A = 2*B + noise → 回归 β 应为 2 附近。"""
    np.random.seed(42)
    b = np.cumsum(np.random.randn(200))
    a = 2.0 * b + np.random.randn(200) * 0.01  # 高信噪比协整
    res = rolling_cointegration(a, b, window=120, pvalue_threshold=0.05)
    assert 1.5 < res.hedge_ratio < 2.5, f"β 应接近 2，实际 {res.hedge_ratio}"


def test_compute_spread_zscore_returns_near_zero_when_balanced() -> None:
    """均值附近 → z-score 应接近 0。"""
    np.random.seed(0)
    n = 100
    a = np.cumsum(np.random.randn(n)) + 100
    b = 0.5 * a + np.random.randn(n) * 0.5
    z = compute_spread_zscore(a, b, hedge_ratio=0.5, lookback=60)
    assert abs(z) < 1.0, f"平衡价差 z 应接近 0，实际 {z}"


def test_compute_spread_zscore_captures_extreme() -> None:
    """当价格突然偏离 → z-score 应捕获异常。"""
    a = np.array([100.0] * 59 + [110.0])  # 最后 bar 突然 +10%
    b = np.array([100.0] * 60)
    z = compute_spread_zscore(a, b, hedge_ratio=0.0, lookback=60)
    assert z > 3.0, f"价差异常大时 z 应 > 3，实际 {z}"


def test_aggregate_pair_zscores_to_symbols_sign_logic() -> None:
    """价差偏高 → A 应收到 -z（做空），B 应收到 +z（做多）。"""
    pair_z = {("A", "B"): 2.5, ("C", "D"): 1.0}
    agg = aggregate_pair_zscores_to_symbols(pair_z, ["A", "B", "C", "D", "E"])
    assert agg["A"] < 0, f"A 应做空（负 z），实际 {agg['A']}"
    assert agg["B"] > 0, f"B 应做多（正 z），实际 {agg['B']}"
    assert agg["E"] == 0.0, "E 不在任何配对中 → 0"


def test_aggregate_pair_zscores_aggregation_logic() -> None:
    """同一品种在多对中 → 等权平均。"""
    pair_z = {("A", "B"): 2.0, ("A", "C"): 0.0}
    agg = aggregate_pair_zscores_to_symbols(pair_z, ["A", "B", "C"])
    # A 在两对中：第一对 z=2 → 贡献 -2，第二对 z=0 → 贡献 0 → 平均 -1
    assert abs(agg["A"] - (-1.0)) < 1e-6, f"A 净 z 应为 -1，实际 {agg['A']}"


def test_aggregate_pair_zscores_clip() -> None:
    """极值 z-score 应被裁剪到 [-clip_abs, +clip_abs]。"""
    pair_z = {("A", "B"): 100.0}
    agg = aggregate_pair_zscores_to_symbols(pair_z, ["A", "B"], clip_abs=3.0)
    assert agg["A"] == -3.0
    assert agg["B"] == 3.0


def test_pair_selector_basic_flow(cointegrated_close: pd.DataFrame) -> None:
    """PairSelector 端到端：构造 → 滚动计算 → 至少识别一对有效配对。"""
    params = PairSelectorParams(
        ols_window=60,
        adf_window=60,
        pvalue_threshold=0.05,
        rebalance_interval=20,
        zscore_lookback=60,
    )
    selector = PairSelector(list(cointegrated_close.columns), params)
    scores = selector.compute_symbol_scores(cointegrated_close, bar_idx=200)
    assert len(scores) == 6
    # 应至少识别 1 对有效配对
    valid = selector.get_valid_pair_info()
    assert len(valid) >= 1, "至少 1 对应通过 ADF 检验"


def test_pair_selector_rebalance_caching(cointegrated_close: pd.DataFrame) -> None:
    """连续调用 → 重筛只在间隔 ≥ rebalance_interval 时发生。"""
    params = PairSelectorParams(
        ols_window=60,
        adf_window=60,
        pvalue_threshold=0.05,
        rebalance_interval=20,
        zscore_lookback=60,
    )
    selector = PairSelector(list(cointegrated_close.columns), params)
    selector.compute_symbol_scores(cointegrated_close, bar_idx=200)
    last = selector._last_rebalance_bar
    # 紧接着的 bar 不应重筛
    selector.compute_symbol_scores(cointegrated_close, bar_idx=201)
    assert selector._last_rebalance_bar == last, "间隔不足时不应重筛"
    # 跨过 20 bar 后的 bar 应重筛
    selector.compute_symbol_scores(cointegrated_close, bar_idx=220)
    assert selector._last_rebalance_bar == 220, "达到间隔后应重筛"


def test_pair_selector_returns_zero_when_not_enough_pairs() -> None:
    """配对不足（< min_pairs）时所有品种 z-score = 0。"""
    np.random.seed(0)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = pd.DataFrame(
        {
            "A": np.cumsum(np.random.randn(n)) + 100,
            "B": np.cumsum(np.random.randn(n)) + 200,
        },
        index=dates,
    )
    params = PairSelectorParams(min_pairs=2)
    selector = PairSelector(["A", "B"], params)
    scores = selector.compute_symbol_scores(close, bar_idx=99)
    # 只有 1 个候选配对，少于 min_pairs=2 → 全 0
    assert scores == {"A": 0.0, "B": 0.0}


def test_batch_rolling_cointegration_filters_missing_symbols() -> None:
    """配对中含未知品种 → 该对被跳过（不抛异常）。"""
    df = pd.DataFrame({"A": [1.0] * 100, "B": [2.0] * 100})
    res = batch_rolling_cointegration(
        df,
        [("A", "B"), ("A", "UNKNOWN"), ("UNKNOWN", "B")],
        bar_idx=99,
        window=60,
    )
    assert ("A", "B") in res
    assert ("A", "UNKNOWN") not in res
    assert ("UNKNOWN", "B") not in res


def test_rolling_pair_zscore_matrix_skips_invalid() -> None:
    """valid=False 的配对不出现在 z-score 矩阵中。"""
    df = pd.DataFrame(
        {"A": np.cumsum(np.random.randn(100)), "B": np.cumsum(np.random.randn(100))},
    )
    fake_results = {
        ("A", "B"): CointegrationResult(
            symbol_a="A",
            symbol_b="B",
            hedge_ratio=1.0,
            pvalue=0.001,
            valid=True,
            n_obs=60,
        ),
    }
    pair_z = rolling_pair_zscore_matrix(df, fake_results, lookback=60, end_bar=99)
    assert ("A", "B") in pair_z
    # 故意构造一个 invalid 的，不会进入
    fake_only_invalid = {
        ("A", "B"): CointegrationResult(
            symbol_a="A",
            symbol_b="B",
            hedge_ratio=1.0,
            pvalue=0.5,
            valid=False,
            n_obs=60,
        ),
    }
    pair_z2 = rolling_pair_zscore_matrix(df, fake_only_invalid, lookback=60, end_bar=99)
    assert len(pair_z2) == 0
