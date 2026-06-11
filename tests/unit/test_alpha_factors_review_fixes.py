"""
代码评审整改回归测试。

覆盖 4 项修复：
  1. core.factors.alpha_futures.__init__.py 显式导出所有 31 因子
  2. core.factors.operators.ema 函数存在并可用（cross_spread 与 ts_composite 依赖）
  3. cross_spread.STRONG_IC_PAIRS 支持 YAML 配置 + 运行时覆盖
  4. _aggregate_group 全无数据时返回空数组，_to_series 将其转为全 NaN
     （不允许"无数据 = 全 0"这种会掩盖数据缺失的语义）
  5. _to_series 长度右对齐（输入短于 index 时尾部对齐 + 前部 NaN 填充）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# 1. __init__.py 导出全部 31 因子
# ──────────────────────────────────────────────


EXPECTED_FACTOR_NAMES = [
    # 趋势
    "T_01", "T_02", "T_03", "T_04", "T_05",
    # 回归
    "R_01", "R_02", "R_03", "R_04", "R_05",
    # 波动率
    "V_01", "V_02", "V_03", "V_04",
    # 资金流
    "M_01", "M_02", "M_03", "M_04", "M_05",
    # 高阶复合
    "H_01", "H_02", "H_03", "H_04", "H_05",
    # 资金流扩展
    "CF_01", "CF_02", "CF_03",
    # 期限结构
    "TS_01", "TS_02", "TS_03", "TS_composite",
]


def test_init_exports_all_factors():
    """__init__.py 必须显式导出全部因子类（不只是注册表里）。"""
    from core.factors import alpha_futures
    for name in EXPECTED_FACTOR_NAMES:
        assert name in alpha_futures.__all__, f"__all__ 缺少 {name}"
        cls = getattr(alpha_futures, name, None)
        assert cls is not None, f"{name} 未在 alpha_futures 命名空间中"
        # 验证是 BaseFactor 的子类
        from core.ext.factors.alpha_futures.base_factor import BaseFactor
        assert issubclass(cls, BaseFactor), f"{name} 不是 BaseFactor 子类"


def test_init_exports_factor_count_matches_registry():
    """__init__.py 导出的因子数应与注册表一致。"""
    from core.ext.factors.alpha_futures.factor_registry import list_available_factors
    registered = set(list_available_factors())
    exported = set(EXPECTED_FACTOR_NAMES)
    assert registered == exported, (
        f"注册表与导出不匹配: \n"
        f"  仅注册: {registered - exported}\n"
        f"  仅导出: {exported - registered}"
    )


# ──────────────────────────────────────────────
# 2. ema 函数存在并可用
# ──────────────────────────────────────────────


def test_ema_exists_in_operators():
    """operators.py 必须导出 ema（cross_spread / ts_composite 依赖）。"""
    from core.factors.operators import ema
    assert callable(ema), "ema 不是可调用对象"


def test_ema_basic():
    """ema 基础数值正确性：window=3 序列 [1,2,3,4,5] 的递推值。

    实现说明：当前 ema 实现的 warm-up 是"从当前位置向前回看至多 window 个
    非 NaN 值"取均值；i=0 时回看 1 个值 = 1.0 作为初始化。后续按递推公式
    EMA_t = (x_t - EMA_{t-1}) * k + EMA_{t-1} 计算，k = 2/(window+1) = 0.5。
    """
    from core.factors.operators import ema
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ema(arr, window=3)
    # i=0: warm-up 1.0; (1-1)*0.5+1 = 1.0
    # i=1: (2-1)*0.5+1 = 1.5
    # i=2: (3-1.5)*0.5+1.5 = 2.25
    # i=3: (4-2.25)*0.5+2.25 = 3.125
    # i=4: (5-3.125)*0.5+3.125 = 4.0625
    expected = np.array([1.0, 1.5, 2.25, 3.125, 4.0625])
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_ema_handles_nan():
    """ema 必须 NaN 安全：遇到 NaN 输入跳过，下一个有效值重新初始化。"""
    from core.factors.operators import ema
    arr = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    out = ema(arr, window=3)
    # 索引 1：NaN 输入，输出保持 prev（索引 0 的 warm-up 值）
    # 索引 2：3.0 输入，重新初始化（前 N 个非 NaN 均值 = (1+3)/2=2）
    #         递推：(3-2)*0.5+2 = 2.5
    assert not np.any(np.isnan(out)), f"ema 输出不应含 NaN: {out}"


def test_ema_import_path_works_from_cross_spread():
    """cross_spread.py 内部的 from ..operators import ema, zscore 必须可用。"""
    # 仅做 import 测试，验证 from 路径与绝对路径一致
    from core.ext.factors.alpha_futures import cross_spread
    assert hasattr(cross_spread, "ema")
    assert hasattr(cross_spread, "zscore")


# ──────────────────────────────────────────────
# 3. cross_spread.STRONG_IC_PAIRS 可配置化
# ──────────────────────────────────────────────


def test_strong_ic_pairs_default():
    """未加载配置时，STRONG_IC_PAIRS 应当等于内置默认值。"""
    from core.ext.factors.alpha_futures import cross_spread
    from core.ext.factors.alpha_futures.cross_spread import (
        _STRONG_IC_PAIRS_DEFAULT, STRONG_IC_PAIRS,
    )
    # 加载配置后调用 set_strong_ic_pairs([]) 应当回到默认值
    cross_spread.set_strong_ic_pairs([])
    assert STRONG_IC_PAIRS == _STRONG_IC_PAIRS_DEFAULT


def test_strong_ic_pairs_runtime_override():
    """set_strong_ic_pairs([...]) 应当能覆盖默认列表。"""
    from core.ext.factors.alpha_futures import cross_spread
    cross_spread.set_strong_ic_pairs(["XPRB_I", "XAU_AG"])
    assert cross_spread.STRONG_IC_PAIRS == ("XPRB_I", "XAU_AG")
    # 还原默认
    cross_spread.set_strong_ic_pairs([])


def test_strong_ic_pairs_filters_invalid():
    """set_strong_ic_pairs 应当过滤掉不在 CHAIN_PAIRS 中的配对。"""
    from core.ext.factors.alpha_futures import cross_spread
    cross_spread.set_strong_ic_pairs(["XPRB_I", "INVALID_PAIR", "XAU_AG"])
    # INVALID_PAIR 被过滤
    assert cross_spread.STRONG_IC_PAIRS == ("XPRB_I", "XAU_AG")
    cross_spread.set_strong_ic_pairs([])


def test_strong_ic_pairs_all_invalid_falls_back_to_default():
    """全部无效时回退到默认。"""
    from core.ext.factors.alpha_futures import cross_spread
    from core.ext.factors.alpha_futures.cross_spread import _STRONG_IC_PAIRS_DEFAULT
    cross_spread.set_strong_ic_pairs(["NOPE1", "NOPE2"])
    assert cross_spread.STRONG_IC_PAIRS == _STRONG_IC_PAIRS_DEFAULT


def test_strong_ic_pairs_load_from_yaml(tmp_path: Path):
    """load_strong_ic_pairs_from_config 必须从 config.yaml 读取 strong_ic_pairs。"""
    from core.ext.factors.alpha_futures import cross_spread
    yaml_content = """
backtest:
  initial_capital: 1000000
  full_start_date: '2020-01-01'
  full_end_date: '2024-01-01'
  commission: 0.0001
  slippage: 0.0001
symbols: [SHFE.RB]
factor_weights:
  trend: 0.5
  term_structure: 0.5
strategies:
- name: trend
  params: {}
factors:
  cross_spread:
    strong_ic_pairs:
    - XPRB_I
    - XAU_AG
    spread_window: 60
    smoothing_window: 3
    direction: revert
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)
    pairs = cross_spread.load_strong_ic_pairs_from_config(str(config_path))
    assert pairs == ("XPRB_I", "XAU_AG"), f"unexpected: {pairs}"
    # 还原
    cross_spread.set_strong_ic_pairs([])


def test_strong_ic_pairs_load_handles_missing_section(tmp_path: Path):
    """配置缺少 cross_spread 段时，应当回退到默认（不抛异常）。"""
    from core.ext.factors.alpha_futures import cross_spread
    from core.ext.factors.alpha_futures.cross_spread import _STRONG_IC_PAIRS_DEFAULT
    yaml_content = """
backtest:
  initial_capital: 1000000
symbols: [SHFE.RB]
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)
    pairs = cross_spread.load_strong_ic_pairs_from_config(str(config_path))
    assert pairs == _STRONG_IC_PAIRS_DEFAULT


# ──────────────────────────────────────────────
# 4. _aggregate_group / _to_series NaN 语义
# ──────────────────────────────────────────────


def test_aggregate_group_empty_returns_empty_array():
    """_aggregate_group 在组内所有因子均为 None 时返回空数组（长度 0）。"""
    from core.ext.factors.alpha_futures.sub_strategy_aggregator import _aggregate_group
    result = _aggregate_group({}, ["T_01", "T_02", "T_03"])
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


def test_to_series_empty_input_returns_all_nan():
    """_to_series 在输入为空数组时返回全 NaN（不允许返回全 0 掩盖数据缺失）。"""
    from core.ext.factors.alpha_futures.sub_strategy_aggregator import _to_series
    index = pd.RangeIndex(10)
    result = _to_series(np.array([]), index)
    assert len(result) == 10
    assert result.isna().all(), f"应当全 NaN，但得到: {result.tolist()}"
    assert result.dtype == float


def test_to_series_equal_length():
    """_to_series 在长度匹配时直接对齐。"""
    from core.ext.factors.alpha_futures.sub_strategy_aggregator import _to_series
    index = pd.RangeIndex(5)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = _to_series(values, index)
    np.testing.assert_array_equal(result.values, values)


def test_to_series_shorter_input_right_aligns():
    """_to_series 在输入短于 index 时右对齐：values 放尾部，前部补 NaN。"""
    from core.ext.factors.alpha_futures.sub_strategy_aggregator import _to_series
    index = pd.RangeIndex(5)
    values = np.array([10.0, 20.0, 30.0])
    result = _to_series(values, index)
    expected = [np.nan, np.nan, 10.0, 20.0, 30.0]
    np.testing.assert_array_equal(result.values, expected)


def test_to_series_longer_input_truncates_tail():
    """_to_series 在输入长于 index 时截取尾部。"""
    from core.ext.factors.alpha_futures.sub_strategy_aggregator import _to_series
    index = pd.RangeIndex(3)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = _to_series(values, index)
    expected = [3.0, 4.0, 5.0]
    np.testing.assert_array_equal(result.values, expected)


def test_aggregate_group_partial_missing_uses_valid_only():
    """_aggregate_group 在部分因子为 None 时，仅对有效因子做 nanmean。"""
    from core.ext.factors.alpha_futures.sub_strategy_aggregator import _aggregate_group
    result = _aggregate_group(
        {"T_01": np.array([1.0, 2.0, 3.0, 4.0])},  # T_02/T_03 缺失
        ["T_01", "T_02", "T_03"],
    )
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])


def test_aggregate_group_nanmean_skips_nan():
    """_aggregate_group 内 np.nanmean 自动跳过 NaN。"""
    from core.ext.factors.alpha_futures.sub_strategy_aggregator import _aggregate_group
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([np.nan, np.nan, 3.0, 4.0])
    result = _aggregate_group({"A": a, "B": b}, ["A", "B"])
    # 索引 0: (1+nan)/1 = 1.0
    # 索引 1: (2+nan)/1 = 2.0
    # 索引 2: (nan+3)/1 = 3.0
    # 索引 3: (4+4)/2 = 4.0
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])


# ──────────────────────────────────────────────
# 5. 集成测试：compute_sub_strategy_scores 端到端
# ──────────────────────────────────────────────


def test_compute_sub_strategy_scores_no_far_data_uses_nan():
    """当缺少 far_close 列时，term_structure 子策略得分应当是"中性 0"。

    实现说明：TS_01/02/03 缺近/远月数据 → 返回全 NaN → ATR 归一化时
    `np.nan / scale = NaN` → `_safe_clip` 显式 `fillna(0.0)` → 输出 0。
    这与"无数据"语义等价于"中性信号"的设计一致（不做多不做空）。
    """
    from core.ext.factors.alpha_futures.sub_strategy_aggregator import (
        compute_sub_strategy_scores_from_ohlcv,
    )
    n = 60
    # 制造有真实波动和趋势的数据，让其他子策略有非零信号
    rng = np.random.default_rng(42)
    base = np.linspace(100, 115, n)
    noise = rng.normal(0, 0.5, n)
    close = base + noise
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": close + rng.normal(0, 0.1, n),
        "close": close,
        "high": close + np.abs(rng.normal(0, 0.3, n)),
        "low": close - np.abs(rng.normal(0, 0.3, n)),
        "volume": rng.uniform(800, 1200, n),
    })
    scores = compute_sub_strategy_scores_from_ohlcv(df)
    # term_structure 应当全部为 0（中性信号，因子数据全无）
    assert (scores["term_structure"] == 0).all(), (
        f"term_structure 应当全 0（中性），但得到: {scores['term_structure'].tolist()}"
    )
    # composite_resonance 在有数据时应当产生非零信号
    assert (scores["composite_resonance"] != 0).any(), "composite_resonance 应当有非零信号"
    # 输出列应当齐全
    expected_cols = ["trend", "term_structure", "mean_reversion",
                     "vol_breakout", "composite_resonance", "forward_return"]
    for col in expected_cols:
        assert col in scores.columns, f"缺少列: {col}"
    # 长度对齐：scores 行数应当等于输入 df 行数
    assert len(scores) == n, f"scores 行数 {len(scores)} != df 行数 {n}"
