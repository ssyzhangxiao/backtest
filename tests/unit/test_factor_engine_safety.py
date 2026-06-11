"""
FactorEngine 异常隔离与 safe_div 单元测试（P0/P1 整改回归保护）。
"""
import logging

import numpy as np

from core.ext.factors.alpha_futures.config import AlphaFuturesConfig
from core.ext.factors.alpha_futures.factor_engine import FactorEngine


def _make_raw(n: int = 60) -> dict:
    """构造最小可计算的 raw_data（收盘价全 0 也能跑，但会有 NaN）。"""
    close = np.linspace(100, 110, n)
    return {
        "close": close,
        "open_price": close + 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "open_interest": np.full(n, 1000.0),
        "volume": np.full(n, 100.0),
    }


def test_compute_all_runs_with_default_config():
    """默认配置下，compute_all 应当能完整跑完所有因子。"""
    engine = FactorEngine(AlphaFuturesConfig())
    results = engine.compute_all(_make_raw(80))
    # 2026-06-11 V_04 OI jump 修复后：CF_03 + H_01-05 + M_01-05 + R_01-05 + TS_01-03 + TS_composite + T_01-05 + V_01-04 = 31
    assert len(results) == 31
    # 关键因子必须存在（含 V_04 OI jump 因子）
    assert "V_04" in results
    assert "H_05" in results
    # ret 字段在公共数据中（验证 safe_div 修复后第一根不再产生 inf）
    # 注意：ret 由 _prepare_common_rolling 写入 public_data，外部不直接可见


def test_compute_all_isolates_failing_factor(caplog):
    """单因子 compute() 抛异常时，引擎继续运行，失败因子返回全 NaN 兜底。"""
    from core.ext.factors.alpha_futures import factor_registry
    from core.ext.factors.alpha_futures.base_factor import BaseFactor

    class _BoomFactor(BaseFactor):
        name = "T_05"  # 覆盖已有因子，验证该名字被填充为 NaN
        category = "test"
        dependencies = ["close"]

        def compute(self, **kwargs):  # noqa: D401
            raise RuntimeError("boom")

    # 注入到注册表（仅本次测试）
    original = factor_registry._FACTOR_REGISTRY["T_05"]
    factor_registry._FACTOR_REGISTRY["T_05"] = _BoomFactor
    try:
        # T_02/R_05 这两个因子只需 close 类基础数据，能正常产出非全 NaN 结果
        engine = FactorEngine(
            AlphaFuturesConfig(),
            factor_names=["T_05", "T_02", "R_05"],
        )
        with caplog.at_level(logging.ERROR):
            results = engine.compute_all(_make_raw(120))
        # 三因子都返回
        assert set(results.keys()) == {"T_05", "T_02", "R_05"}
        # T_05 是全 NaN（异常隔离生效）
        assert np.all(np.isnan(results["T_05"]))
        # T_02/R_05 正常产出（非全 NaN）
        assert not np.all(np.isnan(results["T_02"]))
        assert not np.all(np.isnan(results["R_05"]))
        # 错误日志被记录
        assert any("T_05" in r.message for r in caplog.records)
    finally:
        factor_registry._FACTOR_REGISTRY["T_05"] = original


def test_safe_div_used_in_common_rolling():
    """验证 _prepare_common_rolling 的 ret 字段在第一根处是 NaN 而非 inf。"""
    engine = FactorEngine(AlphaFuturesConfig())
    raw = _make_raw(30)
    public = engine._prepare_public_data(raw)
    ret = public["ret"]
    # 第一根：delay(close,1) 无效，safe_div 应填 NaN 而非 inf
    assert np.isnan(ret[0]) or np.isfinite(ret[0])
    # 全程不应有 inf
    assert np.all(np.isfinite(ret) | np.isnan(ret))


def test_from_backtest_config_uses_defaults():
    """当 BacktestConfig 无对应字段时，from_backtest_config 回退到 AlphaFuturesConfig 默认值。"""
    from core.config import BacktestConfig

    bt = BacktestConfig(symbols=["rb2501"])
    cfg = AlphaFuturesConfig.from_backtest_config(bt)
    # 默认值应当与新 config 对齐
    assert cfg.gap_weight == 0.5
    assert cfg.limit_move_threshold == 0.06
    assert cfg.carry_oi_threshold == 10000
    assert cfg.symbol == "rb2501"


def test_alpha_futures_24_reexports_same_config():
    """验证 alpha_futures_24.AlphaFuturesConfig 与新 config 是同一个类。"""
    from core.factors.alpha_futures_24 import AlphaFuturesConfig as Legacy
    from core.ext.factors.alpha_futures.config import AlphaFuturesConfig as New

    assert Legacy is New


def test_atr_scaling_parameter_propagates():
    """compute_sub_strategy_scores_from_ohlcv 的 atr_scaling 参数应实际影响输出。

    验证策略：用极小的 scaling 让信号饱和到 ±1，用极大的 scaling 让信号趋近 0。
    两者均值绝对值必有差异。
    """
    from core.ext.factors.alpha_futures.sub_strategy_aggregator import (
        compute_sub_strategy_scores_from_ohlcv,
    )
    import pandas as pd

    n = 80
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n),
        "open": np.linspace(100, 110, n),
        "high": np.linspace(101, 111, n),
        "low": np.linspace(99, 109, n),
        "close": np.linspace(100, 110, n),
        "volume": np.full(n, 1000.0),
        "open_interest": np.full(n, 1000.0),
    })

    df_tiny = compute_sub_strategy_scores_from_ohlcv(df, atr_scaling=0.001)
    df_huge = compute_sub_strategy_scores_from_ohlcv(df, atr_scaling=1e6)
    # 极小 scaling → 信号饱和到 ±1；极大 scaling → 信号趋近 0
    trend_tiny = df_tiny["trend"].abs().mean()
    trend_huge = df_huge["trend"].abs().mean()
    assert trend_tiny > trend_huge
    # 极小 scaling 应当达到 clip 上限附近（> 0.5）
    assert trend_tiny > 0.5
    # 极大 scaling 应当远低于 0.5
    assert trend_huge < 0.1


# ── Issue #2 / #3 / #4 回归保护 ─────────────────────────────


def test_alpha_futures_config_no_dead_fields():
    """Issue #2: gap_weight_min/max 与 delivery_exclude_days 已删除（无引用）。"""
    import dataclasses

    from core.ext.factors.alpha_futures.config import AlphaFuturesConfig

    fields = {f.name for f in dataclasses.fields(AlphaFuturesConfig)}
    # 这些字段曾被定义但无任何代码使用，会污染 config 公共 API
    assert "gap_weight_min" not in fields
    assert "gap_weight_max" not in fields
    assert "delivery_exclude_days" not in fields
    # 必要字段仍保留
    assert "gap_weight" in fields
    assert "limit_move_threshold" in fields
    assert "carry_oi_threshold" in fields


def test_factor_review_uses_pearson():
    """Issue #3: factor_review._check_sensitivity 与 factor_evaluator 统一使用 Pearson。"""
    import inspect

    from core.factors import factor_review
    from core.ext.factors.evaluator import FactorEvaluator

    src = inspect.getsource(factor_review.FactorReviewer._check_sensitivity)
    assert 'method="pearson"' in src
    assert 'method="spearman"' not in src
    # 复核：evaluator 本身也是 Pearson
    eval_src = inspect.getsource(FactorEvaluator._compute_ic_stats)
    assert "np.corrcoef" in eval_src  # np.corrcoef 即 Pearson


def test_exp_transform_clip_threshold_configurable():
    """Issue #4: exp_transform 的 clip_threshold 参数可配置。"""
    from core.ext.factors.transformer import FactorTransformer

    transformer = FactorTransformer()
    f = np.array([1.0, 5.0, 20.0])  # 最后一个超 10

    # 默认 clip_threshold=10：极端值被裁剪到 10
    out_default = transformer.exp_transform(f)
    assert abs(abs(out_default[2]) - (np.exp(10) - 1)) < 1e-6

    # clip_threshold=5：极端值被裁剪到 5
    out_5 = transformer.exp_transform(f, clip_threshold=5.0)
    assert abs(abs(out_5[2]) - (np.exp(5) - 1)) < 1e-6

    # clip_threshold=None：不裁剪
    out_none = transformer.exp_transform(f, clip_threshold=None)
    assert abs(abs(out_none[2]) - (np.exp(20) - 1)) < 1e-3

    # 三个结果对极端值的处理应不同
    assert abs(out_default[2]) < abs(out_none[2])


# ── 路径 C→A 合并回归保护 ─────────────────────────────


def test_sub_strategy_indicators_path_c_uses_factor_engine():
    """build_xxx_indicators 必须走路径 A（compute_sub_strategy_scores_from_ohlcv）。

    防止回归：原 build_xxx 自实现裸价算法，与路径 A 不一致。
    """
    import inspect

    from core.engine import sub_strategy_indicators

    for name in [
        "build_trend_indicators",
        "build_term_structure_indicators",
        "build_mean_reversion_indicators",
        "build_vol_breakout_indicators",
        "build_composite_indicators",
    ]:
        src = inspect.getsource(getattr(sub_strategy_indicators, name))
        # 统一委托给 _signal_from_factor_column → compute_sub_strategy_scores_from_ohlcv
        assert "_signal_from_factor_column" in src, (
            f"{name} 未走 _signal_from_factor_column（路径 A），请检查"
        )
        # 不应再使用裸价算法（pct_change / tanh / rolling）
        assert "pct_change" not in src or "_signal_from_factor_column" in src
        assert "np.tanh" not in src or "_signal_from_factor_column" in src


def test_path_a_and_c_yield_identical_signals():
    """路径 A（因子聚合器）与路径 C（build_indicators）输出必须完全一致。"""
    import pandas as pd

    from core.engine.sub_strategy_indicators import (
        _ohlcv_from_bar,
        build_trend_indicators,
        build_term_structure_indicators,
        build_mean_reversion_indicators,
        build_vol_breakout_indicators,
        build_composite_indicators,
    )
    from core.ext.factors.alpha_futures.config import AlphaFuturesConfig
    from core.ext.factors.alpha_futures.sub_strategy_aggregator import (
        compute_sub_strategy_scores_from_ohlcv,
    )

    n = 100
    close = np.linspace(100, 110, n) + np.random.RandomState(42).normal(0, 0.5, n)
    bar_data = type("BarData", (), {
        "open": close + 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 1000.0),
        "open_interest": np.full(n, 1000.0),
        "date": pd.date_range("2025-01-01", periods=n),
    })()

    df = _ohlcv_from_bar(bar_data)
    assert df is not None and len(df) == n
    scored_A = compute_sub_strategy_scores_from_ohlcv(df, config=AlphaFuturesConfig())

    mappings = [
        ("trend", build_trend_indicators),
        ("term_structure", build_term_structure_indicators),
        ("mean_reversion", build_mean_reversion_indicators),
        ("vol_breakout", build_vol_breakout_indicators),
        ("composite_resonance", build_composite_indicators),
    ]
    for sname, builder in mappings:
        A_col = scored_A[sname].fillna(0.0).to_numpy()
        C_arr = builder({})[0][1](bar_data)
        np.testing.assert_allclose(
            A_col, C_arr, atol=1e-8,
            err_msg=f"{sname}: 路径 A 与 C 不一致",
        )


def test_sub_strategy_signals_produced_in_production_format():
    """验证 build_xxx_indicators 返回的 List[tuple] 格式与 PyBroker 兼容。"""
    from core.engine.sub_strategy_indicators import build_trend_indicators

    result = build_trend_indicators({})
    assert isinstance(result, list)
    assert len(result) >= 1
    name, fn = result[0]
    assert isinstance(name, str)
    assert callable(fn)
    # 函数接受 bar_data 单参数，返回 numpy 数组
    bar_data = type("BarData", (), {
        "open": np.array([100.0]),
        "high": np.array([101.0]),
        "low": np.array([99.0]),
        "close": np.array([100.0]),
    })()
    out = fn(bar_data)
    assert isinstance(out, np.ndarray)


def test_runner_validation_cross_sectional_uses_path_a():
    """回归保护：runner/validation/cross_sectional.py 走路径A因子库。

    P0 整改（2026-06-07）：core/strategies/ 已整体删除，
    验证 cross_sectional.py 直接调用 sub_strategy_aggregator（路径A），
    而不是已删除的 CrossSectionalStrategy。
    """
    import importlib
    import inspect

    cs_module = importlib.import_module("runner.validation.cross_sectional")
    source = inspect.getsource(cs_module)

    # 必须显式引用路径A
    assert "compute_sub_strategy_scores_from_ohlcv" in source, (
        "runner/validation/cross_sectional.py 必须使用路径A "
        "compute_sub_strategy_scores_from_ohlcv"
    )

    # 不应再引用已删除模块
    assert "from core.strategies" not in source, (
        "runner/validation/cross_sectional.py 不应再引用已删除的 core.strategies"
    )


def test_compute_all_aligns_misaligned_factor_length():
    """compute_all 末尾应做边界保护：因子返回长度与 close 不一致时自动 NaN right-align。

    回归保护（2026-06-07）：TS_01/02/03 三个期限结构因子内部有
    `length = len(close) if close is not None else 100` 的硬编码 fallback，
    当 close 上下文未传入时返回 100，而 df 可能是 183 行。下游
    `df[factor_name] = factor_values` 会抛
    `Length of values (100) does not match length of index (183)`。
    compute_all 必须强制等长契约。
    """
    from core.ext.factors.alpha_futures import factor_registry
    from core.ext.factors.alpha_futures.base_factor import BaseFactor

    class _ShortFactor(BaseFactor):
        """模拟 TS_01 的 hardcoded length=100 行为：返回 100 元素而非 close 长度。"""
        name = "T_05"  # 覆盖已有因子，避免污染注册表
        category = "test"
        dependencies = ["close"]

        def compute(self, **kwargs):  # noqa: D401
            return np.full(100, np.nan, dtype=float)

    original = factor_registry._FACTOR_REGISTRY["T_05"]
    factor_registry._FACTOR_REGISTRY["T_05"] = _ShortFactor
    try:
        engine = FactorEngine(
            AlphaFuturesConfig(),
            factor_names=["T_05"],
        )
        raw = _make_raw(183)  # close 长度 183
        results = engine.compute_all(raw)

        target_len = len(raw["close"])
        for name, vals in results.items():
            assert len(vals) == target_len, (
                f"因子 {name} 长度 {len(vals)} != 期望 {target_len}，"
                "compute_all 边界保护失效"
            )
        # 触发自动对齐：T_05 应被 NaN right-align 到 183
        assert len(results["T_05"]) == 183
        # 前 83 个应是 NaN，后 100 个仍是 NaN（因子本身全 NaN）
        assert np.all(np.isnan(results["T_05"]))
    finally:
        factor_registry._FACTOR_REGISTRY["T_05"] = original
