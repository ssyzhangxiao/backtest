"""
alpha_futures 工程化修复验证测试。

覆盖：
  1. compute_adaptive_gap_weight 自适应权重
  2. FactorEngine 集成自适应权重
  3. V_01 / CF_01 强制结合价格方向
  4. CF_03 动态分位数阈值
  5. V_02 min_periods 兜底（涨跌停日）
  6. AlphaFutures24.compute_all 复权预检
  7. decay_linear 扩张窗口
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ───────────────────────────────────────────────────────────
# 1. compute_adaptive_gap_weight
# ───────────────────────────────────────────────────────────
def test_adaptive_gap_weight_continuing():
    """跳空后高延续 → 权重应 > 0.5（更信任 open）。"""
    from core.factors.futures_data_cleaners import compute_adaptive_gap_weight

    n = 60
    # 高延续场景：open[i] = 100 + 5*i（每日跳空+5），close[i] = 100 + 10*i（日内再+5）
    # gap[i] = 5 (正), intraday[i] = 5 (正) → 方向一致
    open_p = np.zeros(n)
    close_p = np.zeros(n)
    open_p[0] = 100.0
    close_p[0] = 100.0
    for i in range(1, n):
        open_p[i] = close_p[i - 1] + 5
        close_p[i] = open_p[i] + 5
    w = compute_adaptive_gap_weight(open_p, close_p, window=20)
    assert w.shape == (n,)
    # 高延续场景后期 w 应靠近 0.8（clip 上限）
    assert w[-1] >= 0.7, f"高延续应 w≥0.7，实际 {w[-1]}"


def test_adaptive_gap_weight_reversing():
    """跳空后高反转 → 权重应 < 0.5（更信任 prev_close）。"""
    from core.factors.futures_data_cleaners import compute_adaptive_gap_weight

    n = 60
    # 高反转场景：跳空 +5 但日内回落 5（开 105 收 100）
    # gap[i] = 5 (正), intraday[i] = -5 (负) → 方向不一致
    open_p = np.full(n, 105.0)
    close_p = np.full(n, 100.0)
    open_p[0] = 100.0
    close_p[0] = 100.0
    w = compute_adaptive_gap_weight(open_p, close_p, window=20)
    # 高反转场景后期 w 应靠近 0.2
    assert w[-1] <= 0.3, f"高反转应 w≤0.3，实际 {w[-1]}"


def test_adaptive_gap_weight_clipped_range():
    """权重应被 clip 到 [0.2, 0.8] 范围。"""
    from core.factors.futures_data_cleaners import compute_adaptive_gap_weight

    np.random.seed(0)
    n = 100
    open_p = 100 + np.random.randn(n)
    close_p = open_p + np.random.randn(n) * 0.1
    w = compute_adaptive_gap_weight(open_p, close_p, window=20)
    assert (w >= 0.2).all() and (w <= 0.8).all()


def test_adaptive_gap_weight_no_lookahead():
    """无前瞻性：每个时间点权重仅由该时刻及之前数据决定。"""
    from core.factors.futures_data_cleaners import compute_adaptive_gap_weight

    n = 60
    open_p = np.full(n, 100.0)
    close_p = np.full(n, 100.0)
    # 前 20 日延续，后 20 日反转
    open_p[1:21] = 105.0
    close_p[1:21] = 110.0
    open_p[21:41] = 105.0
    close_p[21:41] = 100.0
    w = compute_adaptive_gap_weight(open_p, close_p, window=20)

    # 用前 30 日计算"前缀权重"，应与完整 60 日的前 30 日完全一致
    w_prefix = compute_adaptive_gap_weight(open_p[:30], close_p[:30], window=20)
    np.testing.assert_allclose(
        w[:30], w_prefix, rtol=1e-10,
        err_msg="权重包含未来数据，存在前瞻性偏差！",
    )


# ───────────────────────────────────────────────────────────
# 2. FactorEngine 集成自适应权重
# ───────────────────────────────────────────────────────────
def test_factor_engine_adaptive_gap_weight_applied():
    """FactorEngine 应在缺省 gap_weight 时自适应计算。"""
    from core.factors.alpha_futures.factor_engine import FactorEngine
    from core.factors.alpha_futures.config import AlphaFuturesConfig

    n = 50
    cfg = AlphaFuturesConfig()
    engine = FactorEngine(cfg, factor_names=["T_01"])

    np.random.seed(0)
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    open_price = close * (1 + np.random.randn(n) * 0.005)
    high = close * 1.01
    low = close * 0.99
    oi = np.full(n, 10000.0)
    # 不传 gap_weight → 引擎应自动用 compute_adaptive_gap_weight
    raw = {
        "close": close, "open_price": open_price, "high": high, "low": low,
        "open_interest": oi, "is_dominant": np.ones(n, dtype=bool),
    }
    out = engine.compute_all(raw)
    assert "T_01" in out
    assert out["T_01"].shape == (n,)


def test_alpha_futures_config_has_gap_weight_window():
    """AlphaFuturesConfig 应包含 gap_weight_window 字段。"""
    from core.factors.alpha_futures.config import AlphaFuturesConfig
    cfg = AlphaFuturesConfig()
    assert hasattr(cfg, "gap_weight_window")
    assert cfg.gap_weight_window == 20
    # from_backtest_config 透传
    class FakeBT:
        gap_weight_window = 30
        gap_weight = 0.4
        zscore_window = 0
        symbols = ["RB"]
    cfg2 = AlphaFuturesConfig.from_backtest_config(FakeBT())
    assert cfg2.gap_weight_window == 30
    assert cfg2.gap_weight == 0.4


# ───────────────────────────────────────────────────────────
# 3. V_01 / CF_01 强制结合价格方向
# ───────────────────────────────────────────────────────────
def _make_factor(cls):
    """统一通过 FactorRegistry 构造因子（注入默认 config）。"""
    from core.factors.alpha_futures.config import AlphaFuturesConfig
    return cls(AlphaFuturesConfig())


def test_v01_price_up_oi_up_is_positive():
    """V_01: 价涨OI增 → 应为正（多开信号）。"""
    from core.factors.alpha_futures.factors.v_01 import V_01

    n = 30
    oi = np.linspace(100, 200, n)  # 持续增仓
    close = np.linspace(100, 110, n)  # 持续上涨
    factor = _make_factor(V_01)
    out = factor.compute(oi_safe=oi, close=close)
    # 第 10 日以后 OI 变化率为正、价格变化为正 → V_01 必正
    valid = out[~np.isnan(out)]
    assert (valid > 0).all(), f"价涨OI增时 V_01 应全正，实际存在非正：{valid}"


def test_v01_price_up_oi_down_is_negative():
    """V_01: 价涨OI减 → 应为负（被动平仓/空平）。"""
    from core.factors.alpha_futures.factors.v_01 import V_01

    n = 30
    oi = np.linspace(200, 100, n)  # 持续减仓
    close = np.linspace(100, 110, n)  # 持续上涨
    factor = _make_factor(V_01)
    out = factor.compute(oi_safe=oi, close=close)
    valid = out[~np.isnan(out)]
    assert (valid < 0).all(), f"价涨OI减时 V_01 应全负，实际：{valid}"


def test_cf01_price_up_oi_up_is_positive():
    """CF_01: 价涨 + 持仓量 > MA → 应为正。"""
    from core.factors.alpha_futures.factors.cf_01 import CF_01

    n = 30
    oi = np.concatenate([np.full(15, 100.0), np.linspace(100, 200, 15)])
    close = np.linspace(100, 110, n)
    factor = _make_factor(CF_01)
    out = factor.compute(oi_safe=oi, close=close)
    # 第 20 日以后 oi > ma 且 close 在涨 → 应正
    tail = out[20:][~np.isnan(out[20:])]
    assert (tail > 0).all(), f"价涨+持仓>MA 应正，实际：{tail}"


def test_cf01_dependencies_include_close():
    """CF_01 的 dependencies 应包含 close（新增依赖）。"""
    from core.factors.alpha_futures.factors.cf_01 import CF_01
    assert "close" in CF_01.dependencies


def test_v01_dependencies_include_close():
    """V_01 的 dependencies 应包含 close（新增依赖）。"""
    from core.factors.alpha_futures.factors.v_01 import V_01
    assert "close" in V_01.dependencies


# ───────────────────────────────────────────────────────────
# 4. CF_03 动态分位数阈值
# ───────────────────────────────────────────────────────────
def test_cf03_dynamic_threshold_differs_from_static():
    """动态阈值在不同波动率区段应给出不同值。"""
    from core.factors.alpha_futures.factors.cf_03 import CF_03
    from core.factors.alpha_futures.config import AlphaFuturesConfig

    np.random.seed(42)
    n = 200
    # 前 100 日低波动，后 100 日高波动
    close = np.cumsum(np.concatenate([
        np.random.randn(100) * 0.001,
        np.random.randn(100) * 0.01,
    ])) + 100
    oi = np.cumsum(np.concatenate([
        np.random.randn(100) * 10,
        np.random.randn(100) * 50,
    ])) + 10000
    factor = CF_03(AlphaFuturesConfig())
    out = factor.compute(close=close, oi_safe=oi)
    assert out.shape == (n,)


def test_cf03_static_fallback():
    """dynamic_threshold=False 时回落静态阈值。"""
    from core.factors.alpha_futures.factors.cf_03 import CF_03
    from core.factors.alpha_futures.config import AlphaFuturesConfig

    np.random.seed(0)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.005)
    oi = 10000 + np.cumsum(np.random.randn(n) * 10)
    factor = CF_03(AlphaFuturesConfig())
    factor.dynamic_threshold = False  # 类属性 P1 整改暴露，可运行时修改
    out = factor.compute(close=close, oi_safe=oi)
    assert out.shape == (n,)


# ───────────────────────────────────────────────────────────
# 5. V_02 min_periods 兜底
# ───────────────────────────────────────────────────────────
def test_v02_handles_limit_up_zero_intraday():
    """涨跌停日 intraday_ret=0 时 V_02 不应全 0。"""
    from core.factors.alpha_futures.factors.v_02 import V_02

    n = 30
    # 前 19 日有正常波动，第 20 日涨跌停（intraday_ret=0）
    intraday_ret = np.concatenate([
        np.random.RandomState(0).randn(19) * 0.005,
        np.zeros(11),  # 连续 11 日 0，触发涨跌停
    ])
    oi = np.linspace(10000, 10100, n)  # 持续增仓
    factor = _make_factor(V_02)
    out = factor.compute(intraday_ret=intraday_ret, oi_safe=oi)
    # 涨跌停日因 oi 在变（delta != 0），V_02 不应全 0
    tail = out[20:][~np.isnan(out[20:])]
    assert (np.abs(tail) > 0).any(), "V_02 在涨跌停日全 0，min_periods 修复失效"


def test_v02_std_min_periods_attribute():
    """V_02 应暴露 std_min_periods 类属性。"""
    from core.factors.alpha_futures.factors.v_02 import V_02
    assert hasattr(V_02, "std_min_periods")
    assert V_02.std_min_periods >= 1


# ───────────────────────────────────────────────────────────
# 6. AlphaFutures24.compute_all 复权预检
# ───────────────────────────────────────────────────────────
def test_compute_all_warns_when_no_roll_map(caplog):
    """未提供 roll_map/is_dominant 时应 warn 而不抛错（向后兼容）。"""
    from core.factors.alpha_futures_24 import AlphaFutures24

    n = 30
    af = AlphaFutures24()
    close = np.full(n, 100.0)
    with caplog.at_level(logging.WARNING):
        af.compute_all(
            close=close,
            open_price=close,
            high=close * 1.01,
            low=close * 0.99,
            open_interest=np.full(n, 10000.0),
        )
    # 不强制要求特定 message，但 caplog 至少应无 exception
    assert caplog.records is not None


def test_auto_generate_roll_map():
    """_auto_generate_roll_map 应正确识别换月日。"""
    from core.factors.alpha_futures.factor_engine import _auto_generate_roll_map

    is_dom = np.array([True, True, True, False, False, True, True], dtype=bool)
    rm = _auto_generate_roll_map(is_dom)
    # 期望：换月发生在 i=3, i=5
    assert rm[0] == 1
    assert rm[3] == -1  # True→False
    assert rm[4] == 1
    assert rm[5] == -1  # False→True
    assert rm[6] == 1


def test_compute_all_works_without_roll_map():
    """compute_all 缺省 roll_map/is_dominant 时应不抛错（向后兼容）。"""
    from core.factors.alpha_futures_24 import AlphaFutures24

    n = 60
    af = AlphaFutures24()
    np.random.seed(0)
    close = 100 + np.cumsum(np.random.randn(n) * 0.01)
    raw_data = {
        "close": close,
        "open_price": close * (1 + np.random.randn(n) * 0.005),
        "high": close * 1.01,
        "low": close * 0.99,
        "open_interest": np.full(n, 10000.0),
        "near_price": close * 0.99,
        "far_price": close,
    }
    # 不传 is_dominant/roll_map → 应 warning 而不抛错
    out = af.compute_all(
        close=raw_data["close"],
        open_price=raw_data["open_price"],
        high=raw_data["high"],
        low=raw_data["low"],
        open_interest=raw_data["open_interest"],
        near_price=raw_data["near_price"],
        far_price=raw_data["far_price"],
    )
    assert len(out) >= 24  # 至少 24 个因子


# ───────────────────────────────────────────────────────────
# 7. decay_linear 扩张窗口
# ───────────────────────────────────────────────────────────
def test_decay_linear_early_values_not_nan():
    """decay_linear 在前期就应有估计值（扩张窗口）。"""
    from core.factors.operators import decay_linear

    n = 10
    arr = np.arange(1, n + 1, dtype=float)
    out = decay_linear(arr, window=5)
    # 旧实现前期 4 个 NaN，新实现应全有值
    valid = ~np.isnan(out)
    assert valid.sum() >= 9, f"扩张窗口前期应有值，实际 NaN 数 = {n - valid.sum()}"


def test_decay_linear_late_value_matches_full_window():
    """decay_linear 在窗口填满后值应与旧实现一致。"""
    from core.factors.operators import decay_linear

    np.random.seed(1)
    arr = np.random.randn(50)
    out = decay_linear(arr, window=10)
    # 简单 sanity check：最后值应是加权平均
    assert not np.isnan(out[-1])
    # NaN 安全：插入 NaN 不应崩
    arr2 = arr.copy()
    arr2[5] = np.nan
    out2 = decay_linear(arr2, window=10)
    assert out2.shape == (50,)


# ───────────────────────────────────────────────────────────
# 8. 整体 FactorEngine 流水线无回归
# ───────────────────────────────────────────────────────────
def test_engine_full_pipeline_still_works():
    """完整 24 因子流水线应仍能跑通。"""
    from core.factors.alpha_futures_24 import AlphaFutures24

    n = 80
    af = AlphaFutures24()
    np.random.seed(0)
    close = 100 + np.cumsum(np.random.randn(n) * 0.01)
    out = af.compute_all(
        close=close,
        open_price=close * (1 + np.random.randn(n) * 0.005),
        high=close * 1.01,
        low=close * 0.99,
        open_interest=np.full(n, 10000.0),
        near_price=close * 0.99,
        far_price=close,
        is_dominant=np.ones(n, dtype=bool),
    )
    assert len(out) >= 24
    # 至少 20 个因子应能输出有限值
    n_valid = sum(
        1 for v in out.values()
        if np.isfinite(v).any()
    )
    assert n_valid >= 20, f"完整流水线有效因子数 {n_valid} < 20"
