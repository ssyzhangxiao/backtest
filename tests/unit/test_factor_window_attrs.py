"""P1-3 验证：所有硬编码窗口已参数化为类属性。"""
import numpy as np

from core.ext.factors.alpha_futures.factors import (
    cf_01, cf_02, cf_03,
    h_01, h_02, h_03, h_04, h_05,
    m_01, m_02, m_03, m_04, m_05,
    r_01, r_02, r_03, r_04, r_05,
    t_01, t_02, t_03, t_04, t_05,
    v_01, v_02, v_03, v_04,
    ts_01, ts_02, ts_03,
)


def make_inputs(n=120):
    np.random.seed(42)
    close = np.cumsum(np.random.randn(n)) + 100
    oi = np.abs(np.random.randn(n)) * 1000 + 5000
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    volume = np.abs(np.random.randn(n)) * 100 + 1000
    intraday = np.random.randn(n) * 0.01
    carry = np.random.randn(n) * 0.05
    open_adj = close + np.random.randn(n) * 0.1
    near = close - 0.5
    far = close + 0.5
    return close, oi, high, low, volume, intraday, carry, open_adj, near, far


def test_factor_class_has_window_attrs():
    """检查所有因子都暴露窗口类属性。"""
    inputs = make_inputs()
    close, oi, high, low, volume, intraday, carry, open_adj, near, far = inputs

    factors = [
        (cf_01.CF_01, {"oi_safe": oi, "close": close}),  # P整改：CF_01 新增 close 依赖
        (cf_02.CF_02, {"close": close, "volume": volume}),
        (cf_03.CF_03, {"close": close, "oi_safe": oi}),
        (h_01.H_01, {"close": close, "oi_safe": oi, "carry": carry}),
        (h_02.H_02, {"close": close, "oi_safe": oi}),
        (h_03.H_03, {"close": close, "oi_safe": oi}),
        (h_04.H_04, {"close": close, "oi_safe": oi}),
        (h_05.H_05, {"carry": carry, "oi_safe": oi, "close": close}),
        (m_01.M_01, {"close": close, "high": high, "low": low, "oi_safe": oi}),
        (m_02.M_02, {"close": close, "high": high, "low": low, "oi_safe": oi}),
        (m_03.M_03, {"close": close, "oi_safe": oi}),
        (m_04.M_04, {"carry": carry, "oi_safe": oi}),
        (m_05.M_05, {"oi_safe": oi}),
        (r_01.R_01, {"oi_safe": oi, "intraday_ret": intraday}),
        (r_02.R_02, {"high": high, "oi_safe": oi}),
        (r_03.R_03, {"close": close, "open_adj": open_adj, "oi_safe": oi}),
        (r_04.R_04, {"carry": carry}),
        (r_05.R_05, {"oi_safe": oi}),
        (t_01.T_01, {"close": close, "oi_safe": oi}),
        (t_02.T_02, {"close": close, "oi_safe": oi}),
        (t_03.T_03, {"close": close, "oi_safe": oi}),
        (t_04.T_04, {"carry": carry, "oi_safe": oi}),
        (t_05.T_05, {"close": close, "oi_safe": oi}),
        (v_01.V_01, {"oi_safe": oi, "close": close}),  # P整改：V_01 新增 close 依赖
        (v_02.V_02, {"intraday_ret": intraday, "oi_safe": oi}),
        (v_03.V_03, {"high": high, "low": low, "oi_safe": oi}),
        (v_04.V_04, {"oi_safe": oi}),
        (ts_01.TS_01, {"near_price": near, "far_price": far, "close": close}),
        (ts_02.TS_02, {"near_price": near, "far_price": far, "close": close}),
        (ts_03.TS_03, {"near_price": near, "far_price": far, "close": close}),
    ]

    # 这些因子无窗口（直接计算）或全部由 kwargs 注入
    skip_attrs_check = {r_04.R_04, ts_01.TS_01, ts_02.TS_02}

    for cls, kwargs in factors:
        if cls in skip_attrs_check:
            inst = cls(config=None)
            result = inst.compute(**kwargs)
            assert len(result) == 120, f"{cls.__name__} 输出长度错误：{len(result)}"
            continue
        # 类属性可在不实例化的情况下访问
        window_attrs = [
            k for k in dir(cls)
            if (("window" in k.lower() or "period" in k.lower() or k.endswith("_ma"))
                and not k.startswith("_"))
        ]
        assert len(window_attrs) >= 1, f"{cls.__name__} 缺少窗口类属性"
        inst = cls(config=None)
        result = inst.compute(**kwargs)
        assert len(result) == 120, f"{cls.__name__} 输出长度错误：{len(result)}"

    print(f"PASS: {len(factors)} factors have class-level window params and produce correct output")


if __name__ == "__main__":
    test_factor_class_has_window_attrs()
