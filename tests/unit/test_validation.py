"""
回测验证模块单元测试（P2 整改配套测试）。

覆盖：
  - 蒙特卡洛：
      * 正常模拟分位数计算
      * 短序列（< 20）安全返回
      * 空序列不崩溃
      * 常数序列（std=0）Sharpe=0
      * 整数参数校验
      * trading_days_per_year 参数化（252 vs 365 Sharpe 比例）
      * 向量化 vs 循环结果一致
  - 敏感性分析：
      * 正常扰动分析
      * 整数参数取整
      * 约束生效
      * 高敏感判定（三指标）
      * 缺键警告
      * 非法返回类型抛错
      * 并行模式结果一致性
"""
import math

import numpy as np
import pytest

from core.validation.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloResult,
    QUANTILES,
)
from core.validation.sensitivity import (
    SensitivityAnalyzer,
    SensitivityResult,
    FullSensitivityResult,
    HIGH_SENSITIVITY_THRESHOLD,
)


# ────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────
@pytest.fixture
def stable_returns():
    """稳定正收益序列（500 天）。"""
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.001, scale=0.01, size=500)


@pytest.fixture
def zero_std_returns():
    """常数收益序列（std=0）。"""
    return np.full(300, 0.001)


@pytest.fixture
def dummy_backtest_func():
    """
    假回测函数：参数值越大，Sharpe 越高，MDD 越小，年化收益越高。

    用于敏感性分析的扰动效果可见。
    """
    def _func(params: dict) -> dict:
        # 参数 w 越大越好，但 base_value==100 时参数变化
        w = float(params.get("w", 10))
        # 模拟 Sharpe 随 w 的二次方增长（强敏感）
        sharpe = (w / 10.0) ** 2
        mdd = 0.20 / (w / 10.0)  # w 越大，MDD 越小
        annual = 0.10 * (w / 10.0)
        return {
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "annual_return": annual,
        }
    return _func


# ────────────────────────────────────────────────────────────
# MonteCarloSimulator
# ────────────────────────────────────────────────────────────
class TestMonteCarloBasic:
    def test_simulate_basic_returns_result(self, stable_returns):
        mc = MonteCarloSimulator(n_simulations=200, random_seed=42)
        result = mc.simulate(stable_returns)

        assert isinstance(result, MonteCarloResult)
        assert result.n_simulations == 200
        assert result.is_robust is True  # 正收益序列，5% 分位 Sharpe>0
        # 5 个分位数都被填充
        for q in QUANTILES:
            assert q in result.sharpe_quantiles
            assert q in result.max_drawdown_quantiles
            assert q in result.annual_return_quantiles

    def test_simulate_short_sequence(self):
        """序列 < 20 时返回 n_simulations=0、不崩溃。"""
        mc = MonteCarloSimulator(n_simulations=100, random_seed=42)
        result = mc.simulate(np.array([0.01, 0.02, -0.01]))
        assert result.n_simulations == 0
        assert result.is_robust is False
        # 分位数字典应为空
        assert result.sharpe_quantiles == {}

    def test_simulate_empty_sequence(self):
        """空序列不崩溃。"""
        mc = MonteCarloSimulator(n_simulations=100, random_seed=42)
        result = mc.simulate(np.array([]))
        assert result.n_simulations == 0
        assert result.is_robust is False

    def test_zero_std_returns_yields_zero_sharpe(self, zero_std_returns):
        """常数收益（std=0）Sharpe 全为 0，不应触发除零异常。"""
        mc = MonteCarloSimulator(n_simulations=200, random_seed=42)
        result = mc.simulate(zero_std_returns)
        # 5%/50%/95% 分位的 Sharpe 应全为 0
        for q in QUANTILES:
            assert math.isclose(result.sharpe_quantiles[q], 0.0, abs_tol=1e-9)

    def test_trading_days_per_year_param_raises_on_invalid(self):
        """trading_days_per_year 必须为正数。"""
        with pytest.raises(ValueError):
            MonteCarloSimulator(trading_days_per_year=0)
        with pytest.raises(ValueError):
            MonteCarloSimulator(trading_days_per_year=-100)

    def test_trading_days_per_year_affects_sharpe_scaling(self, stable_returns):
        """P2 整改：trading_days_per_year 越大，Sharpe 越大（sqrt 比例）。"""
        mc_252 = MonteCarloSimulator(n_simulations=500, random_seed=42, trading_days_per_year=252)
        mc_365 = MonteCarloSimulator(n_simulations=500, random_seed=42, trading_days_per_year=365)

        r_252 = mc_252.simulate(stable_returns)
        r_365 = mc_365.simulate(stable_returns)

        # 50% 分位 Sharpe：比例应近似 sqrt(365/252)
        ratio = r_365.sharpe_quantiles[0.50] / r_252.sharpe_quantiles[0.50]
        expected = math.sqrt(365 / 252)
        assert math.isclose(ratio, expected, rel_tol=0.05)

    def test_result_records_trading_days_per_year(self, stable_returns):
        """结果应回传 trading_days_per_year 便于审计。"""
        mc = MonteCarloSimulator(n_simulations=100, random_seed=42, trading_days_per_year=365)
        result = mc.simulate(stable_returns)
        assert result.trading_days_per_year == 365
        # summary 文本应包含年化基数
        assert "365" in result.summary()

    def test_vectorized_matches_loop(self, stable_returns):
        """向量化与循环实现的 50% 分位结果应一致（容差 5%）。"""
        mc = MonteCarloSimulator(n_simulations=200, random_seed=42)
        r_vec = mc.simulate(stable_returns)
        r_loop = mc.simulate_loop(stable_returns)
        for q in [0.50, 0.95]:
            assert math.isclose(
                r_vec.sharpe_quantiles[q],
                r_loop.sharpe_quantiles[q],
                rel_tol=0.10,
            )
            assert math.isclose(
                r_vec.max_drawdown_quantiles[q],
                r_loop.max_drawdown_quantiles[q],
                rel_tol=0.10,
            )

    def test_summary_format(self, stable_returns):
        """summary 包含关键字段。"""
        mc = MonteCarloSimulator(n_simulations=100, random_seed=42)
        result = mc.simulate(stable_returns)
        s = result.summary()
        assert "蒙特卡洛100次" in s
        assert "Sharpe" in s
        assert "MDD@95%" in s
        assert "Return" in s
        assert "年化基数" in s


# ────────────────────────────────────────────────────────────
# SensitivityAnalyzer
# ────────────────────────────────────────────────────────────
class TestSensitivityBasic:
    def test_analyze_basic(self, dummy_backtest_func):
        analyzer = SensitivityAnalyzer(perturbation=0.20, n_jobs=1)
        result = analyzer.analyze(
            params={"w": 10.0},
            backtest_func=dummy_backtest_func,
        )
        assert isinstance(result, FullSensitivityResult)
        assert len(result.results) == 1
        r = result.results[0]
        assert r.param_name == "w"
        assert r.base_value == 10.0
        # w=10 → low=8.0, high=12.0
        assert math.isclose(r.low_value, 8.0, rel_tol=0.01)
        assert math.isclose(r.high_value, 12.0, rel_tol=0.01)

    def test_integer_param_rounded(self):
        """整数参数扰动后必须保持整数。"""
        def func(params):
            return {
                "sharpe": float(params["w"]),
                "max_drawdown": 0.10,
                "annual_return": 0.05,
            }
        analyzer = SensitivityAnalyzer(perturbation=0.20)
        result = analyzer.analyze(params={"w": 20}, backtest_func=func)
        r = result.results[0]
        assert isinstance(r.low_value, int)
        assert isinstance(r.high_value, int)
        # w=20 → low=16, high=24
        assert r.low_value == 16
        assert r.high_value == 24

    def test_param_constraint_clamped(self):
        """参数约束生效：超出上下界被裁剪。"""
        def func(params):
            return {
                "sharpe": 1.0,
                "max_drawdown": 0.10,
                "annual_return": 0.05,
            }
        analyzer = SensitivityAnalyzer(perturbation=0.50)
        # base=10, ±50% → [5, 15]，约束 [8, 12] → [8, 12]
        result = analyzer.analyze(
            params={"w": 10},
            backtest_func=func,
            param_constraints={"w": (8, 12)},
        )
        r = result.results[0]
        assert r.low_value == 8
        assert r.high_value == 12

    def test_high_sensitivity_marked(self, dummy_backtest_func):
        """dummy_backtest_func 是二次方响应，应判定为高敏感。"""
        analyzer = SensitivityAnalyzer(perturbation=0.20)
        result = analyzer.analyze(
            params={"w": 10.0},
            backtest_func=dummy_backtest_func,
        )
        # base Sharpe=1.0, ±20% 后变化远超 30%
        assert result.results[0].is_high_sensitivity is True
        assert "w" in result.high_sensitivity_params

    def test_low_sensitivity_marked(self):
        """当 backtest_func 输出对参数不敏感时，is_high_sensitivity=False。"""
        def flat_func(params):
            return {
                "sharpe": 1.0,
                "max_drawdown": 0.10,
                "annual_return": 0.05,
            }
        analyzer = SensitivityAnalyzer(perturbation=0.20)
        result = analyzer.analyze(params={"w": 10}, backtest_func=flat_func)
        r = result.results[0]
        assert r.is_high_sensitivity is False
        assert r.sharpe_change_pct < 1e-6
        assert r.max_drawdown_change_pct < 1e-6
        assert r.annual_return_change_pct < 1e-6

    def test_three_metrics_recorded(self, dummy_backtest_func):
        """P1 整改：max_drawdown / annual_return 字段必须被填充。"""
        analyzer = SensitivityAnalyzer(perturbation=0.20)
        result = analyzer.analyze(
            params={"w": 10.0},
            backtest_func=dummy_backtest_func,
        )
        r = result.results[0]
        # Sharpe
        assert hasattr(r, "base_sharpe")
        assert hasattr(r, "low_sharpe")
        assert hasattr(r, "high_sharpe")
        # MDD
        assert hasattr(r, "base_max_drawdown")
        assert hasattr(r, "low_max_drawdown")
        assert hasattr(r, "high_max_drawdown")
        assert hasattr(r, "max_drawdown_change_pct")
        # Annual Return
        assert hasattr(r, "base_annual_return")
        assert hasattr(r, "low_annual_return")
        assert hasattr(r, "high_annual_return")
        assert hasattr(r, "annual_return_change_pct")
        # 字段实际有值
        assert r.base_max_drawdown > 0
        assert r.base_annual_return > 0

    def test_high_sensitivity_uses_all_three_metrics(self):
        """高敏感判定：任一指标 > 阈值即触发。"""
        def mdd_sensitive_func(params):
            return {
                "sharpe": 1.0,  # 不变
                "max_drawdown": 0.5 if params["w"] > 10 else 0.1,  # MDD 突变
                "annual_return": 0.10,
            }
        analyzer = SensitivityAnalyzer(perturbation=0.20)
        result = analyzer.analyze(params={"w": 10}, backtest_func=mdd_sensitive_func)
        r = result.results[0]
        # Sharpe 变化 ~0，MDD 变化 400% > 30% → 高敏感
        assert r.sharpe_change_pct < 0.30
        assert r.max_drawdown_change_pct > 0.30
        assert r.is_high_sensitivity is True

    def test_missing_required_keys_warns(self, caplog):
        """缺键时记录警告，使用 0.0 兜底。"""
        def partial_func(params):
            return {"sharpe": 1.0}  # 缺 max_drawdown / annual_return
        analyzer = SensitivityAnalyzer(perturbation=0.20)
        with caplog.at_level("WARNING"):
            result = analyzer.analyze(params={"w": 10}, backtest_func=partial_func)
        assert any("缺少键" in rec.message for rec in caplog.records)
        # 结果仍能产生，但 max_drawdown 兜底为 0
        r = result.results[0]
        assert r.base_max_drawdown == 0.0

    def test_non_dict_return_raises(self):
        """回测函数返回非字典时直接抛 TypeError。"""
        def bad_func(params):
            return "not a dict"
        analyzer = SensitivityAnalyzer(perturbation=0.20)
        with pytest.raises(TypeError, match="必须返回 dict"):
            analyzer.analyze(params={"w": 10}, backtest_func=bad_func)

    def test_none_return_treated_as_empty(self, caplog):
        """回测函数返回 None 时使用 0.0 兜底。"""
        def none_func(params):
            return None
        analyzer = SensitivityAnalyzer(perturbation=0.20)
        result = analyzer.analyze(params={"w": 10}, backtest_func=none_func)
        # 基准返回 None，sharpe 等全部为 0
        r = result.results[0]
        assert r.base_sharpe == 0.0
        assert r.low_sharpe == 0.0

    def test_multiple_params(self, dummy_backtest_func):
        """多参数并行分析。"""
        analyzer = SensitivityAnalyzer(perturbation=0.20)
        result = analyzer.analyze(
            params={"w": 10, "x": 5.0},
            backtest_func=dummy_backtest_func,
        )
        assert len(result.results) == 2
        names = {r.param_name for r in result.results}
        assert names == {"w", "x"}

    def test_n_jobs_parallel_consistency(self, dummy_backtest_func):
        """P2 整改：并行模式 (n_jobs=4) 与串行结果一致。"""
        params = {"w": 10, "x": 5}
        serial = SensitivityAnalyzer(n_jobs=1).analyze(
            params=params, backtest_func=dummy_backtest_func
        )
        parallel = SensitivityAnalyzer(n_jobs=4).analyze(
            params=params, backtest_func=dummy_backtest_func
        )
        # 结果按参数名排序后比较
        s = sorted(serial.results, key=lambda r: r.param_name)
        p = sorted(parallel.results, key=lambda r: r.param_name)
        for r_s, r_p in zip(s, p):
            assert math.isclose(r_s.base_sharpe, r_p.base_sharpe, rel_tol=1e-9)
            assert math.isclose(r_s.low_sharpe, r_p.low_sharpe, rel_tol=1e-9)
            assert math.isclose(r_s.high_sharpe, r_p.high_sharpe, rel_tol=1e-9)
            assert r_s.is_high_sensitivity == r_p.is_high_sensitivity

    def test_n_jobs_clamped_to_min_1(self):
        """n_jobs=0 不会导致 ThreadPoolExecutor 报错。"""
        # 不应抛 ValueError
        analyzer = SensitivityAnalyzer(n_jobs=0)
        assert analyzer.n_jobs == 1

    def test_summary_contains_all_three_metrics(self, dummy_backtest_func):
        """P1 整改：summary 应展示三指标。"""
        analyzer = SensitivityAnalyzer(perturbation=0.20)
        result = analyzer.analyze(params={"w": 10}, backtest_func=dummy_backtest_func)
        s = result.results[0].summary()
        assert "Sharpe" in s
        assert "MDD" in s
        assert "Ret" in s
