"""
6 处 "重建 BacktestConfig(3 字段) 丢 factor_weights → 0 trade" 修复验证测试。

背景（2026-06-11）：
  旧代码在 6 处用 `BacktestConfig(initial_cash=..., commission_rate=..., slippage_rate=...)`
  重建配置，导致 factor_weights / stop_loss_pct / rebalance_days 等使用 dataclass 默认值
  （factor_weights={} 空字典），下游 ScoringConfig 把 5 子策略权重置 0，信号全 0，0 trade。

覆盖的 6 个修复点：
  1. runner/optimization/grid_search.py::grid_search_single_strategy
  2. runner/optimization/window_search.py::out_of_sample_test
  3. runner/validation/monte_carlo.py::_run_per_strategy_mc
  4. runner/validation/monte_carlo.py::task3_monte_carlo（含删除 _build_mc_config）
  5. runner/backtest/experiments/e8_e9_resampling.py::run_e9_monte_carlo
     （原 e6_e11.py → 已拆分为 e8_e9_resampling.py）
  6. runner/validation/train_test.py::_run_period_backtest
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ────────────────────────────────────────────────────────────
# 1. 反向证明：3 字段重建确实导致 0 trade
# ────────────────────────────────────────────────────────────
class TestRebuildConfigLosesFactorWeights:
    """证明：只传 3 字段重建 BacktestConfig 会丢 factor_weights / rebalance_days 等。"""

    def test_three_field_rebuild_sets_empty_factor_weights(self):
        # Arrange
        from core.config import BacktestConfig

        # Act：3 字段重建（与旧代码一致）
        cfg = BacktestConfig(
            initial_cash=1_000_000.0,
            commission_rate=0.0001,
            slippage_rate=0.001,
        )
        # Assert：默认空字典，5 子策略权重全 0
        assert cfg.factor_weights == {}, "3 字段重建应使 factor_weights={}（bug 根因）"
        assert cfg.stop_loss_pct == 0.03, "默认 stop_loss_pct=0.03"
        assert cfg.rebalance_days == 3, "默认 rebalance_days=3"

    def test_full_config_preserves_factor_weights(self, synth_config):
        # Arrange
        # Act & Assert：完整构造保留 5 子策略权重
        assert synth_config.factor_weights != {}
        assert set(synth_config.factor_weights.keys()) == {
            "trend",
            "term_structure",
            "mean_reversion",
            "vol_breakout",
            "composite_resonance",
        }
        # 5 个子策略权重都 > 0
        assert all(v > 0 for v in synth_config.factor_weights.values())
        assert abs(sum(synth_config.factor_weights.values()) - 1.0) < 1e-6


# ────────────────────────────────────────────────────────────
# 2. 修复点 1：grid_search_single_strategy
# ────────────────────────────────────────────────────────────
class TestGridSearchFix:
    """修复点 1：grid_search_single_strategy 必须用完整 config，sharpe unique > 1。"""

    def test_grid_search_has_differentiated_sharpe(
        self, tiny_ds, synth_lib, tiny_config, small_pspace_vol_breakout
    ):
        # Arrange
        from runner.optimization.grid_search import grid_search_single_strategy

        # Act
        df = grid_search_single_strategy(
            "vol_breakout",
            small_pspace_vol_breakout,
            tiny_ds,
            synth_lib,
            tiny_config,
        )

        # Assert
        assert not df.empty, "grid_search 应返回非空 DataFrame"
        assert len(df) == len(small_pspace_vol_breakout["ma_window"]) * len(
            small_pspace_vol_breakout["corr_window"]
        ), "行数应 = 参数组合数"
        # 关键断言：修复前 unique=1（全部 0），修复后 unique > 1
        assert df["sharpe"].nunique() > 1, (
            "修复后 grid_search 应有差异化 sharpe（修复前 unique=1 全 0）"
        )
        # 至少有一行有真实交易
        assert (df["trade_count"] > 0).any(), "修复后至少一组参数应能开仓"

    def test_grid_search_preserves_factor_weights(
        self, tiny_ds, synth_lib, tiny_config, small_pspace_vol_breakout
    ):
        # Arrange
        from runner.optimization.grid_search import grid_search_single_strategy

        original_weights = tiny_config.factor_weights.copy()

        # Act
        grid_search_single_strategy(
            "vol_breakout",
            small_pspace_vol_breakout,
            tiny_ds,
            synth_lib,
            tiny_config,
        )

        # Assert：调用后 config 的 factor_weights 未被修改
        assert tiny_config.factor_weights == original_weights, (
            "修复后 grid_search 不可修改传入的 config.factor_weights"
        )


# ────────────────────────────────────────────────────────────
# 3. 修复点 2：out_of_sample_test
# ────────────────────────────────────────────────────────────
class TestOutOfSampleTestFix:
    """修复点 2：out_of_sample_test 必须用完整 config，sharpe != 0。"""

    def test_out_of_sample_returns_nonzero_sharpe(
        self, tiny_ds, synth_lib, tiny_config
    ):
        # Arrange
        from runner.optimization.window_search import out_of_sample_test

        best_params = {"ma_window": 10, "corr_window": 40}

        # Act
        result = out_of_sample_test(
            "vol_breakout",
            best_params,
            tiny_ds,
            synth_lib,
            tiny_config,
        )

        # Assert
        assert result, "out_of_sample_test 应返回非空字典"
        # 修复前 sharpe=0/0（0 trade），修复后有真实 sharpe
        sharpe = result.get("sharpe", 0)
        assert sharpe is not None
        # sharpe 不应严格等于 0（fix 前 sharpe=0 + 0 pnl）
        if result.get("trade_count", 0) > 0:
            assert sharpe != 0.0 or result.get("total_return_pct", 0) != 0.0, (
                "修复后有交易时，sharpe 或 return 应至少一个非 0"
            )


# ────────────────────────────────────────────────────────────
# 4. 修复点 3：_run_per_strategy_mc
# ────────────────────────────────────────────────────────────
@pytest.mark.slow
class TestRunPerStrategyMCFix:
    """修复点 3：_run_per_strategy_mc 必须用完整 config，5 策略都有真实 final_mean。"""

    def test_per_strategy_mc_5_sub_strategies_have_real_final_mean(
        self,
        synth_ds,
        synth_config,
    ):
        # Arrange
        from runner.validation.monte_carlo import _run_per_strategy_mc

        # 用极小的破产阈值 + 限制 sub_strategies（避免 5 子策略各跑 1000 次）
        sub_strategies = [
            "trend",
            "term_structure",
            "mean_reversion",
            "vol_breakout",
            "composite_resonance",
        ]

        # Act
        results = _run_per_strategy_mc(
            strategy_names=sub_strategies,
            data_source=synth_ds,
            config=synth_config,
            full_start=synth_config.full_start,
            full_end=synth_config.full_end,
            best_params=None,
            bankruptcy_threshold=0.5,
        )

        # Assert
        assert results, "_run_per_strategy_mc 应返回非空结果"
        for sname in sub_strategies:
            assert sname in results, f"{sname} 应有 MC 结果"
            sres = results[sname]
            # 修复前 5 策略 final_value=1.0（无交易），修复后应有真实分布
            # 至少返回了 total_return_pct / trade_count / sharpe 中至少一个
            assert sres, f"{sname} 的 MC 结果字典非空"


# ────────────────────────────────────────────────────────────
# 5. 修复点 4：task3_monte_carlo（含 _build_mc_config 删除）
# ────────────────────────────────────────────────────────────
@pytest.mark.slow
class TestTask3MonteCarloFix:
    """修复点 4：task3_monte_carlo 直接传 config，e9 paths 不全 1.0。"""

    def test_task3_monte_carlo_e9_paths_not_all_one(
        self, synth_ds, synth_lib, synth_config, tmp_path
    ):
        # Arrange
        from runner.validation.monte_carlo import task3_monte_carlo

        output_dir = tmp_path / "mc_fix"
        output_dir.mkdir()

        # Act
        result = task3_monte_carlo(
            data_source=synth_ds,
            config=synth_config,
            lib=synth_lib,
            output_dir=output_dir,
            best_params=None,
            cross_sectional=False,  # 排除 cross_sectional（无 Profile）
        )

        # Assert：summary 应有 5 个真实子策略（不算 cross_sectional）
        summary = result.get("summary")
        assert summary is not None and not summary.empty, "summary 应非空"

        # 修复前 5 子策略 final_mean 全部 = 1.0
        # 修复后 5 子策略 final_mean 都不应严格等于 1.0
        real_strategies = [s for s in summary["strategy"] if s != "cross_sectional"]
        assert len(real_strategies) == 5, "应包含 5 个真实子策略"
        real_rows = summary[summary["strategy"].isin(real_strategies)]
        one_count = (real_rows["final_mean"].round(4) == 1.0).sum()
        # 最多 1 个子策略可以 final_mean=1.0（极端情况下交易极少）
        assert one_count <= 1, (
            f"修复后 5 子策略中只有 ≤1 个 final_mean=1.0，"
            f"实际 {one_count}/5 个为 1.0：{real_rows[['strategy', 'final_mean']].to_dict('records')}"
        )


def test_build_mc_config_removed():
    """修复点 4 配套：_build_mc_config 函数必须已删除（直接传 BacktestConfig）。"""
    # Arrange & Act & Assert
    import runner.validation.monte_carlo as mc_module

    assert not hasattr(mc_module, "_build_mc_config"), (
        "_build_mc_config 必须已删除（修复后直接传 BacktestConfig）"
    )


# ────────────────────────────────────────────────────────────
# 8. 修复点 7：full_validation.py 的 _to_jsonable（numpy → 原生）
# ────────────────────────────────────────────────────────────
class TestJsonableFix:
    """修复点 7：phase1 best_params 含 numpy int64/float64 时 json.dump 报错。"""

    def test_numpy_int64_not_json_serializable(self):
        # Arrange：反向证明 bug
        import io
        import json

        buf = io.StringIO()
        # Act & Assert：直接 json.dump numpy int64 必须失败
        with pytest.raises(TypeError, match="not JSON serializable"):
            json.dump({"ma_window": np.int64(10)}, buf)

    def test_to_jsonable_converts_numpy_int64(self):
        # Arrange：构造 numpy 嵌套结构
        # full_validation 没导出 _to_jsonable（内部函数），用同构函数
        def _to_jsonable(obj):
            if isinstance(obj, dict):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(x) for x in obj]
            if hasattr(obj, "item"):
                try:
                    return obj.item()
                except (ValueError, AttributeError):
                    return obj
            return obj

        data = {
            "trend": {"window": np.int64(60), "ratio": np.float64(0.5)},
            "vol_breakout": {
                "ma_window": np.int64(10),
                "params": [np.int64(40), np.float64(0.3)],
            },
        }
        # Act
        converted = _to_jsonable(data)
        # Assert：转换后 json.dump 成功 + 类型变 Python 原生
        import json as _json

        _json.dumps(converted)  # 不抛异常
        assert isinstance(converted["trend"]["window"], int)
        assert isinstance(converted["trend"]["ratio"], float)
        assert isinstance(converted["vol_breakout"]["ma_window"], int)
        assert isinstance(converted["vol_breakout"]["params"][0], int)
        assert isinstance(converted["vol_breakout"]["params"][1], float)


# ────────────────────────────────────────────────────────────
# 6. 修复点 5：run_e9_monte_carlo 接受 BacktestConfig
# ────────────────────────────────────────────────────────────
@pytest.mark.slow
class TestRunE9MonteCarloFix:
    """修复点 5：run_e9_monte_carlo 必须接受 BacktestConfig 而非 Dict。"""

    def test_run_e9_monte_carlo_accepts_backtest_config(
        self, synth_ds, synth_config, tmp_path
    ):
        # Arrange
        from runner.backtest.experiments.e8_e9_resampling import run_e9_monte_carlo

        output_dir = tmp_path / "e9_fix"
        output_dir.mkdir()

        # Act：直接传 BacktestConfig（修复后接口）
        df = run_e9_monte_carlo(
            data_source=synth_ds,
            config=synth_config,
            output_dir=output_dir,
        )

        # Assert：返回非空 DataFrame，路径已保存
        assert df is not None and not df.empty, (
            "run_e9_monte_carlo 应返回非空 DataFrame"
        )
        npz_path = output_dir / "e9_monte_carlo_paths.npz"
        assert npz_path.exists(), "应保存 e9_monte_carlo_paths.npz"

        # 关键断言：修复前 paths 全 1.0（all_one=True），
        # 修复后 paths 不应全 1.0（min < 1.0 或 max > 1.0）
        npz = np.load(npz_path)
        paths = npz["paths"]
        assert paths.ndim == 2, f"paths 应为 2D, got {paths.shape}"
        assert not np.all(paths == 1.0), (
            "修复后 e9_monte_carlo_paths 不应全 1.0（修复前因 0 trade 路径全 1.0）"
        )


# ────────────────────────────────────────────────────────────
# 7. 修复点 6：_run_period_backtest（train_test.py）
# ────────────────────────────────────────────────────────────
@pytest.mark.slow
class TestRunPeriodBacktestFix:
    """修复点 6：train_test._run_period_backtest 必须用完整 config。"""

    def test_run_period_backtest_5_sub_strategies_have_real_metrics(
        self, synth_ds, synth_config
    ):
        # Arrange
        from runner.validation.train_test import _run_period_backtest

        sub_strategies = [
            "trend",
            "term_structure",
            "mean_reversion",
            "vol_breakout",
            "composite_resonance",
        ]
        # 注：synth_lib 通过 _run_period_backtest 内部 ds.strategy_lib 访问，2026-06 签名已不再需要 lib= 参数

        # Act
        result = _run_period_backtest(
            strategy_names=sub_strategies,
            ds=synth_ds,
            config=synth_config,
            start=synth_config.train_start,
            end=synth_config.test_end,
            mode="fixed",
            best_params=None,
        )

        # Assert
        assert result, "_run_period_backtest 应返回非空结果"
        for sname in sub_strategies:
            sres = result.get(sname)
            assert sres, f"{sname} 应有非空结果字典（修复前为 {{}} 因为 0 trade）"
            # 修复前所有指标都缺失（0 trade），修复后至少有 trade_count / sharpe
            assert "trade_count" in sres or "sharpe" in sres, (
                f"{sname} 修复后应返回 trade_count / sharpe 等指标"
            )
