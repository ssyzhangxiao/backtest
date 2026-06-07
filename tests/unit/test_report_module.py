"""
报告层单元测试：_convert_results / MetricsCalculator.aggregate_stats / exporters。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.engine.backtest_runner import PyBrokerResult
from runner.report.exporters import (
    export_metrics_summary,
    export_results_csv,
    export_validation_summary,
)
from runner.report.html_report import (
    _convert_dataframe_result,
    _convert_results,
)
from utils.metrics import MetricsCalculator


# ─── MetricsCalculator.aggregate_stats ─────────────────────────────────


def test_aggregate_stats_normal_series():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = MetricsCalculator.aggregate_stats(s)
    assert out["mean"] == 3.0
    assert out["min"] == 1.0
    assert out["max"] == 5.0
    # pandas std 默认 ddof=1（样本标准差）
    assert out["std"] == pytest.approx(s.std())


def test_aggregate_stats_with_nan():
    s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
    out = MetricsCalculator.aggregate_stats(s)
    assert out["mean"] == 3.0  # NaN 自动 drop
    assert out["min"] == 1.0
    assert out["max"] == 5.0


def test_aggregate_stats_empty_series():
    s = pd.Series([], dtype=float)
    out = MetricsCalculator.aggregate_stats(s)
    assert out == {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}


def test_aggregate_stats_all_nan():
    s = pd.Series([np.nan, np.nan])
    out = MetricsCalculator.aggregate_stats(s)
    assert out == {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}


def test_aggregate_stats_single_value():
    s = pd.Series([42.0])
    out = MetricsCalculator.aggregate_stats(s)
    assert out["mean"] == 42.0
    assert out["min"] == 42.0
    assert out["max"] == 42.0
    assert out["std"] == 0.0  # 单值 std=0


# ─── _convert_results ─────────────────────────────────────────────────


def test_convert_results_pynebroker_result():
    eq = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5), "equity": [1.0, 1.01, 1.02, 1.03, 1.04]})
    res = PyBrokerResult(
        metrics={"sharpe": 1.5, "total_return": 0.04},
        equity_curve=eq,
        trades=pd.DataFrame(),
        switch_log=pd.DataFrame(),
    )
    out = _convert_results({"trend": res})
    assert "trend" in out
    assert out["trend"]["metrics"]["sharpe"] == 1.5
    assert len(out["trend"]["dates"]) == 5
    assert out["trend"]["equity"][-1] == 1.04


def test_convert_results_dict():
    out = _convert_results({"strat1": {"metrics": {"sharpe": 0.5}}})
    assert "strat1" in out
    assert out["strat1"]["metrics"]["sharpe"] == 0.5


def test_convert_results_dataclass():
    @dataclass
    class MockResult:
        sharpe: float = 1.2
        total_return: float = 0.1

    out = _convert_results({"x": MockResult()})
    assert "x" in out
    assert out["x"]["metrics"]["sharpe"] == 1.2


def test_convert_results_dataframe_no_group():
    df = pd.DataFrame({"sharpe": [1.0, 2.0, 3.0], "total_return": [0.1, 0.2, 0.3]})
    out = _convert_results({"df_exp": df})
    assert "df_exp" in out
    m = out["df_exp"]["metrics"]
    # 验证委托 MetricsCalculator.aggregate_stats
    assert m["sharpe_mean"] == pytest.approx(2.0)
    assert m["sharpe_min"] == 1.0
    assert m["sharpe_max"] == 3.0
    assert m["total_return_mean"] == pytest.approx(0.2)


def test_convert_results_dataframe_with_group():
    df = pd.DataFrame({
        "strategy": ["A", "A", "B", "B"],
        "sharpe": [1.0, 1.5, 0.8, 1.2],
    })
    out = _convert_results({"df_exp": df})
    assert "A" in out
    assert "B" in out
    assert out["A"]["metrics"]["sharpe_mean"] == pytest.approx(1.25)
    assert out["B"]["metrics"]["sharpe_mean"] == pytest.approx(1.0)


def test_convert_results_recursive_all_key():
    eq = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2), "equity": [1.0, 1.01]})
    inner_res = PyBrokerResult(
        metrics={"sharpe": 0.8}, equity_curve=eq,
        trades=pd.DataFrame(), switch_log=pd.DataFrame(),
    )
    out = _convert_results({"all": {"x": inner_res}})
    assert "x" in out
    assert out["x"]["metrics"]["sharpe"] == 0.8


def test_convert_results_skip_none():
    out = _convert_results({"a": None, "b": {"metrics": {"sharpe": 0.1}}})
    assert "a" not in out
    assert "b" in out


# ─── _convert_dataframe_result 直接测试 ──────────────────────────────


def test_convert_dataframe_result_excludes_date_column():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "sharpe": [0.5, 0.6, 0.7],
    })
    out: dict = {}
    _convert_dataframe_result("exp1", df, out)
    # date 列不应进入统计
    assert "date_mean" not in out["exp1"]["metrics"]
    assert out["exp1"]["metrics"]["sharpe_mean"] == pytest.approx(0.6)


# ─── exporters ────────────────────────────────────────────────────────


def test_export_results_csv_dataframe(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    paths = export_results_csv({"my_df": df}, tmp_path)
    assert len(paths) == 1
    assert paths[0].exists()
    df2 = pd.read_csv(paths[0])
    assert len(df2) == 3


def test_export_results_csv_dict(tmp_path: Path):
    paths = export_results_csv({"my_dict": {"x": 1, "y": 2}}, tmp_path)
    assert len(paths) == 1
    df = pd.read_csv(paths[0])
    assert set(["x", "y"]).issubset(set(df.columns))


def test_export_results_csv_skips_none(tmp_path: Path):
    paths = export_results_csv({"a": None, "b": pd.DataFrame({"c": [1]})}, tmp_path)
    assert len(paths) == 1


def test_export_metrics_summary(tmp_path: Path):
    out = export_metrics_summary(
        {"s1": {"sharpe": 1.0, "return": 0.1}, "s2": {"sharpe": 0.8, "return": 0.05}},
        tmp_path / "summary.csv",
    )
    assert out is not None
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == 2
    assert "strategy" in df.columns


def test_export_validation_summary(tmp_path: Path):
    summary = export_validation_summary({"task1": {"compare": pd.DataFrame()}}, tmp_path)
    assert summary.exists()
    assert "验证完成时间" in summary.read_text(encoding="utf-8")
