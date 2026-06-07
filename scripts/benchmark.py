"""
性能基准测试脚本。

对核心路径进行微基准测试：
  - 因子综合得分合成（带 symbol）
  - 调仓日判断（基于日期）
  - 数据查询

用法: PYTHONPATH=. python scripts/benchmark.py
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core.engine.switch_engine import FactorScoringEngine, ScoringConfig
from core.config.strategy_profiles import StrategyLibrary


def generate_fake_data(n_bars: int = 1000) -> pd.DataFrame:
    """生成模拟行情数据。"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="B")
    price = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    return pd.DataFrame({
        "date": dates,
        "symbol": "TEST",
        "open": price * (1 - np.random.rand(n_bars) * 0.01),
        "high": price * (1 + np.random.rand(n_bars) * 0.02),
        "low": price * (1 - np.random.rand(n_bars) * 0.02),
        "close": price,
        "volume": np.random.randint(10000, 100000, n_bars).astype(float),
        "is_dominant": True,
    })


def benchmark_composite_score(n_runs: int = 100000):
    """综合得分计算基准测试。

    当前 API: FactorScoringEngine.compute_composite_score(symbol, factor_scores)
    """
    library = StrategyLibrary()
    engine = FactorScoringEngine(library)
    symbol = "SHFE.RB"
    factor_scores = {
        "trend": 0.5,
        "term_structure": -0.3,
        "mean_reversion": 0.8,
        "vol_breakout": 0.1,
    }

    start = time.perf_counter()
    for _ in range(n_runs):
        engine.compute_composite_score(symbol, factor_scores)
    elapsed = time.perf_counter() - start

    ns_per_op = (elapsed / n_runs) * 1e9
    print(f"  compute_composite_score: {n_runs:,} 次 / {elapsed:.3f}s = {ns_per_op:.0f} ns/次")


def benchmark_rebalance_check(n_runs: int = 100000):
    """调仓日判断基准测试。

    当前 API: FactorScoringEngine.is_rebalance_day(dt)，仅接受日期参数。
    为了避免 pandas DatetimeIndex 纳秒级溢出（n_runs 个工作日 > 1677 年），
    改用有限日期池 + 循环复用，仅测量纯判定开销。
    """
    library = StrategyLibrary()
    config = ScoringConfig(rebalance_days=3)
    engine = FactorScoringEngine(library, config)
    # 1000 个工作日（≈ 4 年）足够覆盖调仓周期
    dates = pd.date_range("2023-01-01", periods=1000, freq="B")
    n_dates = len(dates)

    start = time.perf_counter()
    for i in range(n_runs):
        engine.is_rebalance_day(dates[i % n_dates])
    elapsed = time.perf_counter() - start

    ns_per_op = (elapsed / n_runs) * 1e9
    print(f"  is_rebalance_day:        {n_runs:,} 次 / {elapsed:.3f}s = {ns_per_op:.0f} ns/次")


def benchmark_data_query(n_runs: int = 1000):
    """数据查询基准测试。"""
    from core.engine.pybroker_data_source import PyBrokerDataSource

    df = generate_fake_data(252 * 5)
    ds = PyBrokerDataSource(df)

    start = time.perf_counter()
    for _ in range(n_runs):
        ds.query("2023-01-01", "2023-06-30")
    elapsed = time.perf_counter() - start

    ms_per_op = (elapsed / n_runs) * 1000
    print(f"  PyBrokerDataSource.query: {n_runs:,} 次 / {elapsed:.3f}s = {ms_per_op:.3f} ms/次")


def main():
    print("=" * 60)
    print("性能基准测试")
    print("=" * 60)
    print()

    benchmark_composite_score()
    benchmark_rebalance_check()
    benchmark_data_query()

    print()
    print("=" * 60)
    print("基准测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
