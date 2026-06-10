#!/usr/bin/env python3
"""
全量回测 E1/E2/E11/E9 实验 + baseline 对比。
"""

import sys
import os
import pandas as pd  # 必须在顶部以避免 UnboundLocalError
from datetime import datetime
from pathlib import Path

from loguru import logger


# 4 个目标实验
TARGET_EXPERIMENTS = ["e1", "e2", "e11", "e9"]


def main() -> int:
    from runner import Pipeline
    from runner.backtest.experiments import run_experiment

    print("=" * 80)
    print("  全量回测 E1/E2/E11/E9")
    print(f"  开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    pipe = Pipeline("config.yaml").load_data()
    config = pipe._config
    ds = pipe._data
    raw_config = pipe._raw_config

    # E1/E2/E9 走单策略 + 等权组合
    print("\n[1/4] E1: 单策略 baseline")
    e1_df = run_experiment("e1", config, ds, raw_config, cross_sectional=False)
    print(f"  E1 行数: {len(e1_df) if e1_df is not None else 0}")
    if e1_df is not None and not e1_df.empty:
        avg_sharpe = e1_df.get("sharpe", pd.Series([0])).astype(float).mean()
        print(f"  E1 avg_sharpe = {avg_sharpe:.4f}")

    print("\n[2/4] E2: 多品种等权组合")
    e2_df = run_experiment("e2", config, ds, raw_config, cross_sectional=False)
    print(f"  E2 行数: {len(e2_df) if e2_df is not None else 0}")
    if e2_df is not None and not e2_df.empty:
        avg_sharpe = e2_df.get("sharpe", pd.Series([0])).astype(float).mean()
        print(f"  E2 avg_sharpe = {avg_sharpe:.4f}")

    print("\n[3/4] E11: 因子分析")
    e11_df = run_experiment("e11", config, ds, raw_config, cross_sectional=True)
    print(f"  E11 行数: {len(e11_df) if e11_df is not None else 0}")

    print("\n[4/4] E9: 蒙特卡洛")
    e9_df = run_experiment("e9", config, ds, raw_config, cross_sectional=True)
    print(f"  E9 行数: {len(e9_df) if e9_df is not None else 0}")

    # Baseline 对比
    print("\n" + "=" * 80)
    print("  Baseline 对比（目标 E2 avg Sharpe ≥ +0.0332）")
    print("=" * 80)
    if e2_df is not None and not e2_df.empty:
        e2_avg = e2_df.get("sharpe", pd.Series([0])).astype(float).mean()
        target = 0.0332
        diff = e2_avg - target
        pct = (e2_avg / target - 1) * 100 if target else 0
        status = "✅ 达标" if diff >= 0 else "⚠️  需改进"
        print(f"  E2 当前 avg Sharpe = {e2_avg:.4f}")
        print(f"  Baseline target    = {target:.4f}")
        print(f"  差异               = {diff:+.4f} ({pct:+.1f}%)")
        print(f"  状态               = {status}")
    else:
        print("  E2 结果为空，无法对比")

    return 0


if __name__ == "__main__":
    sys.exit(main())
