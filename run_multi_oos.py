#!/usr/bin/env python3
"""
多窗口 OOS 验证 — 子策略版。

用 3 个 OOS 窗口（2022/2023/2024）跑各子策略回测，对比 Sharpe 衰减率。
绕过 cross_sectional（其 E11 依赖过重），直接验证 5 子策略 + 等权组合。
"""

import sys
import os
from datetime import datetime
from pathlib import Path

from loguru import logger


# 3 个 OOS 窗口定义：(窗口名, test_start, test_end)
OOS_WINDOWS = [
    ("OOS_2022", "2022-01-01", "2022-12-31"),
    ("OOS_2023", "2023-01-01", "2023-12-31"),
    ("OOS_2024", "2024-01-01", "2024-12-31"),
]

# 5 个子策略 + 等权组合
STRATEGIES = ["trend", "term_structure", "mean_reversion", "vol_breakout", "composite_resonance"]


def run_oos_window(pipe, test_start: str, test_end: str) -> dict:
    """
    在 OOS 窗口内对 5 子策略做回测，提取 Sharpe/Return/MaxDD。
    """
    from core.engine.backtest_runner import PyBrokerBacktestRunner

    ds = pipe._data
    config = pipe._config

    results = {}
    for sname in STRATEGIES:
        try:
            runner = PyBrokerBacktestRunner(ds, config)
            runner.register_strategies([sname])
            res = runner.run(start_date=test_start, end_date=test_end)
            metrics = res.metrics if res and hasattr(res, "metrics") else {}
            results[sname] = {
                "sharpe": round(metrics.get("sharpe", 0.0) or 0.0, 4),
                "total_return": round(metrics.get("total_return", 0.0) or 0.0, 4),
                "max_drawdown": round(metrics.get("max_drawdown", 0.0) or 0.0, 4),
                "trade_count": int(metrics.get("trade_count", 0) or 0),
            }
        except Exception as e:
            logger.error(f"{sname} OOS 回测失败: {e}")
            results[sname] = {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "trade_count": 0, "error": str(e)}

    # 等权组合：各策略 Sharpe 等权平均
    valid_sharpes = [v["sharpe"] for v in results.values() if v.get("trade_count", 0) > 0]
    results["EW_BLEND"] = {
        "sharpe": round(sum(valid_sharpes) / len(valid_sharpes), 4) if valid_sharpes else 0.0,
        "total_trades": sum(v.get("trade_count", 0) for v in results.values()),
    }
    return results


def main() -> int:
    from runner import Pipeline

    print("=" * 80)
    print("  多窗口 OOS 验证 — 5 子策略 + 等权组合 (2022/2023/2024)")
    print(f"  开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    pipe = Pipeline("config.yaml").load_data()

    summary = {}
    for window_name, test_start, test_end in OOS_WINDOWS:
        print(f"\n{'#' * 80}")
        print(f"# 窗口: {window_name} | OOS: {test_start} ~ {test_end}")
        print(f"{'#' * 80}")

        res = run_oos_window(pipe, test_start, test_end)
        summary[window_name] = res
        print(f"\n{window_name} 5 子策略 + EW:")
        print(f"  {'策略':<22}{'Sharpe':<10}{'Return':<10}{'MaxDD':<10}{'Trades':<8}")
        for sname, m in res.items():
            if sname == "EW_BLEND":
                print(f"  {sname:<22}avg_sharpe={m['sharpe']:.4f}  total_trades={m['total_trades']}")
            else:
                print(f"  {sname:<22}{m['sharpe']:<10}{m['total_return']:<10}{m['max_drawdown']:<10}{m['trade_count']:<8}")

    # 汇总
    print("\n" + "=" * 80)
    print("  多窗口 OOS 汇总：5 子策略 Sharpe 演化")
    print("=" * 80)
    print(f"{'策略':<22}" + "".join(f"{w:<14}" for w, _, _ in OOS_WINDOWS))
    print("-" * 80)
    for sname in STRATEGIES:
        row = f"{sname:<22}"
        for w, _, _ in OOS_WINDOWS:
            sh = summary[w].get(sname, {}).get("sharpe", 0.0)
            row += f"{sh:<14.4f}"
        print(row)
    # 等权组合
    row = f"{'EW_BLEND':<22}"
    for w, _, _ in OOS_WINDOWS:
        sh = summary[w].get("EW_BLEND", {}).get("sharpe", 0.0)
        row += f"{sh:<14.4f}"
    print("-" * 80)
    print(row)
    print("=" * 80)

    # 保存
    import json
    out_dir = Path("output_backtest_pybroker/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "multi_window_oos_substrategies.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {out_dir / 'multi_window_oos_substrategies.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
