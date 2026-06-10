#!/usr/bin/env python3
"""
全量验证脚本：3 阶段串行执行（调参 → 6 品种 EW 横截面 → 蒙特卡洛）

时间划分：
  - in_sample (训练): 2020-01-01 ~ 2022-12-31
  - out_sample (OOS): 2023-01-01 ~ 2024-12-31
  - 蒙特卡洛使用全段: 2020-01-01 ~ 2024-12-31

6 品种：SHFE.AL, SHFE.CU, CZCE.FG, SHFE.RU, DCE.PP, CZCE.CF

三阶段：
  Phase 1 - 全量调参：在 in_sample 区间对 5 子策略网格搜索 → OOS 优选
  Phase 2 - 6 品种横截面 EW 组合回测：用 OOS 最优参数在 in_sample 和 OOS 段分别跑
  Phase 3 - 蒙特卡洛 1000 次鲁棒性测试：在全段对 OOS 最优参数做扰动

输出:
  output_backtest_pybroker/full_validation/
    ├── phase1_best_params.json
    ├── phase2_ew_results.csv
    ├── phase2_ew_by_year.csv
    ├── phase3_mc_summary.csv
    └── full_validation_report.md
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from loguru import logger


# 6 品种横截面 EW 组合
SYMBOLS_6 = ["SHFE.AL", "SHFE.CU", "CZCE.FG", "SHFE.RU", "DCE.PP", "CZCE.CF"]
STRATEGIES_5 = ["trend", "term_structure", "mean_reversion", "vol_breakout", "composite_resonance"]

# 时间划分
IN_SAMPLE_START = "2020-01-01"
IN_SAMPLE_END = "2023-01-01"
OOS_START = "2023-01-01"
OOS_END = "2024-12-31"
FULL_START = "2020-01-01"
FULL_END = "2024-12-31"

OUTPUT_DIR = Path("output_backtest_pybroker/full_validation")


def phase1_optimize(pipe) -> Dict[str, Any]:
    """
    Phase 1: 全量调参 (in_sample: 2020-2022)。

    委托 Pipeline.optimize() 调用 runner/optimization/grid_search.py + oos_selector.py。
    返回 best_params 字典。
    """
    print("\n" + "=" * 80)
    print("  [Phase 1] 全量调参 (in_sample: 2020-01-01 ~ 2022-12-31)")
    print("=" * 80)

    # 临时把 full_start 切到 2020-01-01 避免跑老数据
    pipe = pipe.with_config(
        full_start=IN_SAMPLE_START,
        full_end=OOS_END,
    )
    # 在 in_sample 内做优化
    pipe = pipe.with_config(
        in_sample_end=IN_SAMPLE_END,
    )

    pipe = pipe.optimize(tasks=["grid", "oos"], save_to_config=True)

    opt = pipe._results.get("optimization", {})
    best_params = opt.get("best_params", {}) or {}
    print(f"  [Phase 1] 完成: {len(best_params)} 个策略的最优参数")
    for sname, params in best_params.items():
        print(f"    - {sname}: {params}")

    return {"best_params": best_params, "pipe": pipe}


def phase2_ew_backtest(pipe, best_params: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Phase 2: 6 品种横截面 EW 组合回测。

    对每个 (in_sample, OOS) 窗口，对 5 子策略分别跑全 6 品种，
    汇总为等权组合 Sharpe。
    """
    print("\n" + "=" * 80)
    print("  [Phase 2] 6 品种横截面 EW 组合回测 (train=2020-2022 / OOS=2023-2024)")
    print("=" * 80)

    from core.engine.backtest_runner import PyBrokerBacktestRunner

    ds = pipe._data
    config = pipe._config

    rows = []
    for window_name, start, end in [
        ("TRAIN_2020_2022", IN_SAMPLE_START, IN_SAMPLE_END),
        ("OOS_2023_2024", OOS_START, OOS_END),
    ]:
        for sname in STRATEGIES_5:
            try:
                runner = PyBrokerBacktestRunner(ds, config)
                runner.register_strategies([sname])
                if best_params.get(sname):
                    runner.set_custom_params({sname: best_params[sname]})
                res = runner.run(start_date=start, end_date=end)
                metrics = res.metrics if res and hasattr(res, "metrics") else {}
                # PyBroker metrics_df 使用 pct 后缀，使用百分号字段
                rows.append({
                    "window": window_name,
                    "strategy": sname,
                    "sharpe": round(float(metrics.get("sharpe", 0) or 0), 4),
                    "total_return_pct": round(float(metrics.get("total_return_pct", 0) or 0), 4),
                    "max_dd_pct": round(float(metrics.get("max_drawdown_pct", 0) or 0), 4),
                    "total_pnl": round(float(metrics.get("total_pnl", 0) or 0), 2),
                    "trade_count": int(metrics.get("trade_count", 0) or 0),
                    "win_rate": round(float(metrics.get("win_rate", 0) or 0), 4),
                })
            except Exception as e:
                logger.error(f"  {window_name} {sname} 失败: {e}")
                rows.append({
                    "window": window_name,
                    "strategy": sname,
                    "sharpe": 0.0, "total_return_pct": 0.0, "max_dd_pct": 0.0,
                    "total_pnl": 0.0, "trade_count": 0, "win_rate": 0.0, "error": str(e),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "phase2_ew_results.csv", index=False)

    # 计算 EW 等权组合
    print(f"\n  Phase 2 详细结果 (按 window × strategy):")
    print(f"    {'window':<15} {'strategy':<22} {'sharpe':>8} {'ret%':>8} {'dd%':>8} {'pnl':>12} {'trades':>8}")
    for _, r in df.iterrows():
        print(f"    {r['window']:<15} {r['strategy']:<22} {r['sharpe']:>8.4f} {r['total_return_pct']:>8.4f} {r['max_dd_pct']:>8.4f} {r['total_pnl']:>12.2f} {int(r['trade_count']):>8d}")

    # EW 聚合
    print(f"\n  [EW 聚合 (5 策略等权平均)]")
    summary_rows = []
    for window_name in df["window"].unique():
        sub = df[df["window"] == window_name]
        valid = sub[sub["trade_count"] > 0]
        ew_sharpe = round(valid["sharpe"].mean(), 4) if not valid.empty else 0.0
        ew_return_pct = round(valid["total_return_pct"].mean(), 4) if not valid.empty else 0.0
        ew_mdd_pct = round(valid["max_dd_pct"].mean(), 4) if not valid.empty else 0.0
        ew_pnl = round(float(valid["total_pnl"].sum()), 2)
        ew_trades = int(sub["trade_count"].sum())
        summary_rows.append({
            "window": window_name,
            "ew_sharpe": ew_sharpe,
            "ew_return_pct": ew_return_pct,
            "ew_mdd_pct": ew_mdd_pct,
            "ew_pnl": ew_pnl,
            "total_trades": ew_trades,
            "valid_strategies": len(valid),
        })
        print(f"    {window_name:<15} EW_Sharpe={ew_sharpe:>7.4f}  EW_Ret%={ew_return_pct:>7.4f}  EW_DD%={ew_mdd_pct:>7.4f}  PnL={ew_pnl:>12.2f}  trades={ew_trades}")

    # 衰减率（Sharpe 衰减 = (OOS - Train) / |Train|）
    train = next((r for r in summary_rows if r["window"] == "TRAIN_2020_2022"), None)
    oos = next((r for r in summary_rows if r["window"] == "OOS_2023_2024"), None)
    if train and oos:
        if abs(train["ew_sharpe"]) > 1e-6:
            decay = round((oos["ew_sharpe"] - train["ew_sharpe"]) / abs(train["ew_sharpe"]), 4)
        else:
            decay = 0.0
        print(f"    OOS Sharpe 衰减率: {decay:+.1%} (训练 {train['ew_sharpe']:.4f} → OOS {oos['ew_sharpe']:.4f})")
        summary_rows.append({
            "window": "OOS_DECAY",
            "ew_sharpe": decay,
            "ew_return_pct": 0, "ew_mdd_pct": 0, "ew_pnl": 0,
            "total_trades": 0, "valid_strategies": 0,
        })

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "phase2_ew_summary.csv", index=False)
    return df


def phase3_monte_carlo(pipe, best_params: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Phase 3: 蒙特卡洛 1000 次鲁棒性测试。

    委托 runner.validation.task3_monte_carlo（基于 core.validation.MonteCarloSimulator）。
    """
    print("\n" + "=" * 80)
    print("  [Phase 3] 蒙特卡洛 1000 次鲁棒性测试 (full: 2020-2024)")
    print("=" * 80)

    from runner.validation.monte_carlo import task3_monte_carlo
    from core.config.strategy_profiles import StrategyLibrary

    ds = pipe._data
    config = pipe._config
    lib = StrategyLibrary()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = task3_monte_carlo(
        data_source=ds,
        config=config,
        lib=lib,
        output_dir=OUTPUT_DIR / "phase3_mc",
        best_params=best_params,
        cross_sectional=False,
    )
    summary = result.get("summary")
    if summary is not None and isinstance(summary, pd.DataFrame):
        print("\n  蒙特卡洛汇总 (1000 次模拟):")
        print(summary.to_string(index=False))
        return summary
    return pd.DataFrame()


def write_report(phase1: Dict, phase2_df: pd.DataFrame, phase3_df: pd.DataFrame):
    """生成 markdown 报告。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bp = phase1.get("best_params", {})

    lines = [
        "# 全量验证报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**时间划分**: train={IN_SAMPLE_START} ~ {IN_SAMPLE_END}, OOS={OOS_START} ~ {OOS_END}",
        f"**品种**: {', '.join(SYMBOLS_6)}",
        "",
        "---",
        "",
        "## Phase 1: 调参最优参数",
        "",
    ]
    for sname, params in bp.items():
        lines.append(f"- **{sname}**: `{params}`")
    lines.append("")

    if not phase2_df.empty:
        lines += [
            "## Phase 2: 6 品种横截面 EW 组合回测",
            "",
            "| Window | Strategy | Sharpe | Ret% | DD% | PnL | Trades |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in phase2_df.iterrows():
            lines.append(
                f"| {r['window']} | {r['strategy']} | {r['sharpe']:.4f} | {r['total_return_pct']:.4f} | {r['max_dd_pct']:.4f} | {r['total_pnl']:.2f} | {int(r['trade_count'])} |"
            )
        lines.append("")

    if not phase3_df.empty:
        lines += [
            "## Phase 3: 蒙特卡洛 1000 次鲁棒性测试",
            "",
            "| Strategy | Final Mean | Final Median | 5% CI | 95% CI | Bankruptcy | Avg MaxDD | Calmar |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, r in phase3_df.iterrows():
            lines.append(
                f"| {r.get('strategy', '')} | {r.get('final_mean', 0):.4f} | {r.get('final_median', 0):.4f} | "
                f"{r.get('final_5pct', 0):.4f} | {r.get('final_95pct', 0):.4f} | {r.get('bankruptcy_prob', 0):.4f} | "
                f"{r.get('avg_max_dd', 0):.4f} | {r.get('calmar_mean', 0):.4f} |"
            )
        lines.append("")

    out = OUTPUT_DIR / "full_validation_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [报告] {out}")


def main() -> int:
    print("=" * 80)
    print("  全量验证: 调参 → 6 品种 EW → 蒙特卡洛")
    print(f"  开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 写运行日志（同时打到 stderr 和文件）
    log_path = OUTPUT_DIR / "run.log"
    logger.add(str(log_path), level="INFO", enqueue=True)

    from runner import Pipeline

    pipe = Pipeline("config.yaml").load_data()

    # Phase 1: 调参
    p1 = phase1_optimize(pipe)
    best_params = p1.get("best_params", {})

    # 保存 best_params
    with open(OUTPUT_DIR / "phase1_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    # Phase 2: 6 品种横截面 EW
    p2 = phase2_ew_backtest(pipe, best_params)

    # Phase 3: 蒙特卡洛
    p3 = phase3_monte_carlo(pipe, best_params)

    # 汇总报告
    write_report(p1, p2, p3)

    print("\n" + "=" * 80)
    print("  全量验证完成")
    print(f"  结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  日志: {log_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
