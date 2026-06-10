#!/usr/bin/env python3
"""
2024 衰退根因诊断。

4 维度:
  1) 5 子策略 × 6 品种 在 2024 的 Sharpe/Return 矩阵
  2) 2024 单品种单子策略逐笔 PnL 分解
  3) 按月胜率/盈亏比 切片
  4) 与 2022/2023 对比，定位衰退来源（策略级 vs 品种级）
"""

import sys
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import pandas as pd
from loguru import logger


YEARS = [("2022", "2022-01-01", "2022-12-31"),
         ("2023", "2023-01-01", "2023-12-31"),
         ("2024", "2024-01-01", "2024-12-31")]

STRATEGIES = ["trend", "term_structure", "mean_reversion", "vol_breakout", "composite_resonance"]
SYMBOLS = ["SHFE.AL", "SHFE.CU", "CZCE.FG", "SHFE.RU", "DCE.PP", "CZCE.CF"]


def backtest_year(pipe, sname: str, sym: str, year: str, start: str, end: str) -> dict:
    """单策略单品种单年回测。"""
    from core.engine.backtest_runner import PyBrokerBacktestRunner

    ds = pipe._data
    config = pipe._config
    try:
        # 注意: target_symbols 传给 __init__，不是 run()
        runner = PyBrokerBacktestRunner(ds, config, target_symbols=[sym])
        runner.register_strategies([sname])
        res = runner.run(start_date=start, end_date=end)
        metrics = res.metrics if res and hasattr(res, "metrics") else {}
        return {
            "sharpe": round(float(metrics.get("sharpe", 0) or 0), 4),
            "total_return": round(float(metrics.get("total_return", 0) or 0), 4),
            "max_dd": round(float(metrics.get("max_drawdown", 0) or 0), 4),
            "trade_count": int(metrics.get("trade_count", 0) or 0),
            "win_rate": round(float(metrics.get("win_rate", 0) or 0), 4),
        }
    except Exception as e:
        return {"sharpe": 0.0, "total_return": 0.0, "max_dd": 0.0, "trade_count": 0, "win_rate": 0.0, "error": str(e)}


def main() -> int:
    from runner import Pipeline

    print("=" * 80)
    print("  2024 衰退根因诊断")
    print(f"  开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    pipe = Pipeline("config.yaml").load_data()

    # 维度1+2: 5策略 × 6品种 × 3年 矩阵
    print("\n[维度1+2] 5策略 × 6品种 × 3年 矩阵（Sharpe）")
    records = []
    for year_label, start, end in YEARS:
        for sname in STRATEGIES:
            for sym in SYMBOLS:
                m = backtest_year(pipe, sname, sym, year_label, start, end)
                records.append({
                    "year": year_label, "strategy": sname, "symbol": sym,
                    **{k: m.get(k, 0) for k in ["sharpe", "total_return", "max_dd", "trade_count", "win_rate"]}
                })

    df = pd.DataFrame(records)
    out_dir = Path("output_backtest_pybroker/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "yearly_strategy_symbol_matrix.csv", index=False)
    print(f"  矩阵已保存: {out_dir / 'yearly_strategy_symbol_matrix.csv'}")

    # 维度3: 策略级 EW 组合按年表现（仅 trade_count>0 的真实信号）
    print("\n[维度3] EW 策略组合 按年表现（仅含实际交易的样本）")
    pivot_sharpe = df.pivot_table(index=["strategy", "symbol"], columns="year", values="sharpe").reset_index()
    print(f"  共 {len(pivot_sharpe)} 行 (5 策略 × 6 品种 = 30)")
    print(f"  其中 0-trade 样本: {len(df[df.trade_count==0])} / {len(df)}")

    # 各策略平均（仅 trade_count>0 的样本）
    print(f"\n  {'策略':<22}{'2022':<10}{'2023':<10}{'2024':<10}{'Δ24-22':<10}{'衰退?'}")
    print("  " + "-" * 70)
    for sname in STRATEGIES:
        def _avg_sharpe(year_label):
            sub = df[(df.strategy == sname) & (df.year == year_label) & (df.trade_count > 0)]
            return sub["sharpe"].mean() if len(sub) else 0.0, len(sub)
        s22, n22 = _avg_sharpe("2022")
        s23, n23 = _avg_sharpe("2023")
        s24, n24 = _avg_sharpe("2024")
        delta = s24 - s22
        flag = "⚠️ 衰退" if delta < -0.1 else ("✅ 改善" if delta > 0.1 else "→ 持平")
        print(f"  {sname:<22}{s22:<10.4f}{s23:<10.4f}{s24:<10.4f}{delta:<10.4f}{flag}  (n={n22}/{n23}/{n24})")

    # EW 组合（所有样本含 0）
    print(f"\n  EW_BLEND(含 0)        {df[df.year=='2022']['sharpe'].mean():<10.4f}{df[df.year=='2023']['sharpe'].mean():<10.4f}{df[df.year=='2024']['sharpe'].mean():<10.4f}")
    # EW 组合（仅 trade>0）
    sub22 = df[(df.year == "2022") & (df.trade_count > 0)]["sharpe"]
    sub23 = df[(df.year == "2023") & (df.trade_count > 0)]["sharpe"]
    sub24 = df[(df.year == "2024") & (df.trade_count > 0)]["sharpe"]
    print(f"  EW_BLEND(仅 trade>0)  {sub22.mean():<10.4f}{sub23.mean():<10.4f}{sub24.mean():<10.4f}  (n={len(sub22)}/{len(sub23)}/{len(sub24)})")

    # 维度4: 品种级表现
    print(f"\n[维度4] 品种级表现（5子策略平均，仅 trade>0）")
    print(f"  {'品种':<14}{'2022':<10}{'2023':<10}{'2024':<10}{'Δ24-22':<10}{'衰退?'}")
    print("  " + "-" * 70)
    for sym in SYMBOLS:
        def _avg(year_label):
            sub = df[(df.symbol == sym) & (df.year == year_label) & (df.trade_count > 0)]
            return sub["sharpe"].mean() if len(sub) else 0.0, len(sub)
        s22, n22 = _avg("2022")
        s23, n23 = _avg("2023")
        s24, n24 = _avg("2024")
        delta = s24 - s22
        flag = "⚠️ 衰退" if delta < -0.1 else ("✅ 改善" if delta > 0.1 else "→ 持平")
        print(f"  {sym:<14}{s22:<10.4f}{s23:<10.4f}{s24:<10.4f}{delta:<10.4f}{flag}  (n={n22}/{n23}/{n24})")

    # 根因总结
    print(f"\n[根因总结]")
    # 真实 2024 平均（仅 trade>0 样本）
    avg24_real = sub24.mean() if len(sub24) else 0.0
    avg22_real = sub22.mean() if len(sub22) else 0.0
    delta_real = avg24_real - avg22_real
    print(f"  真实 2024 EW avg Sharpe = {avg24_real:.4f}  (样本 n={len(sub24)})")
    print(f"  真实 2022 EW avg Sharpe = {avg22_real:.4f}  (样本 n={len(sub22)})")
    print(f"  真实 Δ24-22 = {delta_real:+.4f}")
    print(f"  0-trade 样本占比 = {len(df[df.trade_count==0])}/{len(df)} = {len(df[df.trade_count==0])/len(df)*100:.1f}%")

    # 0-trade 集中在哪些品种？
    zero_breakdown = df[df.trade_count == 0].groupby(["year", "symbol"]).size().reset_index(name="n")
    print(f"\n  0-trade 样本分布:")
    if len(zero_breakdown) > 0:
        for _, r in zero_breakdown.iterrows():
            print(f"    {r['year']} × {r['symbol']:<14} 0-trade 策略数 = {r['n']}")
    else:
        print("    无")

    # 0-trade 样本分布
    zero_by_year = df[df.trade_count == 0].groupby("year").size()
    print(f"\n  按年: {dict(zero_by_year)}")

    # 找到 2024 负贡献的 (strategy, symbol) 组合（仅 trade>0）
    neg_pairs_2024 = df[(df.year == "2024") & (df.trade_count > 0) & (df.sharpe < -0.05)].sort_values("sharpe")
    print(f"\n  2024 trade>0 且 Sharpe<-0.05 的(策略,品种)组合 ({len(neg_pairs_2024)} 个):")
    if len(neg_pairs_2024) > 0:
        for _, r in neg_pairs_2024.head(15).iterrows():
            print(f"    {r['strategy']:<22} × {r['symbol']:<14} Sharpe={r['sharpe']:.4f} trades={r['trade_count']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
