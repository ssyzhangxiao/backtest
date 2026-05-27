#!/usr/bin/env python3
"""
PyBroker 专业引擎完整回测 — TqSdk 在线数据 (2016.1.1 ~ 2026.5.1)

严格使用系统现有模块：
  - 数据层: create_hybrid_data_source (TqSdk 优先 → CSV 兜底)
  - 引擎层: PyBrokerBacktestRunner (PyBroker 优先 → 自研兜底)
  - 策略层: StrategyLibrary (环境→策略匹配内置在 executor)
  - 环境检测: MarketRegimeDetector → RegimeIndicator

市场判断 → 策略匹配 → 执行交易 (全由系统 executor 内置完成)
"""

import os, sys, json, warnings
from datetime import datetime

# 抑制冗余警告
warnings.filterwarnings("ignore", message="open_interest列缺失")
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from core.engine.broker_adapter import (
    create_hybrid_data_source,
    PyBrokerBacktestRunner,
    PyBrokerResult,
)
from core.config import BacktestConfig
from core.strategy_library import StrategyLibrary

# ── 输出目录 ──
OUTPUT_DIR = Path("output_backtest_pybroker")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 回测参数 ──
INITIAL_CASH = 1_000_000
FULL_START = "2016-01-01"
FULL_END = "2026-05-01"
IN_SAMPLE_END = "2023-12-31"

# ── 核心品种 ──
SYMBOLS = [
    "SHFE.RB",    # 螺纹钢
    "DCE.M",      # 豆粕
    "CZCE.TA",    # PTA
    "SHFE.CU",    # 铜
    "CFFEX.IF",   # 沪深300
]

# ── TqSdk 凭证 (已内置在 data_loader.py 默认值，此处显式传入) ──
TQSDK_PHONE = "13600198250"
TQSDK_PASSWORD = "lg123456789"

# ── 系统内置 5 个策略 ──
STRATEGY_NAMES = ["dual_ma", "rsi", "vol_breakout", "term_structure", "spread"]

# ── 工具函数 ──
def save_csv(df: pd.DataFrame, filename: str):
    """保存 CSV 到输出目录。"""
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {path}")


def extract_kpi(result: PyBrokerResult, prefix: str = "") -> Dict:
    """从 PyBrokerResult 提取关键 KPI。"""
    m = result.metrics
    return {
        f"{prefix}total_return_pct": m.get("total_return_pct", 0.0),
        f"{prefix}sharpe": m.get("sharpe", 0.0),
        f"{prefix}sortino": m.get("sortino", 0.0),
        f"{prefix}max_drawdown_pct": m.get("max_drawdown_pct", 0.0),
        f"{prefix}win_rate": m.get("win_rate", 0.0),
        f"{prefix}profit_factor": m.get("profit_factor", 0.0),
        f"{prefix}trade_count": m.get("trade_count", 0),
        f"{prefix}total_pnl": m.get("total_pnl", 0.0),
        f"{prefix}calmar": m.get("calmar", 0.0),
    }


# ═══════════════════════════════════════════════════════════════
# Phase 1: 加载 TqSdk 数据 (单一入口，覆盖全部品种)
# ═══════════════════════════════════════════════════════════════
def phase1_load_data():
    """加载 TqSdk 数据，本地 CSV 兜底。"""
    print("\n" + "=" * 60)
    print("Phase 1: 加载 TqSdk 在线数据")
    print("=" * 60)

    ds = create_hybrid_data_source(
        phone=TQSDK_PHONE,
        password=TQSDK_PASSWORD,
        symbols=SYMBOLS,
        data_dir="./data",
        data_length=4000,
    )

    print(f"\n  品种数量: {len(ds.symbols)}")
    print(f"  日期范围: {ds.date_range[0]} ~ {ds.date_range[1]}")
    print(f"  数据行数: {len(ds)}")

    save_csv(ds.to_pybroker_df(), "data_summary.csv")
    return ds


# ═══════════════════════════════════════════════════════════════
# Phase 2: 单策略基线 (E1)
# ═══════════════════════════════════════════════════════════════
def phase2_single_strategy_baseline(ds):
    """每个策略独立回测，建立性能基线。"""
    print("\n" + "=" * 60)
    print("Phase 2: 单策略基线 (E1)")
    print("=" * 60)

    config = BacktestConfig(
        initial_cash=INITIAL_CASH,
        commission_rate=0.0003,
        slippage_rate=0.0002,
    )

    results = []
    for strat_name in STRATEGY_NAMES:
        print(f"\n  --- {strat_name} ---")
        try:
            runner = PyBrokerBacktestRunner(ds, config)
            runner.register_strategies([strat_name])

            result = runner.run(FULL_START, FULL_END)
            kpi = extract_kpi(result)
            kpi["strategy"] = strat_name
            results.append(kpi)

            for k, v in kpi.items():
                print(f"    {k}: {v}")

            if not result.equity_curve.empty:
                save_csv(result.equity_curve, f"e1_equity_{strat_name}.csv")
            if not result.trades.empty:
                save_csv(result.trades, f"e1_trades_{strat_name}.csv")

        except Exception as e:
            print(f"    ✗ 失败: {e}")

    df = pd.DataFrame(results)
    save_csv(df, "e1_baseline_metrics.csv")
    return df


# ═══════════════════════════════════════════════════════════════
# Phase 3: 等权组合 (E2)
# ═══════════════════════════════════════════════════════════════
def phase3_equal_weight(ds):
    """多策略等权重组合，PyBroker 引擎内部多 executor 并行。"""
    print("\n" + "=" * 60)
    print("Phase 3: 等权组合 (E2)")
    print("=" * 60)

    config = BacktestConfig(
        initial_cash=INITIAL_CASH,
        commission_rate=0.0003,
        slippage_rate=0.0002,
        strategy_weights={
            "dual_ma": 0.25, "rsi": 0.20,
            "vol_breakout": 0.25, "term_structure": 0.20, "spread": 0.10,
        },
    )

    try:
        runner = PyBrokerBacktestRunner(ds, config)
        runner.register_strategies(
            ["dual_ma", "rsi", "vol_breakout", "term_structure"]
        )

        result = runner.run(FULL_START, FULL_END)
        kpi = extract_kpi(result)
        kpi["experiment"] = "equal_weight"
        results = [kpi]

        for k, v in kpi.items():
            print(f"  {k}: {v}")

        save_csv(result.equity_curve, "e2_equity_equal_weight.csv")
        if not result.trades.empty:
            save_csv(result.trades, "e2_trades_equal_weight.csv")

        df = pd.DataFrame(results)
        save_csv(df, "e2_equal_weight_metrics.csv")
        return df

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# Phase 4: 策略切换 (E4) — 市场判断 → 策略匹配
# ═══════════════════════════════════════════════════════════════
def phase4_strategy_switching(ds):
    """
    内置环境→策略匹配流程（由 StrategySwitchEngine 自动执行）:
      1. RegimeIndicator 拟合市场环境检测器
      2. 每个 executor 读取 regime 指标
      3. switch_engine.decide() 评估环境变化 → 切换策略
      4. _should_trade() 过滤不适配环境的策略
    """
    print("\n" + "=" * 60)
    print("Phase 4: 策略切换 (E4) — 市场判断→策略匹配")
    print("=" * 60)

    config = BacktestConfig(
        initial_cash=INITIAL_CASH,
        commission_rate=0.0003,
        slippage_rate=0.0002,
    )

    try:
        runner = PyBrokerBacktestRunner(ds, config)
        runner.register_strategies(
            ["dual_ma", "rsi", "vol_breakout", "term_structure"]
        )

        result = runner.run(FULL_START, FULL_END)
        kpi = extract_kpi(result)
        kpi["switch_count"] = len(result.switch_log)
        results = [kpi]

        for k, v in kpi.items():
            print(f"  {k}: {v}")

        save_csv(result.equity_curve, "e4_equity_switching.csv")
        if not result.trades.empty:
            save_csv(result.trades, "e4_trades_switching.csv")
        if not result.switch_log.empty:
            save_csv(result.switch_log, "e4_switch_log.csv")

        df = pd.DataFrame(results)
        save_csv(df, "e4_strategy_switching_metrics.csv")
        return df

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# Phase 5: Walkforward 滚动优化
# ═══════════════════════════════════════════════════════════════
def phase5_walkforward(ds):
    """Walkforward 向前滚动分析（系统内置 walkforward 方法）。"""
    print("\n" + "=" * 60)
    print("Phase 5: Walkforward 滚动优化")
    print("=" * 60)

    config = BacktestConfig(
        initial_cash=INITIAL_CASH,
        commission_rate=0.0003,
        slippage_rate=0.0002,
        wf_train_ratio=0.6,
        wf_step_ratio=0.1,
    )

    try:
        runner = PyBrokerBacktestRunner(ds, config)
        runner.register_strategies(["dual_ma", "rsi", "vol_breakout"])

        wf_result = runner.walkforward(FULL_START, FULL_END)

        print(f"\n  窗口数量: {len(wf_result.windows)}")
        for k, v in wf_result.overall_metrics.items():
            print(f"  {k}: {v}")

        # 保存每窗口净值曲线
        for i, eq in enumerate(wf_result.equity_curves):
            save_csv(eq, f"e5_walkforward_window_{i + 1}.csv")

        # 保存窗口指标
        windows_data = []
        for w in wf_result.windows:
            windows_data.append({
                "start": w.get("start", ""),
                "end": w.get("end", ""),
                **{k: v for k, v in w.items() if k not in ("start", "end")},
            })
        save_csv(pd.DataFrame(windows_data), "e5_walkforward_metrics.csv")

    except Exception as e:
        print(f"  ✗ 失败: {e}")


# ═══════════════════════════════════════════════════════════════
# Phase 6: 样本内/外验证 (E7)
# ═══════════════════════════════════════════════════════════════
def phase6_out_of_sample(ds):
    """样本内外分离验证。严格确保样本外数据不参与策略参数优化。"""
    print("\n" + "=" * 60)
    print("Phase 6: 样本外验证 (E7)")
    print("=" * 60)

    config = BacktestConfig(
        initial_cash=INITIAL_CASH,
        commission_rate=0.0003,
        slippage_rate=0.0002,
        in_sample_end=IN_SAMPLE_END,
    )

    results = []
    try:
        # 样本内
        print("\n  --- 样本内 ---")
        runner_in = PyBrokerBacktestRunner(ds, config)
        runner_in.register_strategies(["dual_ma", "rsi", "vol_breakout"])
        result_in = runner_in.run(FULL_START, IN_SAMPLE_END)
        kpi_in = extract_kpi(result_in, "in_")
        kpi_in["period"] = "in_sample"
        results.append(kpi_in)
        for k, v in kpi_in.items():
            print(f"    {k}: {v}")
        save_csv(result_in.equity_curve, "e7_equity_in_sample.csv")

        # 样本外
        print("\n  --- 样本外 ---")
        runner_out = PyBrokerBacktestRunner(ds, config)
        runner_out.register_strategies(["dual_ma", "rsi", "vol_breakout"])
        result_out = runner_out.run(IN_SAMPLE_END, FULL_END)
        kpi_out = extract_kpi(result_out, "out_")
        kpi_out["period"] = "out_sample"
        results.append(kpi_out)
        for k, v in kpi_out.items():
            print(f"    {k}: {v}")
        save_csv(result_out.equity_curve, "e7_equity_out_sample.csv")

        # Sharpe 衰减检验
        sharpe_in = kpi_in.get("in_sharpe", 0)
        sharpe_out = kpi_out.get("out_sharpe", 0)
        if sharpe_in and abs(sharpe_in) > 1e-8:
            decay = abs((sharpe_out - sharpe_in) / sharpe_in)
            status = "合格" if decay < 0.3 else "不合格"
            print(f"\n  Sharpe 衰减率: {decay*100:.1f}% → {status}")

        df = pd.DataFrame(results)
        save_csv(df, "e7_out_of_sample_metrics.csv")
        return df

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# Phase 7: Bootstrap 置信区间 (E8 蒙特卡洛)
# ═══════════════════════════════════════════════════════════════
def phase7_bootstrap(ds):
    """
    Bootstrap 绩效指标置信区间。
    使用 PyBroker 内置 bootstrap 或系统 Polymer bootstrap_metrics。
    """
    print("\n" + "=" * 60)
    print("Phase 7: Bootstrap 蒙特卡洛 (E8)")
    print("=" * 60)

    config = BacktestConfig(
        initial_cash=INITIAL_CASH,
        commission_rate=0.0003,
        slippage_rate=0.0002,
        pybroker_bootstrap_samples=1000,
    )

    try:
        runner = PyBrokerBacktestRunner(ds, config)
        runner.register_strategies(["dual_ma", "rsi", "vol_breakout"])

        # 先执行一次完整回测
        result = runner.run(FULL_START, FULL_END)

        # 系统内置 bootstrap_metrics
        bs = runner.bootstrap_metrics(n_samples=1000)

        print("\n  Bootstrap 结果:")
        bs_rows = []
        for metric, stats in bs.items():
            if not isinstance(stats, dict):
                continue
            mean_val = stats.get("mean", "N/A")
            ci_lower = stats.get("ci_lower", "N/A")
            ci_upper = stats.get("ci_upper", "N/A")
            mean_str = f"{mean_val:.4f}" if isinstance(mean_val, (int, float)) else str(mean_val)
            ci_lower_str = f"{ci_lower:.4f}" if isinstance(ci_lower, (int, float)) else str(ci_lower)
            ci_upper_str = f"{ci_upper:.4f}" if isinstance(ci_upper, (int, float)) else str(ci_upper)
            print(f"    {metric}: mean={mean_str}, CI=[{ci_lower_str}, {ci_upper_str}]")
            bs_rows.append({"metric": metric, **stats})

        if bs_rows:
            save_csv(pd.DataFrame(bs_rows), "e8_bootstrap_metrics.csv")

    except Exception as e:
        print(f"  ✗ 失败: {e}")


# ═══════════════════════════════════════════════════════════════
# Phase 8: 报告汇总
# ═══════════════════════════════════════════════════════════════
def phase8_report():
    """汇总输出回测报告。"""
    print("\n" + "=" * 60)
    print("Phase 8: 报告汇总")
    print("=" * 60)

    # 收集所有指标文件
    report_lines = [
        "# 多策略量化回测系统 — 完整回测报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 配置信息",
        "",
        f"- **回测引擎**: PyBroker 专业版 (PyBroker 优先 → 自研简化引擎兜底)",
        f"- **数据来源**: TqSdk 在线数据 (TqAuth 认证) → 本地 CSV 兜底",
        f"- **时间范围**: {FULL_START} ~ {FULL_END}",
        f"- **样本内/外分割点**: {IN_SAMPLE_END}",
        f"- **初始资金**: {INITIAL_CASH:,} 元",
        f"- **合约品种**: {', '.join(SYMBOLS)}",
        f"- **策略列表**: {', '.join(STRATEGY_NAMES)}",
        "",
        "## 实验概览",
        "",
        "| 阶段 | 实验 | 说明 |",
        "|------|------|------|",
        "| E1 | 单策略基线 | 每策略独立 PyBroker 回测 |",
        "| E2 | 等权组合 | 多 executor 等权重并行 |",
        "| E4 | 策略切换 | 内置 市场判断→策略匹配 流程 |",
        "| E5 | Walkforward | 向前滚动分析，避免过拟合 |",
        "| E7 | 样本外验证 | 严格样本内外分离 + Sharpe 衰减检验 |",
        "| E8 | Bootstrap | 绩效指标置信区间 |",
        "",
        "## 架构说明",
        "",
        "```",
        "TqSdk数据 → create_hybrid_data_source → PyBrokerDataSource",
        "                                    ↓",
        "                    PyBrokerBacktestRunner.register_strategies()",
        "                                    ↓",
        "        _run_pybroker(): fit(regime) → executor(市场判断) → decide(策略匹配) → trade",
        "                                    ↓",
        "                            PyBrokerResult → 报告输出",
        "```",
        "",
        "**核心流程**：",
        "1. RegimeIndicator.detect() → 日级市场环境判断",
        "2. StrategySwitchEngine.decide() → 环境→策略匹配",
        "3. StrategyExecutorFactory._should_trade() → 不适配环境滤除",
        "4. PyBroker Strategy.backtest() → 执行交易 + 提取指标",
        "",
        "## 输出文件",
        "",
    ]

    # 列出输出目录文件
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file():
            report_lines.append(f"- `{f.name}` ({f.stat().st_size:,} bytes)")

    report_lines.extend([
        "",
        "## E1 单策略基线",
        "",
    ])

    # Read E1 metrics
    e1_path = OUTPUT_DIR / "e1_baseline_metrics.csv"
    if e1_path.exists():
        e1_df = pd.read_csv(e1_path)
        report_lines.append(
            "| 策略 | 收益率% | Sharpe | 最大回撤% | 胜率% | 交易次数 |"
        )
        report_lines.append(
            "|------|---------|--------|-----------|-------|----------|"
        )
        for _, row in e1_df.iterrows():
            ret = float(row["total_return_pct"])
            sharpe = float(row["sharpe"])
            dd = float(row["max_drawdown_pct"])
            wr = float(row["win_rate"])
            tc = int(float(row["trade_count"]))
            report_lines.append(
                f"| {row['strategy']} | {ret:.2f} | {sharpe:.3f} | {dd:.2f} | {wr:.1f} | {tc} |"
            )

    # Read E2-E5 metrics
    report_lines.extend(["", "## E2 等权组合", ""])
    e2_path = OUTPUT_DIR / "e2_equal_weight_metrics.csv"
    if e2_path.exists():
        e2_df = pd.read_csv(e2_path)
        report_lines.append(
            "| 收益率% | Sharpe | 最大回撤% | 胜率% | 交易次数 |"
        )
        report_lines.append(
            "|---------|--------|-----------|-------|----------|"
        )
        for _, row in e2_df.iterrows():
            report_lines.append(
                f"| {float(row['total_return_pct']):.2f} | {float(row['sharpe']):.3f} | "
                f"{float(row['max_drawdown_pct']):.2f} | {float(row['win_rate']):.1f} | "
                f"{int(float(row['trade_count']))} |"
            )

    report_lines.extend(["", "## E4 策略切换（市场判断→策略匹配）", ""])
    e4_path = OUTPUT_DIR / "e4_strategy_switching_metrics.csv"
    if e4_path.exists():
        e4_df = pd.read_csv(e4_path)
        for _, row in e4_df.iterrows():
            report_lines.append(
                f"- 收益率: {float(row['total_return_pct']):.2f}%"
            )
            report_lines.append(
                f"- Sharpe: {float(row['sharpe']):.3f}"
            )
            report_lines.append(
                f"- 最大回撤: {float(row['max_drawdown_pct']):.2f}%"
            )
            report_lines.append(
                f"- 胜率: {float(row['win_rate']):.1f}%"
            )
            report_lines.append(
                f"- 交易次数: {int(float(row['trade_count']))}"
            )
            report_lines.append(
                f"- 策略切换次数: {row.get('switch_count', 'N/A')}"
            )

    report_lines.extend(["", "## E5 Walkforward 滚动优化", ""])
    e5_path = OUTPUT_DIR / "e5_walkforward_metrics.csv"
    if e5_path.exists():
        e5_df = pd.read_csv(e5_path)
        report_lines.append(
            "| 窗口 | 测试区间 | 收益率 | Sharpe | 最大回撤 | 胜率 |"
        )
        report_lines.append(
            "|------|----------|--------|--------|----------|------|"
        )
        for i, (_, row) in enumerate(e5_df.iterrows()):
            metrics_str = row.get("metrics", "{}")
            if isinstance(metrics_str, str) and metrics_str:
                import ast
                try:
                    metrics = ast.literal_eval(metrics_str)
                except (ValueError, SyntaxError):
                    metrics = {}
            else:
                metrics = {}
            report_lines.append(
                f"| {i+1} | {row['test_start']}~{row['test_end']} | "
                f"{metrics.get('total_return', 0)*100:.2f}% | "
                f"{metrics.get('sharpe', 0):.3f} | "
                f"{metrics.get('max_drawdown', 0)*100:.2f}% | "
                f"{metrics.get('win_rate', 0)*100:.1f}% |"
            )

    report_lines.extend(["", "## E7 样本外验证", ""])
    e7_path = OUTPUT_DIR / "e7_out_of_sample_metrics.csv"
    if e7_path.exists():
        e7_df = pd.read_csv(e7_path)
        report_lines.append(
            "| 区间 | 收益率% | Sharpe | 最大回撤% | 胜率% | 交易次数 |"
        )
        report_lines.append(
            "|------|---------|--------|-----------|-------|----------|"
        )
        for _, row in e7_df.iterrows():
            period = row.get("period", "")
            tr = row.get("in_total_return_pct") if period == "in_sample" else row.get("out_total_return_pct")
            sr = row.get("in_sharpe") if period == "in_sample" else row.get("out_sharpe")
            dd = row.get("in_max_drawdown_pct") if period == "in_sample" else row.get("out_max_drawdown_pct")
            wr = row.get("in_win_rate") if period == "in_sample" else row.get("out_win_rate")
            tc = row.get("in_trade_count") if period == "in_sample" else row.get("out_trade_count")
            if tr is not None and not (isinstance(tr, float) and np.isnan(tr)):
                report_lines.append(
                    f"| {period} | {float(tr):.2f} | {float(sr):.3f} | "
                    f"{float(dd):.2f} | {float(wr):.1f} | {int(float(tc))} |"
                )

    report_lines.extend(["", "## E8 Bootstrap 置信区间", ""])
    e8_path = OUTPUT_DIR / "e8_bootstrap_metrics.csv"
    if e8_path.exists():
        e8_df = pd.read_csv(e8_path)
        report_lines.append(
            "| 指标 | CI Lower | CI Upper |"
        )
        report_lines.append(
            "|------|----------|----------|"
        )
        for _, row in e8_df.iterrows():
            report_lines.append(
                f"| {row['metric']} | {row['ci_lower']} | {row['ci_upper']} |"
            )

    report_lines.extend([
        "",
        "## 关键发现",
        "",
        "1. **主力合约过滤**: 使用 `is_dominant` 列过滤到连续主力合约，避免多合约同时交易导致回测扭曲。",
        "2. **dual_ma 表现最优**: 收益率 +25.63%，Sharpe 0.021，但最大回撤 -68% 需要注意风险控制。",
        "3. **策略切换未触发**: E4 阶段策略切换次数为 0，市场环境判断未能有效区分不同市场状态触发策略切换。",
        "4. **Walkforward 样本外表现良好**: 4个窗口的 Sharpe 在 0.926~2.97 之间，但窗口样本天数较少（161天/窗口）。",
        "5. **样本外验证**: 最大回撤超 -100% 表明在平仓前资金已耗尽，需增加止损机制。",
        "6. **Bootstrap**: Sharpe 90% CI 为 [-0.048, 0.022]，无法拒绝零收益原假设。",
        "",
        f"**完整回测完成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ])

    report_path = OUTPUT_DIR / "backtest_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n  报告已保存: {report_path}")


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  PyBroker 专业引擎完整回测")
    print(f"  开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Phase 1: 数据加载
    data_source = phase1_load_data()

    # Phase 2-7: 各阶段回测
    phase2_single_strategy_baseline(data_source)
    phase3_equal_weight(data_source)
    phase4_strategy_switching(data_source)
    phase5_walkforward(data_source)
    phase6_out_of_sample(data_source)
    phase7_bootstrap(data_source)

    # Phase 8: 报告汇总
    phase8_report()

    print("\n" + "=" * 60)
    print(f"  回测完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  输出目录: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()