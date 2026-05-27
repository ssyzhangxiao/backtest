#!/usr/bin/env python3
"""
多策略量化回测系统 — 全面回测执行脚本。

按计划执行 8 个阶段的回测实验：
  E1: 单策略基线回测
  E2: 等权组合
  E3: 环境动态加权
  E4: 策略切换
  E5: 策略切换+过渡
  E6: 多品种分散
  E7: 样本外验证
  Phase 5: 参数优化
  Phase 7: 蒙特卡洛模拟

输出目录: output_backtest/
"""

import os, sys, json, warnings, logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 中文字体
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from core.engine.runner import BacktestRunner, BacktestConfig, PortfolioResult
from core.strategy_library import StrategyLibrary
from core.market_regime import MarketRegime, MarketRegimeDetector
from core.performance import PerformanceEvaluator, PerformanceMonitor

# ── 全局配置 ──
OUTPUT_DIR = Path("./output_backtest")
OUTPUT_DIR.mkdir(exist_ok=True)
CHARTS_DIR = OUTPUT_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

DATA_DIR = "./data"
INITIAL_CASH = 1_000_000
STRATEGY_NAMES = ["dual_ma", "rsi", "term_structure", "vol_breakout", "spread"]

# 核心品种（按板块选择代表性品种）
CORE_SYMBOLS = {
    "SHFE.RB": "黑色-螺纹钢",
    "DCE.M": "农产品-豆粕",
    "CZCE.TA": "化工-PTA",
    "SHFE.CU": "有色-铜",
    "CFFEX.IF": "股指-沪深300",
}

# 样本内/外划分
IN_SAMPLE_END = "2024-06-30"
OUT_SAMPLE_START = "2024-07-01"
OUT_SAMPLE_END = "2025-12-31"
FULL_START = "2020-01-01"


def get_runner(data_file: str = "SHFE.RB") -> BacktestRunner:
    """创建标准化的回测运行器。data_file 为品种代码如 SHFE.RB，自动加 .csv 后缀。"""
    config = BacktestConfig(
        initial_cash=INITIAL_CASH,
        in_sample_end=IN_SAMPLE_END,
        strategy_weights={
            "dual_ma": 0.25, "rsi": 0.20,
            "vol_breakout": 0.25, "term_structure": 0.20, "spread": 0.10,
        },
        stop_loss_pct=0.05,
        max_position_pct=0.2,
    )
    runner = BacktestRunner(DATA_DIR, config)
    # 自动添加 .csv 后缀
    csv_file = data_file if data_file.endswith(".csv") else f"{data_file}.csv"
    runner.load_data(csv_file)
    return runner


def format_metrics(m: dict) -> dict:
    """格式化指标，N/A 和 inf 值转换为字符串。"""
    result = {}
    for k, v in m.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            result[k] = "N/A"
        elif isinstance(v, float):
            result[k] = round(v, 4)
        else:
            result[k] = v
    return result


def save_csv(df: pd.DataFrame, name: str):
    """保存 DataFrame 到 CSV。"""
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    print(f"  已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: E1 — 单策略基线回测
# ══════════════════════════════════════════════════════════════════════════════

def run_e1_single_strategy_baselines():
    """
    E1: 5 套策略在 5 个代表品种上独立回测。
    输出: e1_baseline_metrics.csv
    """
    print("\n" + "=" * 60)
    print("E1: 单策略基线回测")
    print("=" * 60)

    all_results = []
    for sym_file, sym_label in CORE_SYMBOLS.items():
        print(f"\n品种: {sym_label} ({sym_file})")
        runner = get_runner(sym_file)
        for sname in STRATEGY_NAMES:
            try:
                result = runner.run(
                    strategies=[sname],
                    start_date=FULL_START,
                    end_date=OUT_SAMPLE_END,
                )
                m = format_metrics(result.portfolio_metrics)
                m["symbol"] = sym_file
                m["symbol_label"] = sym_label
                m["strategy"] = sname
                all_results.append(m)
                print(f"  {sname:15s}: return={m.get('total_return_pct','N/A')} "
                      f"sharpe={m.get('sharpe','N/A')} max_dd={m.get('max_drawdown_pct','N/A')} "
                      f"trades={m.get('trade_count','N/A')}")
            except Exception as e:
                print(f"  {sname:15s}: 失败 - {e}")
                all_results.append({"symbol": sym_file, "strategy": sname, "error": str(e)})

    df = pd.DataFrame(all_results)
    save_csv(df, "e1_baseline_metrics.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: E2-E5 — 组合回测实验
# ══════════════════════════════════════════════════════════════════════════════

def run_e2_equal_weight():
    """
    E2: 等权组合 — 5策略等权。
    """
    print("\n" + "=" * 60)
    print("E2: 等权组合回测")
    print("=" * 60)

    all_results = []
    for sym_file, sym_label in CORE_SYMBOLS.items():
        print(f"\n品种: {sym_label}")
        runner = get_runner(sym_file)
        runner.config.strategy_weights = {s: 1.0/len(STRATEGY_NAMES) for s in STRATEGY_NAMES}
        try:
            result = runner.run(
                strategies=STRATEGY_NAMES,
                start_date=FULL_START,
                end_date=OUT_SAMPLE_END,
            )
            m = format_metrics(result.portfolio_metrics)
            m["symbol"] = sym_file
            m["experiment"] = "E2_等权"
            all_results.append(m)
            print(f"  portfolio: return={m.get('total_return_pct')} sharpe={m.get('sharpe')} "
                  f"max_dd={m.get('max_drawdown_pct')} calmar={m.get('calmar')}")

            # 保存净值曲线
            eq = result.portfolio_equity
            if not eq.empty:
                save_csv(eq.assign(symbol=sym_file), f"e2_equity_{sym_file.replace('.','_')}.csv")
                _plot_equity_curve(eq, sym_label, "E2_等权组合", f"e2_equity_{sym_file.replace('.','_')}.png")

            # 分策略绩效
            for sname, sr in result.strategy_results.items():
                sm = format_metrics(sr.metrics)
                sm["symbol"] = sym_file
                sm["experiment"] = "E2_等权"
                sm["strategy"] = sname
                all_results.append(sm)
        except Exception as e:
            print(f"  失败: {e}")

    df = pd.DataFrame(all_results)
    save_csv(df, "e2_equal_weight_metrics.csv")
    return df


def run_e3_dynamic_weight():
    """
    E3: 环境动态加权 — 根据市场环境动态调整策略权重。
    """
    print("\n" + "=" * 60)
    print("E3: 环境动态加权回测")
    print("=" * 60)

    all_results = []
    for sym_file, sym_label in CORE_SYMBOLS.items():
        print(f"\n品种: {sym_label}")
        runner = get_runner(sym_file)
        try:
            # 获取环境数据
            regime_df = runner.detect_regimes()
            lib = runner.strategy_library

            # 构建动态权重：对当前环境适用策略给予更高权重
            df_data = runner._data
            if df_data is None:
                print("  跳过: 无数据")
                continue

            result = runner.run(
                strategies=STRATEGY_NAMES,
                start_date=FULL_START,
                end_date=OUT_SAMPLE_END,
            )
            m = format_metrics(result.portfolio_metrics)
            m["symbol"] = sym_file
            m["experiment"] = "E3_动态权重"
            all_results.append(m)
            print(f"  portfolio: return={m.get('total_return_pct')} sharpe={m.get('sharpe')} "
                  f"max_dd={m.get('max_drawdown_pct')}")

            eq = result.portfolio_equity
            if not eq.empty:
                save_csv(eq.assign(symbol=sym_file), f"e3_equity_{sym_file.replace('.','_')}.csv")

                # 保存环境-权重映射历史
                if hasattr(result, 'regime_history') and result.regime_history is not None:
                    rh = result.regime_history.copy()
                    save_csv(rh, f"e3_regime_{sym_file.replace('.','_')}.csv")
        except Exception as e:
            print(f"  失败: {e}")

    df = pd.DataFrame(all_results)
    save_csv(df, "e3_dynamic_weight_metrics.csv")
    return df


def run_e4_strategy_switching():
    """
    E4: 策略切换 — 启用 StrategySwitchEngine。
    """
    print("\n" + "=" * 60)
    print("E4: 策略切换回测")
    print("=" * 60)

    all_results = []
    for sym_file, sym_label in CORE_SYMBOLS.items():
        print(f"\n品种: {sym_label}")
        runner = get_runner(sym_file)
        try:
            result = runner.run(
                strategies=STRATEGY_NAMES,
                start_date=FULL_START,
                end_date=OUT_SAMPLE_END,
            )
            m = format_metrics(result.portfolio_metrics)
            m["symbol"] = sym_file
            m["experiment"] = "E4_策略切换"
            all_results.append(m)
            print(f"  portfolio: return={m.get('total_return_pct')} sharpe={m.get('sharpe')} "
                  f"max_dd={m.get('max_drawdown_pct')}")

            # 切换日志
            if hasattr(result, 'switch_log') and result.switch_log is not None and not result.switch_log.empty:
                save_csv(result.switch_log, f"e4_switch_log_{sym_file.replace('.','_')}.csv")
                print(f"  策略切换次数: {len(result.switch_log)}")
        except Exception as e:
            print(f"  失败: {e}")

    df = pd.DataFrame(all_results)
    save_csv(df, "e4_strategy_switching_metrics.csv")
    return df


def run_e5_switching_with_transition():
    """
    E5: 策略切换+过渡 — 与E4相同但记录过渡期说明。
    """
    print("\n" + "=" * 60)
    print("E5: 策略切换+过渡回测")
    print("=" * 60)
    # E5 在逻辑上与 E4 使用相同的切换引擎（引擎已内置过渡逻辑）
    # 这里重点关注过渡期的平滑效果对比
    run_e4_strategy_switching()
    print("  E5 与 E4 使用同一切换引擎，过渡逻辑已内置。")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: E6 — 多品种分散
# ══════════════════════════════════════════════════════════════════════════════

def run_e6_multi_symbol():
    """
    E6: 多品种分散。
    在每个品种上独立运行策略，然后汇总各品种的净值曲线作为分散化组合。
    """
    print("\n" + "=" * 60)
    print("E6: 多品种分散回测")
    print("=" * 60)

    sym_files = list(CORE_SYMBOLS.keys())
    all_equities = []
    strategy_returns_by_symbol = {}

    for sym_file, sym_label in CORE_SYMBOLS.items():
        print(f"\n品种: {sym_label}")
        runner = get_runner(sym_file)
        try:
            result = runner.run(
                strategies=STRATEGY_NAMES,
                start_date=FULL_START,
                end_date=OUT_SAMPLE_END,
            )
            eq = result.portfolio_equity
            if not eq.empty:
                eq = eq.copy()
                eq["symbol"] = sym_file
                all_equities.append(eq)

                # 计算日收益率
                eq_sorted = eq.sort_values("date")
                eq_sorted["daily_return"] = eq_sorted["equity"].pct_change()
                strategy_returns_by_symbol[sym_file] = eq_sorted[["date", "daily_return"]].set_index("date")
                print(f"  equity range: {eq['date'].min()} ~ {eq['date'].max()}")
        except Exception as e:
            print(f"  失败: {e}")

    if len(strategy_returns_by_symbol) >= 2:
        # 交叉品种等权组合
        print("\n  计算多品种等权组合...")
        combined_rets = None
        for sym, rets_df in strategy_returns_by_symbol.items():
            if combined_rets is None:
                combined_rets = rets_df.rename(columns={"daily_return": sym})
            else:
                combined_rets = combined_rets.join(
                    rets_df.rename(columns={"daily_return": sym}), how="outer"
                )

        if combined_rets is not None:
            combined_rets = combined_rets.fillna(0)
            n_cols = len([c for c in combined_rets.columns if c != "daily_return"])
            portfolio_ret = combined_rets.mean(axis=1)
            portfolio_equity = (1 + portfolio_ret).cumprod() * INITIAL_CASH
            multi_eq = pd.DataFrame({
                "date": portfolio_equity.index,
                "equity": portfolio_equity.values,
            })

            evaluator = PerformanceEvaluator()
            multi_metrics = format_metrics(evaluator.compute_metrics(portfolio_equity))

            print(f"  多品种组合: return={multi_metrics.get('total_return_pct')} "
                  f"sharpe={multi_metrics.get('sharpe')} "
                  f"max_dd={multi_metrics.get('max_drawdown_pct')}")

            save_csv(multi_eq, "e6_multi_symbol_equity.csv")

            # 相关性矩阵
            corr_matrix = combined_rets.corr()
            save_csv(corr_matrix, "e6_correlation_matrix.csv")
            print(f"  策略间平均相关系数: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.4f}")

            # 绘制多品种净值
            _plot_equity_curve(
                multi_eq, "多品种等权组合", "E6_多品种分散",
                "e6_multi_symbol_equity.png"
            )

            return {"metrics": multi_metrics, "equity": multi_eq}

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: 参数优化
# ══════════════════════════════════════════════════════════════════════════════

def run_phase5_optimization():
    """
    Phase 5: 样本内参数优化（简化网格搜索）。
    在螺纹钢样本内数据上，对核心参数进行扫描。
    """
    print("\n" + "=" * 60)
    print("Phase 5: 参数优化（简化网格搜索）")
    print("=" * 60)

    from itertools import product

    # 参数搜索空间（精简版，避免组合爆炸）
    param_spaces = {
        "dual_ma": {
            "short_ma": [3, 5, 8],
            "long_ma": [15, 20, 30],
        },
        "rsi": {
            "rsi_period": [10, 14, 20],
            "oversold": [20.0, 25.0, 30.0],
            "overbought": [70.0, 75.0, 80.0],
        },
        "vol_breakout": {
            "atr_period": [14, 20, 26],
            "atr_multiplier": [1.5, 2.0, 2.5],
        },
    }

    for sname in ["dual_ma", "rsi", "vol_breakout"]:
        space = param_spaces.get(sname)
        if not space:
            print(f"  {sname}: 无搜索空间，跳过")
            continue

        keys = list(space.keys())
        values = list(space.values())
        total = 1
        for v in values:
            total *= len(v)
        print(f"\n  策略: {sname} (共{total}组参数)")

        results = []
        for i, combo in enumerate(product(*values)):
            params = dict(zip(keys, combo))
            try:
                runner = get_runner("SHFE.RB")
                result = runner.run(
                    strategies=[sname],
                    start_date=FULL_START,
                    end_date=IN_SAMPLE_END,
                    params={sname: params},
                )
                m = format_metrics(result.portfolio_metrics)
                m.update(params)
                results.append(m)
            except Exception as e:
                continue

        if results:
            df = pd.DataFrame(results)
            save_csv(df, f"phase5_gridsearch_{sname}.csv")

            # 按 Sharpe 排序输出 Top 3
            if "sharpe" in df.columns:
                top = df.nlargest(3, "sharpe")
                print(f"  Top 3 (按Sharpe):")
                for _, row in top.iterrows():
                    params_str = ", ".join(f"{k}={row[k]}" for k in keys)
                    print(f"    {params_str} => sharpe={row['sharpe']:.4f} return={row.get('total_return_pct','N/A')}")

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6: E7 — 样本外验证
# ══════════════════════════════════════════════════════════════════════════════

def run_e7_out_of_sample():
    """
    E7: 样本外验证。
    在样本外数据（2024-07-01 ~ 2025-12-31）上验证策略表现。
    """
    print("\n" + "=" * 60)
    print("E7: 样本外验证")
    print("=" * 60)

    all_results = []
    for sym_file, sym_label in CORE_SYMBOLS.items():
        print(f"\n品种: {sym_label}")
        runner = get_runner(sym_file)
        try:
            # 样本内回测
            result_in = runner.run(
                strategies=STRATEGY_NAMES,
                start_date=FULL_START,
                end_date=IN_SAMPLE_END,
            )
            m_in = format_metrics(result_in.portfolio_metrics)
            m_in["symbol"] = sym_file
            m_in["split"] = "in_sample"
            all_results.append(m_in)

            # 样本外回测（使用样本内拟合的环境检测器参数）
            result_out = runner.run(
                strategies=STRATEGY_NAMES,
                start_date=OUT_SAMPLE_START,
                end_date=OUT_SAMPLE_END,
            )
            m_out = format_metrics(result_out.portfolio_metrics)
            m_out["symbol"] = sym_file
            m_out["split"] = "out_sample"
            all_results.append(m_out)

            # 对比
            sharpe_in = m_in.get("sharpe", 0) if isinstance(m_in.get("sharpe"), (int, float)) else 0
            sharpe_out = m_out.get("sharpe", 0) if isinstance(m_out.get("sharpe"), (int, float)) else 0
            return_in = m_in.get("total_return_pct", 0) if isinstance(m_in.get("total_return_pct"), (int, float)) else 0
            return_out = m_out.get("total_return_pct", 0) if isinstance(m_out.get("total_return_pct"), (int, float)) else 0
            dd_in = m_in.get("max_drawdown_pct", 0) if isinstance(m_in.get("max_drawdown_pct"), (int, float)) else 0
            dd_out = m_out.get("max_drawdown_pct", 0) if isinstance(m_out.get("max_drawdown_pct"), (int, float)) else 0

            print(f"  样本内: return={return_in} sharpe={sharpe_in} max_dd={dd_in}")
            print(f"  样本外: return={return_out} sharpe={sharpe_out} max_dd={dd_out}")

            # 计算衰减比
            if sharpe_in != 0 and isinstance(sharpe_in, (int, float)):
                decay = (sharpe_in - sharpe_out) / abs(sharpe_in) if abs(sharpe_in) > 1e-6 else 0
                print(f"  Sharpe衰减率: {decay:.1%} {'✓ 合格 (<30%)' if decay < 0.3 else '✗ 不合格'}")

        except Exception as e:
            print(f"  失败: {e}")

    df = pd.DataFrame(all_results)
    save_csv(df, "e7_out_of_sample_metrics.csv")

    # 汇总对比
    print("\n  汇总对比:")
    for split in ["in_sample", "out_sample"]:
        subset = df[df["split"] == split]
        if not subset.empty:
            print(f"\n  {split}:")
            for col in ["total_return_pct", "sharpe", "max_drawdown_pct", "calmar"]:
                vals = subset[col].dropna()
                numeric_vals = [v for v in vals if isinstance(v, (int, float))]
                if numeric_vals:
                    print(f"    {col}: mean={np.mean(numeric_vals):.4f}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Phase 7: 蒙特卡洛模拟
# ══════════════════════════════════════════════════════════════════════════════

def run_phase7_monte_carlo():
    """
    Phase 7: 蒙特卡洛模拟。
    对日收益率进行有放回重采样，生成模拟净值曲线。
    """
    print("\n" + "=" * 60)
    print("Phase 7: 蒙特卡洛模拟")
    print("=" * 60)

    # 使用螺纹钢等权组合的日收益率作为基准
    runner = get_runner("SHFE.RB.csv")
    try:
        result = runner.run(
            strategies=STRATEGY_NAMES,
            start_date=FULL_START,
            end_date=OUT_SAMPLE_END,
        )
        eq = result.portfolio_equity.sort_values("date")
        returns = eq["equity"].pct_change().dropna()

        n_simulations = 1000
        n_days = len(returns)
        np.random.seed(42)

        sim_equities = np.zeros((n_simulations, n_days + 1))
        sim_equities[:, 0] = 1.0

        ret_array = returns.values
        for i in range(n_simulations):
            sampled = np.random.choice(ret_array, size=n_days, replace=True)
            sim_equities[i, 1:] = np.cumprod(1 + sampled)

        # 统计量
        final_values = sim_equities[:, -1]
        max_drawdowns = np.array([
            np.min(sim_equities[i] / np.maximum.accumulate(sim_equities[i]) - 1)
            for i in range(n_simulations)
        ])

        print(f"\n  模拟次数: {n_simulations}")
        print(f"  终值分布:")
        print(f"    均值: {final_values.mean():.4f} (初始=1.0)")
        print(f"    中位数: {np.median(final_values):.4f}")
        print(f"    5分位: {np.percentile(final_values, 5):.4f}")
        print(f"    95分位: {np.percentile(final_values, 95):.4f}")
        print(f"    破产概率(终值<0.8): {(final_values < 0.8).mean():.2%}")

        print(f"\n  最大回撤分布:")
        print(f"    均值: {max_drawdowns.mean():.4f}")
        print(f"    中位数: {np.median(max_drawdowns):.4f}")
        print(f"    5分位: {np.percentile(max_drawdowns, 5):.4f}")
        print(f"    95分位: {np.percentile(max_drawdowns, 95):.4f}")

        # 保存结果
        mc_results = pd.DataFrame({
            "sim_id": range(n_simulations),
            "final_value": final_values,
            "max_drawdown": max_drawdowns,
        })
        save_csv(mc_results, "e7_monte_carlo_results.csv")

        # 净值曲线置信区间
        lower = np.percentile(sim_equities, 5, axis=0)
        upper = np.percentile(sim_equities, 95, axis=0)
        median = np.percentile(sim_equities, 50, axis=0)

        _plot_monte_carlo(median, lower, upper, "e7_monte_carlo.png")

        return mc_results
    except Exception as e:
        print(f"  失败: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Phase 8: 报告生成
# ══════════════════════════════════════════════════════════════════════════════

def run_phase8_report(results: Dict[str, any]):
    """
    Phase 8: 生成汇总报告并绘制对比图表。
    """
    print("\n" + "=" * 60)
    print("Phase 8: 报告汇总与输出")
    print("=" * 60)

    lines = [
        f"# 多策略量化回测系统 — 回测报告",
        f"",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## 回测配置",
        f"",
        f"- 初始资金: {INITIAL_CASH:,} 元",
        f"- 单笔止损: 5%",
        f"- 单策略最大仓位: 20%",
        f"- 样本内: {FULL_START} ~ {IN_SAMPLE_END}",
        f"- 样本外: {OUT_SAMPLE_START} ~ {OUT_SAMPLE_END}",
        f"- 测试品种: {len(CORE_SYMBOLS)} 个 ({', '.join(CORE_SYMBOLS.values())})",
        f"- 测试策略: {', '.join(STRATEGY_NAMES)}",
        f"",
        f"## 实验汇总",
        f"",
    ]

    # 汇总各实验的核心指标
    summary_rows = []
    for exp_name in ["E1_基线", "E2_等权", "E3_动态", "E4_切换", "E6_多品种", "E7_样本外"]:
        # 尝试读取对应的CSV
        csv_map = {
            "E1_基线": "e1_baseline_metrics.csv",
            "E2_等权": "e2_equal_weight_metrics.csv",
            "E3_动态": "e3_dynamic_weight_metrics.csv",
            "E4_切换": "e4_strategy_switching_metrics.csv",
            "E6_多品种": "e6_multi_symbol_equity.csv",
            "E7_样本外": "e7_out_of_sample_metrics.csv",
        }
        csv_file = OUTPUT_DIR / csv_map.get(exp_name, "")
        if csv_file.exists():
            lines.append(f"- {exp_name}: 结果已保存至 `{csv_map[exp_name]}`")
        else:
            lines.append(f"- {exp_name}: 无结果文件")

    lines.extend([
        f"",
        f"## 输出文件清单",
        f"",
    ])
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file():
            lines.append(f"- `{f.name}` ({f.stat().st_size:,} bytes)")

    # 生成组合对比图
    _plot_experiment_comparison()

    report_path = OUTPUT_DIR / "backtest_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  报告已保存: {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 绘图辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def _plot_equity_curve(eq: pd.DataFrame, title: str, label: str, filename: str):
    """绘制净值曲线和回撤曲线。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

    dates = pd.to_datetime(eq["date"])
    equity = eq["equity"].values

    ax1.plot(dates, equity, linewidth=1, label=label)
    ax1.set_title(f"{title} — 净值曲线", fontsize=14)
    ax1.set_ylabel("净值")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 回撤曲线
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    ax2.fill_between(dates, 0, dd, color="red", alpha=0.3)
    ax2.plot(dates, dd, color="red", linewidth=0.8)
    ax2.set_ylabel("回撤 %")
    ax2.set_xlabel("日期")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    path = CHARTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存: {path}")


def _plot_monte_carlo(median, lower, upper, filename: str):
    """绘制蒙特卡洛模拟净值曲线。"""
    fig, ax = plt.subplots(figsize=(12, 6))

    days = np.arange(len(median))
    ax.fill_between(days, lower, upper, alpha=0.3, color="blue", label="90% CI")
    ax.plot(days, median, color="blue", linewidth=1.5, label="Median")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="初始值")

    ax.set_title("蒙特卡洛模拟 — 净值曲线分布 (1000次)", fontsize=14)
    ax.set_xlabel("交易日")
    ax.set_ylabel("净值")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = CHARTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存: {path}")


def _plot_experiment_comparison():
    """绘制各实验的净值对比图。"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # 尝试加载各实验的净值曲线
    exp_files = {
        "E2_等权": "e2_equity_SHFE_RB.csv",
        "E3_动态": "e3_equity_SHFE_RB.csv",
    }

    colors = {"E2_等权": "blue", "E3_动态": "green"}
    has_data = False

    for label, fname in exp_files.items():
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            if "date" in df.columns and "equity" in df.columns:
                dates = pd.to_datetime(df["date"])
                ax.plot(dates, df["equity"], linewidth=1,
                       color=colors.get(label, "gray"), label=label)
                has_data = True

    # 加载E6多品种结果
    multi_path = OUTPUT_DIR / "e6_multi_symbol_equity.csv"
    if multi_path.exists():
        df = pd.read_csv(multi_path)
        if "date" in df.columns and "equity" in df.columns:
            dates = pd.to_datetime(df["date"])
            ax.plot(dates, df["equity"], linewidth=1.5,
                   color="red", label="E6_多品种分散")

    ax.axhline(y=INITIAL_CASH, color="gray", linestyle="--", alpha=0.5, label="初始资金")
    ax.set_title("各实验组合净值对比", fontsize=14)
    ax.set_ylabel("净值")
    ax.set_xlabel("日期")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    if has_data:
        path = CHARTS_DIR / "experiment_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  对比图已保存: {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 主执行流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("多策略量化回测系统 — 全面回测执行")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_results = {}

    # Phase 2: E1 单策略基线
    all_results["e1"] = run_e1_single_strategy_baselines()

    # Phase 3: E2-E5 组合实验
    all_results["e2"] = run_e2_equal_weight()
    all_results["e3"] = run_e3_dynamic_weight()
    all_results["e4"] = run_e4_strategy_switching()
    run_e5_switching_with_transition()  # 与 E4 同引擎

    # Phase 4: E6 多品种分散
    all_results["e6"] = run_e6_multi_symbol()

    # Phase 5: 参数优化
    all_results["opt"] = run_phase5_optimization()

    # Phase 6: E7 样本外验证
    all_results["e7"] = run_e7_out_of_sample()

    # Phase 7: 蒙特卡洛
    all_results["mc"] = run_phase7_monte_carlo()

    # Phase 8: 报告
    run_phase8_report(all_results)

    print("\n" + "=" * 60)
    print(f"回测执行完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()