#!/usr/bin/env python3
"""
多策略量化回测系统 — 全面回测执行脚本。

数据源: TqSdk 在线数据优先，本地 CSV 回退
引擎: PyBroker（不可用时自动回退自研简化引擎）
策略流程: 先判断市场环境 → 匹配策略 → 执行交易

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
from itertools import product

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

from core.config import BacktestConfig
from core.engine.broker_adapter import (
    PyBrokerBacktestRunner,
    PyBrokerDataSource,
    PyBrokerResult,
    create_hybrid_data_source,
)
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

CORE_SYMBOLS = {
    "SHFE.RB": "黑色-螺纹钢",
    "DCE.M": "农产品-豆粕",
    "CZCE.TA": "化工-PTA",
    "SHFE.CU": "有色-铜",
    "CFFEX.IF": "股指-沪深300",
}

IN_SAMPLE_END = "2024-06-30"
OUT_SAMPLE_START = "2024-07-01"
OUT_SAMPLE_END = "2025-12-31"
FULL_START = "2020-01-01"

# ── 数据源缓存 ──
_data_source_cache: Dict[str, PyBrokerDataSource] = {}


def _get_data_source(symbols: List[str]) -> PyBrokerDataSource:
    """
    获取混合数据源（TqSdk 优先 + CSV 回退）。
    使用缓存避免重复加载。
    """
    cache_key = ",".join(sorted(symbols))
    if cache_key not in _data_source_cache:
        print(f"  加载数据源: TqSdk 优先, CSV 回退, 品种={symbols}")
        _data_source_cache[cache_key] = create_hybrid_data_source(
            symbols=symbols,
            data_dir=DATA_DIR,
        )
    return _data_source_cache[cache_key]


def get_pybroker_runner(
    symbols: List[str],
    strategies: Optional[List[str]] = None,
    fusion_mode: bool = False,
) -> PyBrokerBacktestRunner:
    """
    创建基于 PyBroker 引擎的回测运行器。

    流程: TqSdk/CSV 数据 → 市场环境检测 → 策略匹配 → PyBroker 执行

    Args:
        symbols: 品种代码列表
        strategies: 策略名称列表
        fusion_mode: True=信号融合模式, False=策略切换模式
    """
    config = BacktestConfig(
        initial_cash=INITIAL_CASH,
        in_sample_end=IN_SAMPLE_END,
        strategy_weights={
            "dual_ma": 0.25,
            "rsi": 0.20,
            "vol_breakout": 0.25,
            "term_structure": 0.20,
            "spread": 0.10,
        },
        stop_loss_pct=0.05,
        max_position_pct=0.2,
        fusion_mode=fusion_mode,
    )
    data_source = _get_data_source(symbols)
    runner = PyBrokerBacktestRunner(data_source, config, target_symbols=symbols)
    if strategies:
        runner.register_strategies(strategies)
    return runner


def get_single_symbol_runner(
    sym_file: str,
    strategies: Optional[List[str]] = None,
    fusion_mode: bool = False,
) -> PyBrokerBacktestRunner:
    """创建单品种回测运行器（便捷方法）。"""
    return get_pybroker_runner(
        symbols=[sym_file],
        strategies=strategies,
        fusion_mode=fusion_mode,
    )


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
    使用 PyBroker 引擎，先判断市场环境再匹配策略。
    输出: e1_baseline_metrics.csv
    """
    print("\n" + "=" * 60)
    print("E1: 单策略基线回测 (PyBroker 引擎 + TqSdk 数据)")
    print("=" * 60)

    all_results = []
    for sym_file, sym_label in CORE_SYMBOLS.items():
        print(f"\n品种: {sym_label} ({sym_file})")
        for sname in STRATEGY_NAMES:
            try:
                runner = get_single_symbol_runner(sym_file, strategies=[sname])
                result = runner.run(
                    start_date=FULL_START,
                    end_date=OUT_SAMPLE_END,
                )
                m = format_metrics(result.metrics)
                m["symbol"] = sym_file
                m["symbol_label"] = sym_label
                m["strategy"] = sname
                all_results.append(m)
                print(
                    f"  {sname:15s}: return={m.get('total_return', m.get('total_return_pct', 'N/A'))} "
                    f"sharpe={m.get('sharpe', 'N/A')} max_dd={m.get('max_drawdown', m.get('max_drawdown_pct', 'N/A'))} "
                    f"trades={m.get('trade_count', 'N/A')}"
                )
            except Exception as e:
                print(f"  {sname:15s}: 失败 - {e}")
                all_results.append(
                    {"symbol": sym_file, "strategy": sname, "error": str(e)}
                )

    df = pd.DataFrame(all_results)
    save_csv(df, "e1_baseline_metrics.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: E2-E5 — 组合回测实验
# ══════════════════════════════════════════════════════════════════════════════


def run_e2_equal_weight():
    """
    E2: 等权组合 — 5策略信号融合（fusion_mode=True）。
    使用 PyBroker 引擎，各策略信号按波动率倒数加权融合。
    """
    print("\n" + "=" * 60)
    print("E2: 等权组合回测 (PyBroker 信号融合)")
    print("=" * 60)

    all_results = []
    for sym_file, sym_label in CORE_SYMBOLS.items():
        print(f"\n品种: {sym_label}")
        runner = get_single_symbol_runner(
            sym_file,
            strategies=STRATEGY_NAMES,
            fusion_mode=True,
        )
        try:
            result = runner.run(
                start_date=FULL_START,
                end_date=OUT_SAMPLE_END,
            )
            m = format_metrics(result.metrics)
            m["symbol"] = sym_file
            m["experiment"] = "E2_等权融合"
            all_results.append(m)
            print(
                f"  portfolio: return={m.get('total_return', m.get('total_return_pct', 'N/A'))} "
                f"sharpe={m.get('sharpe', 'N/A')} "
                f"max_dd={m.get('max_drawdown', m.get('max_drawdown_pct', 'N/A'))} "
                f"calmar={m.get('calmar', 'N/A')}"
            )

            eq = result.equity_curve
            if not eq.empty:
                save_csv(
                    eq.assign(symbol=sym_file),
                    f"e2_equity_{sym_file.replace('.', '_')}.csv",
                )
                _plot_equity_curve(
                    eq,
                    sym_label,
                    "E2_等权融合",
                    f"e2_equity_{sym_file.replace('.', '_')}.png",
                )
        except Exception as e:
            print(f"  失败: {e}")

    df = pd.DataFrame(all_results)
    save_csv(df, "e2_equal_weight_metrics.csv")
    return df


def run_e3_dynamic_weight():
    """
    E3: 环境动态加权 — 先判断市场环境，再动态调整策略权重。

    使用 PyBroker 引擎执行各策略独立回测，然后根据
    MarketRegimeDetector.get_regime_weights() 在每个交易日
    动态计算组合权重，得到动态加权组合净值。
    """
    print("\n" + "=" * 60)
    print("E3: 环境动态加权回测 (PyBroker 引擎)")
    print("=" * 60)

    all_results = []
    for sym_file, sym_label in CORE_SYMBOLS.items():
        print(f"\n品种: {sym_label}")
        try:
            # ── 逐策略独立运行 ──
            strategy_equity = {}
            strategy_regime = None

            for sname in STRATEGY_NAMES:
                try:
                    runner = get_single_symbol_runner(sym_file, strategies=[sname])
                    result = runner.run(
                        start_date=FULL_START,
                        end_date=OUT_SAMPLE_END,
                    )
                    eq = result.equity_curve
                    if not eq.empty and "equity" in eq.columns:
                        strategy_equity[sname] = eq
                    if strategy_regime is None and not result.regime_history.empty:
                        strategy_regime = result.regime_history
                except Exception:
                    pass

            if not strategy_equity:
                print("  跳过: 无策略结果")
                continue

            # ── 构建环境→权重映射 ──
            detector = MarketRegimeDetector()
            regime_weights_map = {}
            if strategy_regime is not None and "regime" in strategy_regime.columns:
                for regime_val in strategy_regime["regime"].unique():
                    if pd.notna(regime_val):
                        regime_weights_map[str(regime_val)] = (
                            detector.get_regime_weights(str(regime_val))
                        )

            # ── 收集各策略归一化净值 ──
            strategy_nav = {}
            for sname, eq_df in strategy_equity.items():
                nav = eq_df["equity"] / eq_df["equity"].iloc[0]
                strategy_nav[sname] = nav

            default_weights = {s: 1.0 / len(strategy_nav) for s in strategy_nav}

            # ── 动态加权计算组合净值 ──
            if strategy_regime is not None and "date" in strategy_regime.columns:
                date_regime = dict(
                    zip(
                        pd.to_datetime(strategy_regime["date"]),
                        strategy_regime["regime"],
                    )
                )
            else:
                date_regime = {}

            nav_len = min(len(v) for v in strategy_nav.values())
            portfolio_nav = np.ones(nav_len)

            # 使用第一个策略的日期序列作为基准
            ref_dates = pd.to_datetime(
                strategy_equity[list(strategy_equity.keys())[0]]["date"]
            )

            for i in range(nav_len):
                if i < len(ref_dates):
                    d = ref_dates.iloc[i]
                    regime_val = date_regime.get(d)
                    if regime_val and str(regime_val) in regime_weights_map:
                        w = regime_weights_map[str(regime_val)]
                        active_w = {s: w.get(s, 0.0) for s in strategy_nav}
                        total_w = sum(active_w.values())
                        if total_w > 0:
                            active_w = {s: v / total_w for s, v in active_w.items()}
                        else:
                            active_w = default_weights
                    else:
                        active_w = default_weights
                else:
                    active_w = default_weights

                weighted = sum(
                    active_w.get(s, 0) * float(strategy_nav[s].iloc[i])
                    for s in strategy_nav
                )
                portfolio_nav[i] = weighted

            portfolio_equity = pd.Series(portfolio_nav * INITIAL_CASH)

            metrics = PerformanceEvaluator.compute_metrics(portfolio_equity)
            m = format_metrics(metrics)
            m["symbol"] = sym_file
            m["experiment"] = "E3_动态权重"
            all_results.append(m)
            print(
                f"  portfolio: return={m.get('total_return_pct', 'N/A')} "
                f"sharpe={m.get('sharpe', 'N/A')} "
                f"max_dd={m.get('max_drawdown_pct', 'N/A')}"
            )

            eq_df = pd.DataFrame(
                {
                    "date": ref_dates.values[:nav_len],
                    "equity": portfolio_equity,
                }
            )
            save_csv(
                eq_df.assign(symbol=sym_file),
                f"e3_equity_{sym_file.replace('.', '_')}.csv",
            )

            if strategy_regime is not None:
                save_csv(
                    strategy_regime,
                    f"e3_regime_{sym_file.replace('.', '_')}.csv",
                )
        except Exception as e:
            print(f"  失败: {e}")
            import traceback

            traceback.print_exc()

    df = pd.DataFrame(all_results)
    save_csv(df, "e3_dynamic_weight_metrics.csv")
    return df


def run_e4_strategy_switching():
    """
    E4: 策略切换 — 先判断市场环境，再切换到最匹配的策略。
    使用 PyBroker 引擎 + StrategySwitchEngine（fusion_mode=False）。
    """
    print("\n" + "=" * 60)
    print("E4: 策略切换回测 (PyBroker 引擎 + 环境匹配)")
    print("=" * 60)

    all_results = []
    for sym_file, sym_label in CORE_SYMBOLS.items():
        print(f"\n品种: {sym_label}")
        runner = get_single_symbol_runner(
            sym_file,
            strategies=STRATEGY_NAMES,
            fusion_mode=False,
        )
        try:
            result = runner.run(
                start_date=FULL_START,
                end_date=OUT_SAMPLE_END,
            )
            m = format_metrics(result.metrics)
            m["symbol"] = sym_file
            m["experiment"] = "E4_策略切换"
            all_results.append(m)
            print(
                f"  portfolio: return={m.get('total_return', m.get('total_return_pct', 'N/A'))} "
                f"sharpe={m.get('sharpe', 'N/A')} "
                f"max_dd={m.get('max_drawdown', m.get('max_drawdown_pct', 'N/A'))}"
            )

            if not result.switch_log.empty:
                save_csv(
                    result.switch_log,
                    f"e4_switch_log_{sym_file.replace('.', '_')}.csv",
                )
                print(f"  策略切换记录: {len(result.switch_log)} 条")

            eq = result.equity_curve
            if not eq.empty:
                save_csv(
                    eq.assign(symbol=sym_file),
                    f"e4_equity_{sym_file.replace('.', '_')}.csv",
                )
        except Exception as e:
            print(f"  失败: {e}")

    df = pd.DataFrame(all_results)
    save_csv(df, "e4_strategy_switching_metrics.csv")
    return df


def run_e5_switching_with_transition():
    """
    E5: 策略切换+过渡 — 与E4相同但记录过渡期说明。
    StrategySwitchEngine 已内置过渡逻辑（cooldown 机制）。
    """
    print("\n" + "=" * 60)
    print("E5: 策略切换+过渡回测")
    print("=" * 60)
    run_e4_strategy_switching()
    print("  E5 与 E4 使用同一切换引擎，过渡逻辑已内置。")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: E6 — 多品种分散
# ══════════════════════════════════════════════════════════════════════════════


def run_e6_multi_symbol():
    """
    E6: 多品种分散。
    使用 PyBroker 引擎在所有品种上同时运行策略切换，
    实现跨品种分散化。
    """
    print("\n" + "=" * 60)
    print("E6: 多品种分散回测 (PyBroker 引擎)")
    print("=" * 60)

    all_equities = []
    strategy_returns_by_symbol = {}

    for sym_file, sym_label in CORE_SYMBOLS.items():
        print(f"\n品种: {sym_label}")
        runner = get_single_symbol_runner(
            sym_file,
            strategies=STRATEGY_NAMES,
            fusion_mode=False,
        )
        try:
            result = runner.run(
                start_date=FULL_START,
                end_date=OUT_SAMPLE_END,
            )
            eq = result.equity_curve
            if not eq.empty:
                eq = eq.copy()
                eq["symbol"] = sym_file
                all_equities.append(eq)

                eq_sorted = eq.sort_values("date")
                eq_sorted["daily_return"] = eq_sorted["equity"].pct_change()
                strategy_returns_by_symbol[sym_file] = eq_sorted[
                    ["date", "daily_return"]
                ].set_index("date")
                print(f"  equity range: {eq['date'].min()} ~ {eq['date'].max()}")
        except Exception as e:
            print(f"  失败: {e}")

    if len(strategy_returns_by_symbol) >= 2:
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
            portfolio_ret = combined_rets.mean(axis=1)
            portfolio_equity = (1 + portfolio_ret).cumprod() * INITIAL_CASH
            multi_eq = pd.DataFrame(
                {
                    "date": portfolio_equity.index,
                    "equity": portfolio_equity.values,
                }
            )

            evaluator = PerformanceEvaluator()
            multi_metrics = format_metrics(evaluator.compute_metrics(portfolio_equity))

            print(
                f"  多品种组合: return={multi_metrics.get('total_return_pct')} "
                f"sharpe={multi_metrics.get('sharpe')} "
                f"max_dd={multi_metrics.get('max_drawdown_pct')}"
            )

            save_csv(multi_eq, "e6_multi_symbol_equity.csv")

            corr_matrix = combined_rets.corr()
            save_csv(corr_matrix, "e6_correlation_matrix.csv")
            print(
                f"  策略间平均相关系数: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.4f}"
            )

            _plot_equity_curve(
                multi_eq,
                "多品种等权组合",
                "E6_多品种分散",
                "e6_multi_symbol_equity.png",
            )

            return {"metrics": multi_metrics, "equity": multi_eq}

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: 参数优化
# ══════════════════════════════════════════════════════════════════════════════


def run_phase5_optimization():
    """
    Phase 5: 样本内参数优化（网格搜索 + 稳健性评分）。
    使用 PyBroker 引擎在螺纹钢样本内数据上对核心参数进行扫描。
    v2: 增加稳健性评分 robust_score = 0.7 * sharpe + 0.3 * neighborhood_avg_sharpe
    """
    print("\n" + "=" * 60)
    print("Phase 5: 参数优化（PyBroker 引擎 + 网格搜索 + 稳健性评分）")
    print("=" * 60)

    param_spaces = {
        "dual_ma": {
            "short_ma": [3, 5, 8, 10, 12, 15],
            "long_ma": [20, 30, 40, 50, 60],
        },
        "rsi": {
            "rsi_period": [10, 14, 20, 28],
            "oversold": [20.0, 25.0, 30.0],
            "overbought": [75.0, 80.0, 85.0],
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
                runner = get_single_symbol_runner("SHFE.RB", strategies=[sname])
                result = runner.run(
                    start_date=FULL_START,
                    end_date=IN_SAMPLE_END,
                    custom_params={sname: params},
                )
                m = format_metrics(result.metrics)
                m.update(params)
                results.append(m)
            except Exception:
                continue

        if results:
            df = pd.DataFrame(results)

            if "sharpe" in df.columns:
                robust_scores = {}
                for idx in df.index:
                    own_sharpe = df.loc[idx, "sharpe"]
                    neighbor_sharpes = []
                    row = df.loc[idx]
                    for col in keys:
                        if col not in df.columns:
                            continue
                        current_val = row[col]
                        param_values = space[col]
                        try:
                            current_pos = param_values.index(current_val)
                        except (ValueError, AttributeError):
                            continue
                        for direction in [-1, 1]:
                            neighbor_pos = current_pos + direction
                            if 0 <= neighbor_pos < len(param_values):
                                neighbor_val = param_values[neighbor_pos]
                                neighbor_rows = df[df[col] == neighbor_val]
                                if not neighbor_rows.empty:
                                    neighbor_mean = neighbor_rows["sharpe"].mean()
                                    if not np.isnan(neighbor_mean):
                                        neighbor_sharpes.append(neighbor_mean)
                    neighbor_avg = (
                        np.mean(neighbor_sharpes) if neighbor_sharpes else own_sharpe
                    )
                    robust_scores[idx] = 0.7 * own_sharpe + 0.3 * neighbor_avg

                df["robust_score"] = df.index.map(robust_scores)

            save_csv(df, f"phase5_gridsearch_{sname}.csv")

            if "robust_score" in df.columns:
                top = df.nlargest(3, "robust_score")
                print(f"  Top 3 (按稳健性评分):")
                for _, row in top.iterrows():
                    params_str = ", ".join(f"{k}={row[k]}" for k in keys)
                    print(
                        f"    {params_str} => robust={row['robust_score']:.4f} "
                        f"sharpe={row['sharpe']:.4f} "
                        f"return={row.get('total_return_pct', 'N/A')}"
                    )
            elif "sharpe" in df.columns:
                top = df.nlargest(3, "sharpe")
                print(f"  Top 3 (按Sharpe):")
                for _, row in top.iterrows():
                    params_str = ", ".join(f"{k}={row[k]}" for k in keys)
                    print(
                        f"    {params_str} => sharpe={row['sharpe']:.4f} "
                        f"return={row.get('total_return_pct', 'N/A')}"
                    )

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6: E7 — 样本外验证
# ══════════════════════════════════════════════════════════════════════════════


def run_e7_out_of_sample():
    """
    E7: 样本外验证。
    使用 PyBroker 引擎在样本外数据上验证策略表现。
    """
    print("\n" + "=" * 60)
    print("E7: 样本外验证 (PyBroker 引擎)")
    print("=" * 60)

    all_results = []
    for sym_file, sym_label in CORE_SYMBOLS.items():
        print(f"\n品种: {sym_label}")
        try:
            # 样本内回测
            runner_in = get_single_symbol_runner(
                sym_file,
                strategies=STRATEGY_NAMES,
                fusion_mode=False,
            )
            result_in = runner_in.run(
                start_date=FULL_START,
                end_date=IN_SAMPLE_END,
            )
            m_in = format_metrics(result_in.metrics)
            m_in["symbol"] = sym_file
            m_in["split"] = "in_sample"
            all_results.append(m_in)

            # 样本外回测
            runner_out = get_single_symbol_runner(
                sym_file,
                strategies=STRATEGY_NAMES,
                fusion_mode=False,
            )
            result_out = runner_out.run(
                start_date=OUT_SAMPLE_START,
                end_date=OUT_SAMPLE_END,
            )
            m_out = format_metrics(result_out.metrics)
            m_out["symbol"] = sym_file
            m_out["split"] = "out_sample"
            all_results.append(m_out)

            # 对比
            def _safe(val):
                return val if isinstance(val, (int, float)) else 0

            sharpe_in = _safe(m_in.get("sharpe"))
            sharpe_out = _safe(m_out.get("sharpe"))
            return_in = _safe(m_in.get("total_return", m_in.get("total_return_pct", 0)))
            return_out = _safe(
                m_out.get("total_return", m_out.get("total_return_pct", 0))
            )
            dd_in = _safe(m_in.get("max_drawdown", m_in.get("max_drawdown_pct", 0)))
            dd_out = _safe(m_out.get("max_drawdown", m_out.get("max_drawdown_pct", 0)))

            print(f"  样本内: return={return_in} sharpe={sharpe_in} max_dd={dd_in}")
            print(f"  样本外: return={return_out} sharpe={sharpe_out} max_dd={dd_out}")

            if abs(sharpe_in) > 1e-6:
                decay = (sharpe_in - sharpe_out) / abs(sharpe_in)
                print(
                    f"  Sharpe衰减率: {decay:.1%} "
                    f"{'✓ 合格 (<30%)' if decay < 0.3 else '✗ 不合格'}"
                )

        except Exception as e:
            print(f"  失败: {e}")

    df = pd.DataFrame(all_results)
    save_csv(df, "e7_out_of_sample_metrics.csv")

    print("\n  汇总对比:")
    for split in ["in_sample", "out_sample"]:
        subset = df[df["split"] == split]
        if not subset.empty:
            print(f"\n  {split}:")
            for col in [
                "sharpe",
                "max_drawdown",
                "max_drawdown_pct",
                "calmar",
                "total_return_pct",
            ]:
                if col not in subset.columns:
                    continue
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
    使用 PyBroker 引擎的回测结果作为基准，对日收益率进行重采样。
    """
    print("\n" + "=" * 60)
    print("Phase 7: 蒙特卡洛模拟")
    print("=" * 60)

    try:
        runner = get_single_symbol_runner(
            "SHFE.RB",
            strategies=STRATEGY_NAMES,
            fusion_mode=False,
        )
        result = runner.run(
            start_date=FULL_START,
            end_date=OUT_SAMPLE_END,
        )
        eq = result.equity_curve.sort_values("date")
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

        final_values = sim_equities[:, -1]
        max_drawdowns = np.array(
            [
                np.min(sim_equities[i] / np.maximum.accumulate(sim_equities[i]) - 1)
                for i in range(n_simulations)
            ]
        )

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

        mc_results = pd.DataFrame(
            {
                "sim_id": range(n_simulations),
                "final_value": final_values,
                "max_drawdown": max_drawdowns,
            }
        )
        save_csv(mc_results, "e7_monte_carlo_results.csv")

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
        f"- 数据源: TqSdk 在线数据优先, 本地 CSV 回退",
        f"- 引擎: PyBroker (不可用时自动回退自研简化引擎)",
        f"- 策略流程: 先判断市场环境 → 匹配策略 → 执行交易",
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

    csv_map = {
        "E1_基线": "e1_baseline_metrics.csv",
        "E2_等权": "e2_equal_weight_metrics.csv",
        "E3_动态": "e3_dynamic_weight_metrics.csv",
        "E4_切换": "e4_strategy_switching_metrics.csv",
        "E6_多品种": "e6_multi_symbol_equity.csv",
        "E7_样本外": "e7_out_of_sample_metrics.csv",
    }
    for exp_name, csv_name in csv_map.items():
        csv_file = OUTPUT_DIR / csv_name
        if csv_file.exists():
            lines.append(f"- {exp_name}: 结果已保存至 `{csv_name}`")
        else:
            lines.append(f"- {exp_name}: 无结果文件")

    lines.extend(
        [
            f"",
            f"## 输出文件清单",
            f"",
        ]
    )
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file():
            lines.append(f"- `{f.name}` ({f.stat().st_size:,} bytes)")

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
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    dates = pd.to_datetime(eq["date"])
    equity = eq["equity"].values

    ax1.plot(dates, equity, linewidth=1, label=label)
    ax1.set_title(f"{title} — 净值曲线", fontsize=14)
    ax1.set_ylabel("净值")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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

    exp_files = {
        "E2_等权": "e2_equity_SHFE_RB.csv",
        "E3_动态": "e3_equity_SHFE_RB.csv",
        "E4_切换": "e4_equity_SHFE_RB.csv",
    }

    colors = {"E2_等权": "blue", "E3_动态": "green", "E4_切换": "orange"}
    has_data = False

    for label, fname in exp_files.items():
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            if "date" in df.columns and "equity" in df.columns:
                dates = pd.to_datetime(df["date"])
                ax.plot(
                    dates,
                    df["equity"],
                    linewidth=1,
                    color=colors.get(label, "gray"),
                    label=label,
                )
                has_data = True

    multi_path = OUTPUT_DIR / "e6_multi_symbol_equity.csv"
    if multi_path.exists():
        df = pd.read_csv(multi_path)
        if "date" in df.columns and "equity" in df.columns:
            dates = pd.to_datetime(df["date"])
            ax.plot(
                dates, df["equity"], linewidth=1.5, color="red", label="E6_多品种分散"
            )

    ax.axhline(
        y=INITIAL_CASH, color="gray", linestyle="--", alpha=0.5, label="初始资金"
    )
    ax.set_title("各实验组合净值对比 (PyBroker 引擎)", fontsize=14)
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
    print(f"数据源: TqSdk 在线数据优先, 本地 CSV 回退")
    print(f"引擎: PyBroker (不可用时自动回退自研简化引擎)")
    print(f"策略流程: 先判断市场环境 → 匹配策略 → 执行交易")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_results = {}

    all_results["e1"] = run_e1_single_strategy_baselines()

    all_results["e2"] = run_e2_equal_weight()
    all_results["e3"] = run_e3_dynamic_weight()
    all_results["e4"] = run_e4_strategy_switching()
    run_e5_switching_with_transition()

    all_results["e6"] = run_e6_multi_symbol()

    all_results["opt"] = run_phase5_optimization()

    all_results["e7"] = run_e7_out_of_sample()

    all_results["mc"] = run_phase7_monte_carlo()

    run_phase8_report(all_results)

    print("\n" + "=" * 60)
    print(f"回测执行完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
