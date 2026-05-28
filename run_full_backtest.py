#!/usr/bin/env python3
"""
多策略量化回测系统 — 完整回测执行脚本（整合版）

整合自 run_full_backtest.py 与 run_pybroker_full_backtest_v2.py，
保留两者核心业务功能，消除代码冗余，统一使用系统标准模块。

所有功能实现严格调用现有系统模块：
  1. 数据加载：core.engine.broker_adapter.create_hybrid_data_source
  2. 回测引擎：core.engine.broker_adapter.PyBrokerBacktestRunner
  3. 市场环境：core.market_regime.MarketRegimeDetector
  4. 策略库：core.strategy_library.StrategyLibrary
  5. 绩效指标：utils.metrics.MetricsCalculator
  6. 策略切换：core.engine.switch_engine.StrategySwitchEngine
  7. 配置管理：config.yaml

实验阶段：
  E1: 单策略基线回测
  E2: 等权信号融合
  E3: 环境动态加权
  E4: 策略切换（含过渡逻辑）
  E5: 多品种分散
  E6: WalkForward 滚动验证
  E7: 样本外验证
  E8: Bootstrap 置信区间
  E9: 蒙特卡洛模拟
  E10: HTML 报告生成
"""

import os
import sys
import yaml
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")

from loguru import logger

from core.engine.broker_adapter import (
    PyBrokerBacktestRunner,
    PyBrokerDataSource,
    PyBrokerResult,
    create_hybrid_data_source,
)
from core.config import BacktestConfig
from core.market_regime import MarketRegimeDetector
from core.performance import PerformanceEvaluator
from utils.metrics import MetricsCalculator


def load_config(config_path: str = "config.yaml") -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_tqsdk_credentials() -> Tuple[Optional[str], Optional[str]]:
    phone = os.getenv("TQSDK_PHONE")
    password = os.getenv("TQSDK_PASSWORD")
    if not phone or not password:
        logger.warning("TqSdk凭证未设置（环境变量 TQSDK_PHONE/TQSDK_PASSWORD），将仅使用CSV数据")
    return phone, password


def format_metrics(m: dict) -> dict:
    result = {}
    for k, v in m.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            result[k] = "N/A"
        elif isinstance(v, float):
            result[k] = round(v, 4)
        else:
            result[k] = v
    return result


def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {path}")


def _get_strategy_names(config: Dict) -> List[str]:
    return [s["name"] for s in config.get("strategies", []) if "name" in s]


_bt_config_cache: Dict[str, BacktestConfig] = {}


def _build_backtest_config(config: Dict, fusion_mode: bool = False) -> BacktestConfig:
    cache_key = f"{id(config)}_{fusion_mode}"
    if cache_key in _bt_config_cache:
        return _bt_config_cache[cache_key]
    bt_cfg = config["backtest"]
    risk_cfg = config["risk_management"]
    bt_config = BacktestConfig(
        initial_cash=bt_cfg.get("initial_cash", 1_000_000),
        commission_rate=bt_cfg.get("commission_rate", 0.0003),
        slippage_rate=bt_cfg.get("slippage_rate", 0.0002),
        stop_loss_pct=risk_cfg.get("stop_loss_pct", 0.05),
        max_position_pct=risk_cfg.get("position_limit_pct", 0.2),
        max_total_position_pct=risk_cfg.get("total_position_limit", 0.4),
        in_sample_end=bt_cfg.get("in_sample_end_date"),
        strategy_weights=config.get("strategy_weights", {}),
    )
    bt_config.fusion_mode = fusion_mode
    _bt_config_cache[cache_key] = bt_config
    return bt_config


def get_pybroker_runner(
    data_source: PyBrokerDataSource,
    config: Dict,
    strategies: Optional[List[str]] = None,
    fusion_mode: bool = False,
) -> PyBrokerBacktestRunner:
    bt_config = _build_backtest_config(config, fusion_mode=fusion_mode)
    symbols = config.get("symbols", [])
    runner = PyBrokerBacktestRunner(data_source, bt_config, target_symbols=symbols)
    if strategies:
        runner.register_strategies(strategies)
    return runner


# ══════════════════════════════════════════════════════════════════════════════
# E1: 单策略基线回测
# ══════════════════════════════════════════════════════════════════════════════


def run_e1_single_strategy_baselines(
    data_source: PyBrokerDataSource, config: Dict, output_dir: Path
) -> pd.DataFrame:
    logger.info("E1: 单策略基线回测")
    symbols = config.get("symbols", [])
    strategy_names = _get_strategy_names(config)
    bt_cfg = config["backtest"]

    all_results = []
    for sym in symbols:
        logger.info(f"  品种: {sym}")
        for sname in strategy_names:
            try:
                runner = get_pybroker_runner(
                    data_source, config, strategies=[sname]
                )
                result = runner.run(
                    start_date=bt_cfg["full_start_date"],
                    end_date=bt_cfg["full_end_date"],
                )
                m = format_metrics(result.metrics)
                m["symbol"] = sym
                m["strategy"] = sname
                all_results.append(m)
                logger.info(
                    f"  {sname}: return={m.get('total_return_pct', 'N/A')} "
                    f"sharpe={m.get('sharpe', 'N/A')} "
                    f"max_dd={m.get('max_drawdown_pct', 'N/A')}"
                )
            except Exception as e:
                logger.error(f"  {sname}: 失败 - {e}")
                all_results.append({"symbol": sym, "strategy": sname, "error": str(e)})

    df = pd.DataFrame(all_results)
    save_csv(df, output_dir / "e1_baseline_metrics.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# E2: 等权信号融合
# ══════════════════════════════════════════════════════════════════════════════


def run_e2_equal_weight(
    data_source: PyBrokerDataSource, config: Dict, output_dir: Path
) -> pd.DataFrame:
    logger.info("E2: 等权信号融合回测")
    symbols = config.get("symbols", [])
    strategy_names = _get_strategy_names(config)
    bt_cfg = config["backtest"]
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    all_results = []
    for sym in symbols:
        logger.info(f"  品种: {sym}")
        runner = get_pybroker_runner(
            data_source, config, strategies=strategy_names, fusion_mode=True
        )
        try:
            result = runner.run(
                start_date=bt_cfg["full_start_date"],
                end_date=bt_cfg["full_end_date"],
            )
            m = format_metrics(result.metrics)
            m["symbol"] = sym
            m["experiment"] = "E2_等权融合"
            all_results.append(m)
            logger.info(
                f"  portfolio: sharpe={m.get('sharpe', 'N/A')} "
                f"return={m.get('total_return_pct', 'N/A')}"
            )

            eq = result.equity_curve
            if not eq.empty:
                save_csv(eq.assign(symbol=sym), output_dir / f"e2_equity_{sym.replace('.', '_')}.csv")
                _plot_equity_curve(eq, sym, "E2_等权融合", charts_dir / f"e2_equity_{sym.replace('.', '_')}.png")
        except Exception as e:
            logger.error(f"  失败: {e}")

    df = pd.DataFrame(all_results)
    save_csv(df, output_dir / "e2_equal_weight_metrics.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# E3: 环境动态加权
# ══════════════════════════════════════════════════════════════════════════════


def run_e3_dynamic_weight(
    data_source: PyBrokerDataSource, config: Dict, output_dir: Path
) -> pd.DataFrame:
    logger.info("E3: 环境动态加权回测（execute 融合模式）")
    symbols = config.get("symbols", [])
    strategy_names = _get_strategy_names(config)
    bt_cfg = config["backtest"]

    all_results = []
    for sym in symbols:
        logger.info(f"  品种: {sym}")
        try:
            runner = get_pybroker_runner(
                data_source, config, strategies=strategy_names, fusion_mode=True
            )
            result = runner.run(
                start_date=bt_cfg["full_start_date"],
                end_date=bt_cfg["full_end_date"],
                use_execute_fusion=True,
            )
            m = format_metrics(result.metrics)
            m["symbol"] = sym
            m["experiment"] = "E3_动态权重"
            all_results.append(m)
            logger.info(
                f"  portfolio: return={m.get('total_return_pct', 'N/A')} "
                f"sharpe={m.get('sharpe', 'N/A')}"
            )

            eq = result.equity_curve
            if not eq.empty:
                save_csv(eq.assign(symbol=sym), output_dir / f"e3_equity_{sym.replace('.', '_')}.csv")
            if not result.regime_history.empty:
                save_csv(result.regime_history, output_dir / f"e3_regime_{sym.replace('.', '_')}.csv")
        except Exception as e:
            logger.error(f"  失败: {e}")

    df = pd.DataFrame(all_results)
    save_csv(df, output_dir / "e3_dynamic_weight_metrics.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# E4: 策略切换（含过渡逻辑）
# ══════════════════════════════════════════════════════════════════════════════


def run_e4_strategy_switching(
    data_source: PyBrokerDataSource, config: Dict, output_dir: Path
) -> pd.DataFrame:
    logger.info("E4: 策略切换回测（含过渡逻辑）")
    symbols = config.get("symbols", [])
    strategy_names = _get_strategy_names(config)
    bt_cfg = config["backtest"]

    all_results = []
    for sym in symbols:
        logger.info(f"  品种: {sym}")
        runner = get_pybroker_runner(
            data_source, config, strategies=strategy_names, fusion_mode=False
        )
        try:
            result = runner.run(
                start_date=bt_cfg["full_start_date"],
                end_date=bt_cfg["full_end_date"],
            )
            m = format_metrics(result.metrics)
            m["symbol"] = sym
            m["experiment"] = "E4_策略切换"
            all_results.append(m)
            logger.info(
                f"  portfolio: return={m.get('total_return_pct', 'N/A')} "
                f"sharpe={m.get('sharpe', 'N/A')}"
            )

            if not result.switch_log.empty:
                save_csv(result.switch_log, output_dir / f"e4_switch_log_{sym.replace('.', '_')}.csv")

            eq = result.equity_curve
            if not eq.empty:
                save_csv(eq.assign(symbol=sym), output_dir / f"e4_equity_{sym.replace('.', '_')}.csv")
        except Exception as e:
            logger.error(f"  失败: {e}")

    df = pd.DataFrame(all_results)
    save_csv(df, output_dir / "e4_strategy_switching_metrics.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# E5: 多品种分散
# ══════════════════════════════════════════════════════════════════════════════


def run_e5_multi_symbol(
    data_source: PyBrokerDataSource, config: Dict, output_dir: Path
) -> Optional[Dict]:
    logger.info("E5: 多品种分散回测")
    symbols = config.get("symbols", [])
    strategy_names = _get_strategy_names(config)
    bt_cfg = config["backtest"]
    initial_cash = bt_cfg.get("initial_cash", 1_000_000)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    all_equities = []
    strategy_returns_by_symbol = {}

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        runner = get_pybroker_runner(
            data_source, config, strategies=strategy_names, fusion_mode=False
        )
        try:
            result = runner.run(
                start_date=bt_cfg["full_start_date"],
                end_date=bt_cfg["full_end_date"],
            )
            eq = result.equity_curve
            if not eq.empty:
                eq = eq.copy()
                eq["symbol"] = sym
                all_equities.append(eq)

                eq_sorted = eq.sort_values("date")
                eq_sorted["daily_return"] = eq_sorted["equity"].pct_change()
                strategy_returns_by_symbol[sym] = eq_sorted[["date", "daily_return"]].set_index("date")
        except Exception as e:
            logger.error(f"  失败: {e}")

    if len(strategy_returns_by_symbol) >= 2:
        logger.info("  计算多品种等权组合...")
        combined_rets = None
        for sym, rets_df in strategy_returns_by_symbol.items():
            if combined_rets is None:
                combined_rets = rets_df.rename(columns={"daily_return": sym})
            else:
                combined_rets = combined_rets.join(rets_df.rename(columns={"daily_return": sym}), how="outer")

        if combined_rets is not None:
            combined_rets = combined_rets.fillna(0)
            portfolio_ret = combined_rets.mean(axis=1)
            portfolio_equity = (1 + portfolio_ret).cumprod() * initial_cash
            multi_eq = pd.DataFrame({"date": portfolio_equity.index, "equity": portfolio_equity.values})

            multi_metrics = PerformanceEvaluator.compute_metrics(portfolio_equity)
            m = format_metrics(multi_metrics)
            logger.info(
                f"  多品种组合: return={m.get('total_return_pct', 'N/A')} "
                f"sharpe={m.get('sharpe', 'N/A')}"
            )

            save_csv(multi_eq, output_dir / "e5_multi_symbol_equity.csv")

            corr_matrix = combined_rets.corr()
            save_csv(corr_matrix, output_dir / "e5_correlation_matrix.csv")

            _plot_equity_curve(multi_eq, "多品种等权组合", "E5_多品种分散", charts_dir / "e5_multi_symbol_equity.png")

            return {"metrics": m, "equity": multi_eq}

    return None


# ══════════════════════════════════════════════════════════════════════════════
# E6: WalkForward 滚动验证
# ══════════════════════════════════════════════════════════════════════════════


def run_e6_walkforward(
    data_source: PyBrokerDataSource, config: Dict, output_dir: Path
) -> pd.DataFrame:
    logger.info("E6: WalkForward 滚动验证")
    bt_cfg = config["backtest"]
    strategy_names = _get_strategy_names(config)

    all_wf_metrics = []
    for sname in strategy_names:
        try:
            runner = get_pybroker_runner(data_source, config, strategies=[sname])
            wf_result = runner.walkforward(
                start_date=bt_cfg["full_start_date"],
                end_date=bt_cfg["full_end_date"],
            )
            for w in wf_result.windows:
                w["strategy"] = sname
                all_wf_metrics.append(w)
            logger.info(f"  {sname}: {len(wf_result.windows)} 窗口, avg_sharpe={wf_result.overall_metrics.get('sharpe', 'N/A')}")
        except Exception as e:
            logger.error(f"  {sname} WalkForward 失败: {e}")

    df = pd.DataFrame(all_wf_metrics) if all_wf_metrics else pd.DataFrame()
    if not df.empty:
        save_csv(df, output_dir / "e6_walkforward_metrics.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# E7: 样本外验证
# ══════════════════════════════════════════════════════════════════════════════


def run_e7_out_of_sample(
    data_source: PyBrokerDataSource, config: Dict, output_dir: Path
) -> pd.DataFrame:
    logger.info("E7: 样本外验证")
    symbols = config.get("symbols", [])
    strategy_names = _get_strategy_names(config)
    bt_cfg = config["backtest"]
    in_sample_end = bt_cfg.get("in_sample_end_date", bt_cfg["full_end_date"])
    out_sample_start = bt_cfg.get("out_sample_start_date", in_sample_end)

    all_results = []
    for sym in symbols:
        logger.info(f"  品种: {sym}")
        try:
            runner_in = get_pybroker_runner(data_source, config, strategies=strategy_names, fusion_mode=False)
            result_in = runner_in.run(
                start_date=bt_cfg["full_start_date"],
                end_date=in_sample_end,
            )
            m_in = format_metrics(result_in.metrics)
            m_in["symbol"] = sym
            m_in["split"] = "in_sample"
            all_results.append(m_in)

            runner_out = get_pybroker_runner(data_source, config, strategies=strategy_names, fusion_mode=False)
            result_out = runner_out.run(
                start_date=out_sample_start,
                end_date=bt_cfg["full_end_date"],
            )
            m_out = format_metrics(result_out.metrics)
            m_out["symbol"] = sym
            m_out["split"] = "out_sample"
            all_results.append(m_out)

            def _safe(val):
                return val if isinstance(val, (int, float)) else 0

            sharpe_in = _safe(m_in.get("sharpe"))
            sharpe_out = _safe(m_out.get("sharpe"))
            if abs(sharpe_in) > 1e-6:
                decay = (sharpe_in - sharpe_out) / abs(sharpe_in)
                logger.info(f"  Sharpe衰减率: {decay:.1%} {'合格' if decay < 0.3 else '不合格'}")

        except Exception as e:
            logger.error(f"  失败: {e}")

    df = pd.DataFrame(all_results)
    save_csv(df, output_dir / "e7_out_of_sample_metrics.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# E8: Bootstrap 置信区间
# ══════════════════════════════════════════════════════════════════════════════


def run_e8_bootstrap(
    data_source: PyBrokerDataSource, config: Dict, output_dir: Path
) -> Tuple[List, pd.DataFrame]:
    logger.info("E8: Bootstrap 置信区间")
    bt_cfg = config["backtest"]
    bs_config = config.get("bootstrap", {})
    n_samples = bs_config.get("n_samples", 5000)
    strategy_names = _get_strategy_names(config)

    runner = get_pybroker_runner(data_source, config, strategies=strategy_names[:1] if strategy_names else ["dual_ma"])
    result = runner.run(
        start_date=bt_cfg["full_start_date"],
        end_date=bt_cfg["full_end_date"],
    )

    if result.equity_curve.empty:
        return [], pd.DataFrame()

    try:
        bootstrap_result = runner.bootstrap_metrics(n_samples=n_samples)
        logger.info(f"  Bootstrap 完成: {n_samples} 样本")
        logger.info(f"  结果: {bootstrap_result}")
    except Exception as e:
        logger.warning(f"  系统 Bootstrap 失败: {e}, 使用 MetricsCalculator 计算")
        try:
            equity = result.equity_curve["equity"]
            bootstrap_result = MetricsCalculator.bootstrap_confidence_interval(equity, n_samples=n_samples)
            logger.info(f"  MetricsCalculator Bootstrap 完成: {n_samples} 样本")
        except Exception as e2:
            logger.error(f"  MetricsCalculator 也失败: {e2}")
            bootstrap_result = None

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    if bootstrap_result is None:
        return [], pd.DataFrame()

    if isinstance(bootstrap_result, dict):
        sharpe_data = bootstrap_result.get("sharpe", {})
        if isinstance(sharpe_data, dict) and "ci_lower" in sharpe_data:
            rows = []
            for metric_name, vals in bootstrap_result.items():
                if isinstance(vals, dict) and "mean" in vals:
                    rows.append({"metric": metric_name, **vals})
            df_ci = pd.DataFrame(rows)
            save_csv(df_ci, output_dir / "e8_bootstrap_confidence_intervals.csv")
            logger.info(f"  Bootstrap 置信区间: {sharpe_data}")
            return [], df_ci

        sharpe_samples = []
        for key, val in bootstrap_result.items():
            if isinstance(val, list) and len(val) > 0:
                sharpe_samples = val
                break

        if sharpe_samples:
            df_samples = pd.DataFrame({"sharpe": sharpe_samples})
            save_csv(df_samples, output_dir / "e8_bootstrap_samples.csv")

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(sharpe_samples, bins=50, alpha=0.7, color="#1f77b4", edgecolor="black")
            ax.axvline(np.percentile(sharpe_samples, 5), color="#ff7f0e", linestyle="--", label="5% CI")
            ax.axvline(np.percentile(sharpe_samples, 95), color="#ff7f0e", linestyle="--", label="95% CI")
            ax.axvline(np.mean(sharpe_samples), color="#d62728", linestyle="-", label="Mean")
            ax.set_xlabel("Sharpe Ratio", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title(f"Bootstrap Sharpe Ratio Distribution (n={n_samples})", fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3)
            fig.savefig(charts_dir / "bootstrap_sharpe_distribution.png", dpi=config["output"].get("chart_dpi", 150), bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  Bootstrap 图表已保存")
            return sharpe_samples, df_samples

    return [], pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# E9: 蒙特卡洛模拟
# ══════════════════════════════════════════════════════════════════════════════


def run_e9_monte_carlo(
    data_source: PyBrokerDataSource, config: Dict, output_dir: Path
) -> Optional[pd.DataFrame]:
    logger.info("E9: 蒙特卡洛模拟")
    bt_cfg = config["backtest"]
    strategy_names = _get_strategy_names(config)
    mc_config = config.get("monte_carlo", {})
    n_simulations = mc_config.get("n_simulations", 1000)
    random_seed = mc_config.get("random_seed", 42)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    try:
        runner = get_pybroker_runner(data_source, config, strategies=strategy_names, fusion_mode=False)
        result = runner.run(
            start_date=bt_cfg["full_start_date"],
            end_date=bt_cfg["full_end_date"],
        )
        eq = result.equity_curve.sort_values("date")
        returns = eq["equity"].pct_change().dropna()

        n_days = len(returns)
        rng = np.random.default_rng(random_seed)

        sim_equities = np.zeros((n_simulations, n_days + 1))
        sim_equities[:, 0] = 1.0

        ret_array = returns.values
        for i in range(n_simulations):
            sampled = rng.choice(ret_array, size=n_days, replace=True)
            sim_equities[i, 1:] = np.cumprod(1 + sampled)

        final_values = sim_equities[:, -1]
        max_drawdowns = np.array(
            [np.min(sim_equities[i] / np.maximum.accumulate(sim_equities[i]) - 1) for i in range(n_simulations)]
        )

        logger.info(f"  模拟次数: {n_simulations}")
        logger.info(f"  终值均值: {final_values.mean():.4f}, 中位数: {np.median(final_values):.4f}")
        logger.info(f"  破产概率(终值<0.8): {(final_values < 0.8).mean():.2%}")

        mc_results = pd.DataFrame({
            "sim_id": range(n_simulations),
            "final_value": final_values,
            "max_drawdown": max_drawdowns,
        })
        save_csv(mc_results, output_dir / "e9_monte_carlo_results.csv")

        lower = np.percentile(sim_equities, 5, axis=0)
        upper = np.percentile(sim_equities, 95, axis=0)
        median = np.percentile(sim_equities, 50, axis=0)

        _plot_monte_carlo(median, lower, upper, charts_dir / "e9_monte_carlo.png")
        return mc_results
    except Exception as e:
        logger.error(f"  蒙特卡洛模拟失败: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# E10: HTML 报告生成
# ══════════════════════════════════════════════════════════════════════════════


def run_e10_html_report(
    config: Dict, results: Dict[str, PyBrokerResult], output_dir: Path
):
    logger.info("E10: 生成 HTML 报告")
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    bt_cfg = config["backtest"]

    charts_html = []

    fig, ax = plt.subplots(figsize=(14, 6))
    exp_names = []
    sharpe_values = []
    return_values = []
    for name, res in results.items():
        if hasattr(res, "metrics"):
            exp_names.append(name)
            sharpe_values.append(res.metrics.get("sharpe", 0))
            return_values.append(res.metrics.get("total_return_pct", 0))

    if exp_names:
        x = np.arange(len(exp_names))
        width = 0.35
        ax.bar(x - width / 2, sharpe_values, width, label="Sharpe", color="#1f77b4")
        ax.bar(x + width / 2, return_values, width, label="Return%", color="#ff7f0e")
        ax.set_xlabel("Experiment", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title("Experiment Comparison: Sharpe vs Return", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=15)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        bar_path = charts_dir / "experiment_comparison.png"
        fig.savefig(bar_path, dpi=config["output"].get("chart_dpi", 150), bbox_inches="tight")
        plt.close(fig)
        charts_html.append(f'<div class="chart-container"><h3>Experiment Comparison</h3><img src="charts/experiment_comparison.png" style="max-width:100%;border-radius:8px;"></div>')

    fig, ax = plt.subplots(figsize=(14, 6))
    has_curve = False
    for name, res in results.items():
        if hasattr(res, "equity_curve") and not res.equity_curve.empty:
            df = res.equity_curve.copy()
            df["date"] = pd.to_datetime(df["date"])
            ax.plot(df["date"], df["equity"], label=name, linewidth=2)
            has_curve = True
    if has_curve:
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Equity", fontsize=12)
        ax.set_title("Equity Curves", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        eq_path = charts_dir / "equity_curves.png"
        fig.savefig(eq_path, dpi=config["output"].get("chart_dpi", 150), bbox_inches="tight")
        charts_html.append(f'<div class="chart-container"><h3>Equity Curves</h3><img src="charts/equity_curves.png" style="max-width:100%;border-radius:8px;"></div>')
    plt.close(fig)

    risk_cfg = config["risk_management"]
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>量化回测报告</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 16px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }}
        h1 {{ color: #1f77b4; text-align: center; margin-bottom: 10px; }}
        h2 {{ color: #333; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; margin-top: 40px; }}
        .header-info {{ text-align: center; color: #666; font-size: 14px; margin-bottom: 30px; }}
        .chart-container {{ margin: 30px 0; text-align: center; }}
        table {{ width: 100%; border-collapse: collapse; margin: 25px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        th, td {{ padding: 14px; text-align: left; border: 1px solid #ddd; }}
        th {{ background: linear-gradient(180deg, #1f77b4 0%, #0a58ca 100%); color: white; font-weight: 600; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        .positive {{ color: #2ca02c; font-weight: bold; }}
        .negative {{ color: #d62728; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>量化回测报告</h1>
        <div class="header-info">
            生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 回测引擎: PyBroker (系统模块集成版)
        </div>
        <h2>配置说明</h2>
        <table>
            <tr><th>配置项</th><th>值</th></tr>
            <tr><td>初始资金</td><td>{bt_cfg.get('initial_cash', 0):,} 元</td></tr>
            <tr><td>回测区间</td><td>{bt_cfg.get('full_start_date', '')} ~ {bt_cfg.get('full_end_date', '')}</td></tr>
            <tr><td>样本内区间</td><td>{bt_cfg.get('full_start_date', '')} ~ {bt_cfg.get('in_sample_end_date', '')}</td></tr>
            <tr><td>样本外区间</td><td>{bt_cfg.get('out_sample_start_date', '')} ~ {bt_cfg.get('full_end_date', '')}</td></tr>
            <tr><td>品种</td><td>{', '.join(config.get('symbols', []))}</td></tr>
            <tr><td>单笔止损</td><td>-{risk_cfg.get('stop_loss_pct', 0.05)*100:.0f}%</td></tr>
        </table>
        <h2>可视化图表</h2>
        {''.join(charts_html)}
        <h2>绩效指标</h2>
        <table>
            <tr><th>实验</th><th>总收益率%</th><th>Sharpe</th><th>最大回撤%</th><th>交易次数</th></tr>
            {"".join([f"<tr><td>{name}</td><td class='{'positive' if res.metrics.get('total_return_pct', 0)>=0 else 'negative'}'>{res.metrics.get('total_return_pct', 0):.2f}</td><td>{res.metrics.get('sharpe', 0):.3f}</td><td class='negative'>{res.metrics.get('max_drawdown_pct', 0):.2f}</td><td>{res.metrics.get('trade_count', 0)}</td></tr>" for name, res in results.items() if hasattr(res, "metrics")])}
        </table>
    </div>
</body>
</html>"""

    html_path = output_dir / "backtest_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"HTML 报告已保存: {html_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 绘图辅助函数
# ══════════════════════════════════════════════════════════════════════════════


def _plot_equity_curve(eq: pd.DataFrame, title: str, label: str, path: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
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
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_monte_carlo(median, lower, upper, path: Path):
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
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 主执行流程
# ══════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 80)
    print("  多策略量化回测系统 — 完整回测执行（整合版）")
    print(f"  开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    config = load_config()
    log_config = config["logging"]
    log_dir = Path(log_config["log_dir"])
    log_dir.mkdir(exist_ok=True)
    logger.remove()
    logger.add(log_dir / log_config["log_file"], rotation=log_config["rotation"], retention=log_config["retention"], level=log_config["log_level"])
    logger.add(log_dir / log_config["error_file"], level="ERROR")
    logger.add(sys.stdout, level=log_config["log_level"])

    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(exist_ok=True)

    results = {}

    try:
        logger.info("加载数据...")
        phone, password = get_tqsdk_credentials()
        data_source = create_hybrid_data_source(
            phone=phone,
            password=password,
            symbols=config.get("symbols"),
            data_dir=config["data"]["csv_data_dir"],
            data_length=config["data"].get("tqsdk_data_length", 4000),
        )
        save_csv(data_source.to_pybroker_df(), output_dir / "data_summary.csv")

        logger.info("=" * 60)
        logger.info("E1: 单策略基线回测")
        run_e1_single_strategy_baselines(data_source, config, output_dir)

        logger.info("=" * 60)
        logger.info("E2: 等权信号融合")
        run_e2_equal_weight(data_source, config, output_dir)

        logger.info("=" * 60)
        logger.info("E3: 环境动态加权")
        run_e3_dynamic_weight(data_source, config, output_dir)

        logger.info("=" * 60)
        logger.info("E4: 策略切换（含过渡逻辑）")
        run_e4_strategy_switching(data_source, config, output_dir)

        logger.info("=" * 60)
        logger.info("E5: 多品种分散")
        run_e5_multi_symbol(data_source, config, output_dir)

        logger.info("=" * 60)
        logger.info("E6: WalkForward 滚动验证")
        run_e6_walkforward(data_source, config, output_dir)

        logger.info("=" * 60)
        logger.info("E7: 样本外验证")
        run_e7_out_of_sample(data_source, config, output_dir)

        logger.info("=" * 60)
        logger.info("E8: Bootstrap 置信区间")
        run_e8_bootstrap(data_source, config, output_dir)

        logger.info("=" * 60)
        logger.info("E9: 蒙特卡洛模拟")
        run_e9_monte_carlo(data_source, config, output_dir)

        logger.info("=" * 60)
        logger.info("E10: HTML 报告生成")
        bt_cfg = config["backtest"]
        strategy_names = _get_strategy_names(config)
        for sname in strategy_names[:3]:
            try:
                runner = get_pybroker_runner(data_source, config, strategies=[sname])
                res = runner.run(bt_cfg["full_start_date"], bt_cfg["full_end_date"])
                results[f"E1_{sname}"] = res
            except Exception as e:
                logger.error(f"策略 {sname} 失败: {e}")

        try:
            runner = get_pybroker_runner(data_source, config, strategies=strategy_names, fusion_mode=True)
            fusion_res = runner.run(bt_cfg["full_start_date"], bt_cfg["full_end_date"])
            results["E2_Fusion"] = fusion_res
        except Exception as e:
            logger.error(f"Fusion 失败: {e}")

        try:
            runner = get_pybroker_runner(data_source, config, strategies=strategy_names, fusion_mode=False)
            switch_res = runner.run(bt_cfg["full_start_date"], bt_cfg["full_end_date"])
            results["E4_Switching"] = switch_res
        except Exception as e:
            logger.error(f"Switching 失败: {e}")

        run_e10_html_report(config, results, output_dir)

        all_metrics = []
        for name, res in results.items():
            if hasattr(res, "metrics"):
                m = {"experiment": name, **res.metrics}
                all_metrics.append(m)
        if all_metrics:
            save_csv(pd.DataFrame(all_metrics), output_dir / "all_metrics.csv")

        logger.success("=" * 80)
        logger.success("回测完成")
        logger.success(f"输出目录: {output_dir.resolve()}")
        logger.success("=" * 80)

    except Exception as e:
        logger.exception(f"致命错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
