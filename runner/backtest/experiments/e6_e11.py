"""
实验 E6-E11：WalkForward、样本外、Bootstrap、蒙特卡洛、HTML报告、因子分析。

每个实验保持独立函数，委托 runner/backtest/runner.py 执行回测，
委托 runner/common/utils.py 和 runner/strategy/selector.py 处理工具和策略。
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from core.engine.backtest_runner import PyBrokerResult
from core.engine.pybroker_data_source import PyBrokerDataSource
from core.engine.rolling_ic import RollingICWeightEngine, RollingICConfig
from core.engine.factor_decay import FactorDecayMonitor, FactorDecayConfig, DecayStatus
from core.performance import PerformanceEvaluator
from utils.metrics import MetricsCalculator
from runner.backtest.runner import get_pybroker_runner, safe_run_backtest
from runner.common.utils import safe_float, is_valid_number, safe_div, save_csv, format_metrics
from runner.strategy.selector import get_strategy_names

_EPSILON = 1e-10
_SAFE_DECAY_THRESHOLD = 0.3


def run_e6_walkforward(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E6: WalkForward 滚动验证。
    对每个策略执行滚动窗口回测，评估参数稳定性。

    Returns:
        各窗口汇总指标 DataFrame
    """
    logger.info("E6: WalkForward 滚动验证")
    bt_cfg = config["backtest"]
    strategy_names = get_strategy_names(config)

    all_wf_metrics: List[Dict[str, Any]] = []
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
            logger.info(
                f"  {sname}: {len(wf_result.windows)} 窗口, "
                f"avg_sharpe={wf_result.overall_metrics.get('sharpe', 'N/A')}"
            )
        except Exception as e:
            logger.error(f"  {sname} WalkForward 失败: {e}")

    df = pd.DataFrame(all_wf_metrics) if all_wf_metrics else pd.DataFrame()
    if not df.empty:
        save_csv(df, output_dir / "e6_walkforward_metrics.csv")
    return df


def run_e7_out_of_sample(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E7: 样本外验证。
    将数据分为样本内和样本外两段，分别回测并比较 Sharpe 衰减率。

    Returns:
        汇总指标 DataFrame
    """
    logger.info("E7: 样本外验证")
    symbols: List[str] = config.get("symbols", [])
    strategy_names = get_strategy_names(config)
    bt_cfg = config["backtest"]
    in_sample_end = str(bt_cfg.get("in_sample_end_date", bt_cfg["full_end_date"]))
    out_sample_start = str(bt_cfg.get("out_sample_start_date", in_sample_end))

    all_results: List[Dict[str, Any]] = []
    primary_symbol = symbols[0] if symbols else None

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        try:
            runner_in = get_pybroker_runner(data_source, config, strategies=strategy_names)
            result_in = safe_run_backtest(runner_in, bt_cfg["full_start_date"], in_sample_end, f"E7_in_{sym}")
            if result_in is not None:
                m_in = format_metrics(result_in.metrics)
                m_in["symbol"] = sym
                m_in["split"] = "in_sample"
                all_results.append(m_in)
                if sym == primary_symbol:
                    eq_in = result_in.equity_curve
                    if eq_in is not None and not eq_in.empty:
                        save_csv(eq_in, output_dir / "e7_equity_in_sample.csv")

            runner_out = get_pybroker_runner(data_source, config, strategies=strategy_names)
            result_out = safe_run_backtest(runner_out, out_sample_start, bt_cfg["full_end_date"], f"E7_out_{sym}")
            if result_out is not None:
                m_out = format_metrics(result_out.metrics)
                m_out["symbol"] = sym
                m_out["split"] = "out_sample"
                all_results.append(m_out)
                if sym == primary_symbol:
                    eq_out = result_out.equity_curve
                    if eq_out is not None and not eq_out.empty:
                        save_csv(eq_out, output_dir / "e7_equity_out_sample.csv")

            # Sharpe 衰减率
            if result_in is not None and result_out is not None:
                sharpe_in = safe_float(result_in.metrics.get("sharpe", 0))
                sharpe_out = safe_float(result_out.metrics.get("sharpe", 0))
                if abs(sharpe_in) > _EPSILON:
                    decay = (sharpe_in - sharpe_out) / abs(sharpe_in)
                    is_qualified = decay < _SAFE_DECAY_THRESHOLD
                    logger.info(f"  Sharpe衰减率: {decay:.1%} {'合格' if is_qualified else '不合格'}")
        except Exception as e:
            logger.error(f"  {sym}: 失败 - {e}")

    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    save_csv(df, output_dir / "e7_out_of_sample_metrics.csv")
    return df


def run_e8_bootstrap(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> Tuple[List[float], pd.DataFrame]:
    """
    E8: Bootstrap 置信区间。

    对回测收益序列进行 Bootstrap 重采样，估计 Sharpe 等指标的置信区间。

    Returns:
        (sharpe_samples, 置信区间 DataFrame)
    """
    logger.info("E8: Bootstrap 置信区间")
    bt_cfg = config["backtest"]
    bs_config: Dict[str, Any] = config.get("bootstrap", {})
    n_samples = int(bs_config.get("n_samples", 5000))
    strategy_names = get_strategy_names(config)
    default_strategy = strategy_names[:1] if strategy_names else ["ts_momentum"]

    runner = get_pybroker_runner(data_source, config, strategies=default_strategy)
    result = safe_run_backtest(runner, bt_cfg["full_start_date"], bt_cfg["full_end_date"], "E8_base")

    if result is None or result.equity_curve is None or result.equity_curve.empty:
        logger.warning("E8: 无净值数据，跳过Bootstrap")
        return [], pd.DataFrame()

    bootstrap_result: Any = None
    try:
        bootstrap_result = runner.bootstrap_metrics(n_samples=n_samples)
        logger.info(f"  系统Bootstrap完成: {n_samples} 样本")
    except Exception as e:
        logger.warning(f"  系统Bootstrap失败: {e}, 回退到MetricsCalculator")
        try:
            equity = result.equity_curve["equity"].values
            bootstrap_result = MetricsCalculator.bootstrap_confidence_interval(equity, n_samples=n_samples)
            logger.info(f"  MetricsCalculator Bootstrap完成: {n_samples} 样本")
        except Exception as e2:
            logger.error(f"  MetricsCalculator Bootstrap也失败: {e2}")
            return [], pd.DataFrame()

    if bootstrap_result is None:
        return [], pd.DataFrame()

    # 结构化结果
    if isinstance(bootstrap_result, dict):
        first_val = next(iter(bootstrap_result.values()), None)
        if isinstance(first_val, dict) and "mean" in first_val:
            rows: List[Dict[str, Any]] = []
            for metric_name, vals in bootstrap_result.items():
                if isinstance(vals, dict):
                    rows.append({"metric": metric_name, **vals})
            df_ci = pd.DataFrame(rows)
            save_csv(df_ci, output_dir / "e8_bootstrap_confidence_intervals.csv")
            return [], df_ci

        sharpe_samples: List[float] = []
        for val in bootstrap_result.values():
            if isinstance(val, list) and len(val) > 0:
                sharpe_samples = val
                break

        if sharpe_samples:
            df_samples = pd.DataFrame({"sharpe": sharpe_samples})
            save_csv(df_samples, output_dir / "e8_bootstrap_samples.csv")
            return sharpe_samples, df_samples

    return [], pd.DataFrame()


def run_e9_monte_carlo(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    E9: 蒙特卡洛模拟。

    基于历史收益率序列，通过有放回重采样模拟未来净值路径分布。

    Returns:
        模拟结果 DataFrame，失败返回 None
    """
    logger.info("E9: 蒙特卡洛模拟")
    bt_cfg = config["backtest"]
    strategy_names = get_strategy_names(config)
    mc_config: Dict[str, Any] = config.get("monte_carlo", {})
    n_simulations = int(mc_config.get("n_simulations", 1000))
    random_seed = int(mc_config.get("random_seed", 42))
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    try:
        runner = get_pybroker_runner(data_source, config, strategies=strategy_names)
        result = safe_run_backtest(runner, bt_cfg["full_start_date"], bt_cfg["full_end_date"], "E9_base")
        if result is None:
            return None

        eq = result.equity_curve
        if eq is None or eq.empty:
            logger.warning("E9: 无净值数据，跳过蒙特卡洛模拟")
            return None

        eq_sorted = eq.sort_values("date")
        returns = eq_sorted["equity"].pct_change().dropna()
        returns = returns[returns.apply(is_valid_number)]

        if len(returns) == 0:
            logger.warning("E9: 无有效收益率数据")
            return None

        n_days = len(returns)
        rng = np.random.default_rng(random_seed)
        ret_array = returns.values

        sim_equities = np.zeros((n_simulations, n_days + 1))
        sim_equities[:, 0] = 1.0
        for i in range(n_simulations):
            sampled = rng.choice(ret_array, size=n_days, replace=True)
            sim_equities[i, 1:] = np.cumprod(1.0 + sampled)

        final_values = sim_equities[:, -1]
        peak_equities = np.maximum.accumulate(sim_equities, axis=1)
        peak_equities_safe = np.where(peak_equities > 0, peak_equities, 1.0)
        drawdowns = sim_equities / peak_equities_safe - 1.0
        max_drawdowns = np.min(drawdowns, axis=1)

        bankruptcy_prob = float(np.mean(final_values < 0.8))
        logger.info(f"  模拟次数: {n_simulations}")
        logger.info(f"  终值均值: {np.mean(final_values):.4f}, 中位数: {np.median(final_values):.4f}")
        logger.info(f"  破产概率(终值<0.8): {bankruptcy_prob:.2%}")

        mc_results = pd.DataFrame({
            "sim_id": range(n_simulations),
            "final_value": final_values,
            "max_drawdown": max_drawdowns,
        })
        save_csv(mc_results, output_dir / "e9_monte_carlo_results.csv")

        lower = np.percentile(sim_equities, 5, axis=0)
        upper = np.percentile(sim_equities, 95, axis=0)
        median = np.percentile(sim_equities, 50, axis=0)

        from runner.report.plots import plot_monte_carlo
        plot_monte_carlo(median, lower, upper, charts_dir / "e9_monte_carlo.png")
        return mc_results
    except Exception as e:
        logger.error(f"  蒙特卡洛模拟失败: {e}")
        return None


def run_e10_html_report(
    config: Dict[str, Any],
    results: Dict[str, PyBrokerResult],
    output_dir: Path,
    optimization_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    E10: 生成完整的量化回测分析 HTML 报告。

    委托 core/report_builder.generate_report()。
    """
    from core.report_builder import generate_report as build_report

    logger.info("E10: 生成完整 HTML 分析报告")

    strategies_data = {}
    for name, res in results.items():
        sd = {"metrics": dict(res.metrics) if hasattr(res, "metrics") and res.metrics else {}}
        if hasattr(res, "equity_curve") and res.equity_curve is not None and not res.equity_curve.empty:
            df = res.equity_curve
            sd["dates"] = df["date"].astype(str).tolist()
            sd["equity"] = df["equity"].astype(float).tolist()
        strategies_data[name] = sd

    if not strategies_data:
        logger.warning("E10: 无策略数据，跳过报告生成")
        return

    try:
        build_report(
            output_dir=str(output_dir),
            strategies_data=strategies_data,
            title="量化回测分析报告",
            subtitle=f"PyBroker 多策略回测 · {datetime.now().strftime('%Y-%m-%d')}",
            report_name="backtest_report_full.html",
        )
        logger.info(f"E10: 报告已保存至 {output_dir / 'backtest_report_full.html'}")
    except Exception as e:
        logger.error(f"E10 报告生成失败: {e}")


def run_e11_factor_analysis(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    E11: 滚动 IC 加权与因子衰减分析。

    对每个品种独立计算因子得分、滚动 IC、动态权重和衰减状态。

    Returns:
        {symbol: {ic_df, decay_df, summary}} 字典
    """
    logger.info("E11: 滚动IC加权与因子衰减分析")
    symbols: List[str] = config.get("symbols", [])
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    ic_config = RollingICConfig(window=60, forward_period=5, ema_alpha=0.1, min_observations=30)
    decay_config = FactorDecayConfig(
        trend_window=40, ic_healthy_threshold=0.03, ic_dead_threshold=0.01,
        max_consecutive_decline=5, decay_slope_threshold=-0.001,
    )

    all_results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    from runner.data.preprocessor import compute_factor_scores_from_ohlcv

    for symbol in symbols:
        logger.info(f"  分析品种: {symbol}")
        try:
            sym_df = data_source.query(
                data_source.date_range[0], data_source.date_range[1], symbols=[symbol]
            )
            if sym_df is None or len(sym_df) < 60:
                logger.warning(f"    {symbol}: 数据不足，跳过")
                continue

            scored = compute_factor_scores_from_ohlcv(sym_df)
            factor_names = ["ts_momentum", "roll_yield", "alpha019", "alpha032"]

            ic_engine = RollingICWeightEngine(ic_config)
            decay_monitor = FactorDecayMonitor(decay_config)

            ic_rows: List[Dict[str, Any]] = []

            for i in range(len(scored)):
                row = scored.iloc[i]
                forward_ret = float(row["forward_return"])
                if not is_valid_number(forward_ret):
                    continue

                factor_scores = {
                    name: float(row.get(name, 0.0))
                    for name in factor_names
                    if is_valid_number(row.get(name, 0.0))
                }
                if not factor_scores:
                    continue

                ic_engine.update(factor_scores, forward_ret, symbol)

                current_ic = ic_engine.current_ic
                for name, ic_val in current_ic.items():
                    decay_monitor.update(name, ic_val, str(row["date"])[:10])

                if i % 10 == 0:
                    current_weights = ic_engine.get_dynamic_weights()
                    ic_row = {"date": str(row["date"])[:10]}
                    for name, ic_val in current_ic.items():
                        ic_row[f"ic_{name}"] = round(ic_val, 6)
                    for name, w in current_weights.items():
                        ic_row[f"w_{name}"] = round(w, 4)
                    ic_rows.append(ic_row)

            ic_df = pd.DataFrame(ic_rows)
            if not ic_df.empty:
                save_csv(ic_df, output_dir / f"e11_ic_{symbol.replace('.', '_')}.csv")

            alerts = decay_monitor.check_decay()
            decay_rows = []
            for name in factor_names:
                if name in decay_monitor._ic_history:
                    ic_series = decay_monitor._ic_history[name]
                    decay_rows.append({
                        "date": str(scored["date"].iloc[-1])[:10],
                        "factor": name,
                        "current_ic": round(ic_series[-1], 6) if ic_series else 0.0,
                        "mean_ic": round(np.mean(ic_series), 6) if ic_series else 0.0,
                        "status": decay_monitor.current_status.get(name, DecayStatus.HEALTHY).value,
                    })
            decay_df = pd.DataFrame(decay_rows)
            if not decay_df.empty:
                save_csv(decay_df, output_dir / f"e11_decay_{symbol.replace('.', '_')}.csv")

            ic_summary = ic_engine.get_ic_summary()
            for name, stats in ic_summary.items():
                summary_rows.append({
                    "symbol": symbol,
                    "factor": name,
                    "mean_ic": round(stats.get("mean", 0.0), 6),
                    "std_ic": round(stats.get("std", 0.0), 6),
                    "ir": round(stats.get("ir", 0.0), 4),
                    "current_ic": round(stats.get("current", 0.0), 6),
                    "current_weight": round(ic_engine.get_dynamic_weights().get(name, 0.0), 4),
                })

            if not ic_df.empty:
                from runner.report.plots import plot_ic_analysis
                plot_ic_analysis(ic_df, f"{symbol} 滚动IC与动态权重", charts_dir / f"e11_ic_{symbol.replace('.', '_')}.png")

            all_results[symbol] = {
                "ic_df": ic_df,
                "decay_df": decay_df,
                "alerts": alerts,
                "final_weights": ic_engine.get_dynamic_weights(),
            }

            final_weights = ic_engine.get_dynamic_weights()
            logger.info(f"    {symbol}: 最终权重={({k: round(v, 4) for k, v in final_weights.items()})}")
        except Exception as e:
            logger.error(f"    {symbol} 因子分析失败: {e}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        save_csv(summary_df, output_dir / "e11_ic_summary.csv")
        logger.info("\n  因子IC汇总:")
        for _, row in summary_df.iterrows():
            logger.info(
                f"    {row['symbol']}/{row['factor']}: "
                f"mean_IC={row['mean_ic']:.4f}, IR={row['ir']:.2f}, "
                f"weight={row['current_weight']:.4f}"
            )

    return all_results
