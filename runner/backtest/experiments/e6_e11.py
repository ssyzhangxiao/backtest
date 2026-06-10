"""
实验 E6-E11：WalkForward、样本外、Bootstrap、蒙特卡洛、HTML报告、因子分析。

每个实验保持独立函数，委托 runner/backtest/runner.py 执行回测，
委托 runner/common/utils.py 和 runner/strategy/selector.py 处理工具和策略。
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from core.engine.backtest_runner import PyBrokerResult
from core.engine.pybroker_data_source import PyBrokerDataSource
from core.factors.factor_evaluator import FactorEvaluator
from core.performance import PerformanceEvaluator
from core.validation.monte_carlo import MonteCarloSimulator
from utils.metrics import MetricsCalculator
from runner.backtest.runner import get_pybroker_runner, safe_run_backtest
from runner.common.utils import (
    safe_float,
    is_valid_number,
    safe_div,
    save_csv,
    format_metrics,
    sanitize_filename,
    save_equity_curve,
    handle_backtest_errors,
)
from runner.common.config_utils import (
    get_walkforward_config,
    get_montecarlo_config,
    get_factors_list,
)
from runner.strategy.selector import get_strategy_names


_EPSILON = 1e-10
_SAFE_DECAY_THRESHOLD = 0.3


# ============================================
# E8：Bootstrap Result 数据类
# ============================================


@dataclass
class BootstrapResult:
    """Bootstrap 结果数据类"""

    sharpe_samples: List[float]
    confidence_intervals: pd.DataFrame
    n_samples: int
    random_seed: int


# ============================================
# 实验函数
# ============================================


@handle_backtest_errors(return_value=pd.DataFrame())
def run_e6_walkforward(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E6：WalkForward 滚动验证。

    对每个策略执行滚动窗口回测，评估参数稳定性。
    从配置读取 walkforward.window 和 walkforward.step。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        各窗口汇总指标 DataFrame
    """
    logger.info("E6：WalkForward 滚动验证")
    bt_cfg = config["backtest"]
    wf_cfg = get_walkforward_config(config)
    strategy_names = get_strategy_names(config)
    symbols: List[str] = config.get("symbols", [])

    all_wf_metrics: List[Dict[str, Any]] = []
    for sname in strategy_names:
        try:
            # 修复 per-symbol 隔离 bug：传 target_symbols=全部品种（walkforward 内部不分品种）
            runner = get_pybroker_runner(
                data_source, config, strategies=[sname], target_symbols=symbols
            )
            wf_result = runner.walkforward(
                start_date=bt_cfg["full_start_date"],
                end_date=bt_cfg["full_end_date"],
            )
            # 修复 2026-06-10：兼容 _WindowRunner._compute_simple_metrics
            # 返回的 metrics 字段（既有 total_return 也有 total_return_pct，
            # 但 DataFrame 构造时嵌套 dict 会触发 KeyError）。
            # 显式平铺 metrics 到外层 dict，避免 pandas 内部 dict 转换失败。
            for w in wf_result.windows:
                flat_w = dict(w)  # 浅拷贝
                metrics_nested = flat_w.pop("metrics", {}) or {}
                for mk, mv in metrics_nested.items():
                    flat_w[f"metric_{mk}"] = mv
                flat_w["strategy"] = sname
                all_wf_metrics.append(flat_w)
            logger.info(
                f"  {sname}: {len(wf_result.windows)} 窗口, "
                f"avg_sharpe={wf_result.overall_metrics.get('sharpe', 'N/A')}"
            )
        except Exception as e:
            logger.error(f"  {sname} WalkForward 失败: {e}")
            logger.exception(f"  {sname} WalkForward 异常详情")

    df = pd.DataFrame(all_wf_metrics) if all_wf_metrics else pd.DataFrame()
    if not df.empty:
        save_csv(df, output_dir / "e6_walkforward_metrics.csv")
    return df


@handle_backtest_errors(return_value=pd.DataFrame())
def run_e7_out_of_sample(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E7：样本外验证。

    将数据分为样本内和样本外两段，分别回测并比较 Sharpe 衰减率。
    强制要求 in_sample_end_date 和 out_sample_start_date，
    如果未配置则自动按 7:3 比例划分。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        汇总指标 DataFrame
    """
    logger.info("E7：样本外验证")
    symbols: List[str] = config.get("symbols", [])
    strategy_names = get_strategy_names(config)
    bt_cfg = config["backtest"]

    # 确定样本划分日期
    full_start = pd.to_datetime(bt_cfg["full_start_date"])
    full_end = pd.to_datetime(bt_cfg["full_end_date"])

    if "in_sample_end_date" in bt_cfg and bt_cfg["in_sample_end_date"]:
        in_sample_end = pd.to_datetime(bt_cfg["in_sample_end_date"])
    else:
        # P1 整改：使用交易日计数（pd.date_range + len(df['date'].unique())）
        # 比日历天数更准确：避免节假日差异导致样本划分偏移
        # 由于此阶段数据源尚未加载，我们以"工作日频率"近似估计交易日数
        # 实际确切的交易日计数由回测侧 date_range 提供；此处用作粗估
        trading_days_total = len(
            pd.date_range(full_start, full_end, freq="B")  # B = 工作日
        )
        split_day_idx = int(trading_days_total * 0.7)
        # 将 idx 反推为日期：取 start + 70% 个工作日
        in_sample_end = full_start + pd.tseries.offsets.BDay(split_day_idx)
        logger.info(
            f"  自动划分样本内结束日期: {in_sample_end.date()} "
            f"（{trading_days_total} 个工作日中前 {split_day_idx} 天）"
        )

    if "out_sample_start_date" in bt_cfg and bt_cfg["out_sample_start_date"]:
        out_sample_start = pd.to_datetime(bt_cfg["out_sample_start_date"])
    else:
        # 从样本内结束日开始
        out_sample_start = in_sample_end
        logger.info(f"  自动设置样本外开始日期: {out_sample_start.date()}")

    all_results: List[Dict[str, Any]] = []
    primary_symbol = symbols[0] if symbols else None

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        try:
            # 样本内回测（修复 per-symbol 隔离 bug：仅对当前品种做回测）
            runner_in = get_pybroker_runner(
                data_source, config, strategies=strategy_names, target_symbols=[sym]
            )
            result_in = safe_run_backtest(
                runner_in,
                str(full_start.date()),
                str(in_sample_end.date()),
                f"E7_in_{sym}",
            )
            if result_in is not None:
                m_in = format_metrics(result_in.metrics)
                m_in["symbol"] = sym
                m_in["split"] = "in_sample"
                all_results.append(m_in)
                if sym == primary_symbol:
                    eq_in = result_in.equity_curve
                    if eq_in is not None and not eq_in.empty:
                        save_equity_curve(eq_in, output_dir, "e7_equity_in_sample")

            # 样本外回测（修复 per-symbol 隔离 bug：仅对当前品种做回测）
            runner_out = get_pybroker_runner(
                data_source, config, strategies=strategy_names, target_symbols=[sym]
            )
            result_out = safe_run_backtest(
                runner_out,
                str(out_sample_start.date()),
                str(full_end.date()),
                f"E7_out_{sym}",
            )
            if result_out is not None:
                m_out = format_metrics(result_out.metrics)
                m_out["symbol"] = sym
                m_out["split"] = "out_sample"
                all_results.append(m_out)
                if sym == primary_symbol:
                    eq_out = result_out.equity_curve
                    if eq_out is not None and not eq_out.empty:
                        save_equity_curve(eq_out, output_dir, "e7_equity_out_sample")

            # Sharpe 衰减率
            if result_in is not None and result_out is not None:
                sharpe_in = safe_float(result_in.metrics.get("sharpe", 0))
                sharpe_out = safe_float(result_out.metrics.get("sharpe", 0))
                if abs(sharpe_in) > _EPSILON:
                    decay = (sharpe_in - sharpe_out) / abs(sharpe_in)
                    is_qualified = decay < _SAFE_DECAY_THRESHOLD
                    logger.info(
                        f"  Sharpe衰减率: {decay:.1%} {'合格' if is_qualified else '不合格'}"
                    )
        except Exception as e:
            logger.error(f"  {sym}: 失败 - {e}")

    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    save_csv(df, output_dir / "e7_out_of_sample_metrics.csv")
    return df


@handle_backtest_errors(return_value=BootstrapResult([], pd.DataFrame(), 0, 42))
def run_e8_bootstrap(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> BootstrapResult:
    """
    E8：Bootstrap 置信区间。

    对回测收益序列进行 Bootstrap 重采样，估计 Sharpe 等指标的置信区间。
    定义 BootstrapResult 数据类统一返回格式，添加随机种子。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        BootstrapResult 对象
    """
    logger.info("E8：Bootstrap 置信区间")
    bt_cfg = config["backtest"]
    bs_cfg: Dict[str, Any] = config.get("bootstrap", {})
    n_samples = int(bs_cfg.get("n_samples", 5000))
    random_seed = int(bs_cfg.get("random_seed", 42))
    strategy_names = get_strategy_names(config)
    default_strategy = strategy_names[:1] if strategy_names else ["trend"]

    runner = get_pybroker_runner(data_source, config, strategies=default_strategy)
    result = safe_run_backtest(
        runner, bt_cfg["full_start_date"], bt_cfg["full_end_date"], "E8_base"
    )

    if result is None or result.equity_curve is None or result.equity_curve.empty:
        logger.warning("E8：无净值数据，跳过Bootstrap")
        return BootstrapResult([], pd.DataFrame(), n_samples, random_seed)

    bootstrap_result: Any = None
    try:
        bootstrap_result = runner.bootstrap_metrics(n_samples=n_samples)
        logger.info(f"  系统Bootstrap完成: {n_samples} 样本")
    except Exception as e:
        logger.warning(f"  系统Bootstrap失败: {e}, 回退到MetricsCalculator")
        try:
            equity = result.equity_curve["equity"].values
            bootstrap_result = MetricsCalculator.bootstrap_confidence_interval(
                equity, n_samples=n_samples
            )
            logger.info(f"  MetricsCalculator Bootstrap完成: {n_samples} 样本")
        except Exception as e2:
            logger.error(f"  MetricsCalculator Bootstrap也失败: {e2}")
            return BootstrapResult([], pd.DataFrame(), n_samples, random_seed)

    if bootstrap_result is None:
        return BootstrapResult([], pd.DataFrame(), n_samples, random_seed)

    # 结构化结果
    sharpe_samples: List[float] = []
    df_ci = pd.DataFrame()

    if isinstance(bootstrap_result, dict):
        first_val = next(iter(bootstrap_result.values()), None)
        if isinstance(first_val, dict) and "mean" in first_val:
            rows: List[Dict[str, Any]] = []
            for metric_name, vals in bootstrap_result.items():
                if isinstance(vals, dict):
                    rows.append({"metric": metric_name, **vals})
            df_ci = pd.DataFrame(rows)
        else:
            for val in bootstrap_result.values():
                if isinstance(val, list) and len(val) > 0:
                    sharpe_samples = val
                    break
            if sharpe_samples:
                df_samples = pd.DataFrame({"sharpe": sharpe_samples})
                save_csv(df_samples, output_dir / "e8_bootstrap_samples.csv")

    if not df_ci.empty:
        save_csv(df_ci, output_dir / "e8_bootstrap_confidence_intervals.csv")

    return BootstrapResult(sharpe_samples, df_ci, n_samples, random_seed)


@handle_backtest_errors(return_value=None)
def run_e9_monte_carlo(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    E9：蒙特卡洛模拟（P0 整改：迁移到公共 MonteCarloSimulator）。

    基于历史收益率序列，通过有放回重采样模拟未来净值路径分布。
    使用 core.validation.monte_carlo.MonteCarloSimulator 计算，
    删除原手写的 _compute_monte_carlo_stats 私有实现。

    新增（return_paths=True）：
      - 保存完整模拟路径 (n_simulations, n_days+1) 到 e9_monte_carlo_paths.npz
      - 用于路径分布图 / 历史回放
    复用（MonteCarloResult.is_robust）：
      - Sharpe 的 5% 分位数 > 0 判定为稳健
      - 在 e9_monte_carlo_stats.csv 写入 is_robust 列

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        模拟结果 DataFrame，失败返回 None
    """
    logger.info("E9：蒙特卡洛模拟")
    bt_cfg = config["backtest"]
    mc_cfg = get_montecarlo_config(config)
    n_simulations = int(mc_cfg.get("n_simulations", 1000))
    random_seed = int(mc_cfg.get("random_seed", 42))
    bankruptcy_threshold = float(mc_cfg.get("bankruptcy_threshold", 0.8))
    trading_days_per_year = int(
        mc_cfg.get("trading_days_per_year", 252)
    )  # 兼容加密货币 365

    try:
        runner = get_pybroker_runner(
            data_source, config, strategies=get_strategy_names(config)
        )
        result = safe_run_backtest(
            runner, bt_cfg["full_start_date"], bt_cfg["full_end_date"], "E9_base"
        )
        if result is None:
            return None

        eq = result.equity_curve
        if eq is None or eq.empty:
            logger.warning("E9：无净值数据，跳过蒙特卡洛模拟")
            return None

        eq_sorted = eq.sort_values("date")
        returns = eq_sorted["equity"].pct_change().dropna()
        returns = returns[returns.apply(is_valid_number)]

        if len(returns) == 0:
            logger.warning("E9：无有效收益率数据")
            return None

        # P0 整改：调用公共模拟器，return_paths=True 获取完整路径
        simulator = MonteCarloSimulator(
            n_simulations=n_simulations,
            random_seed=random_seed,
            trading_days_per_year=trading_days_per_year,
        )
        mc_result = simulator.simulate(returns.values, return_paths=True)
        if mc_result.n_simulations == 0:
            logger.warning("E9：模拟失败（数据不足）")
            return None

        # 终值 = 路径末端值（沿用旧 E9 字段命名以兼容下游）
        final_values = (
            mc_result.paths[:, -1] if mc_result.paths is not None else None
        )
        # 破产概率（终值 < 阈值）
        bankruptcy_prob = (
            float(np.mean(final_values < bankruptcy_threshold))
            if final_values is not None
            else float("nan")
        )
        # P0 整改：通过 MonteCarloResult.is_robust 判定（Sharpe 5% 分位 > 0）
        is_robust = mc_result.is_robust

        logger.info(f"  模拟次数: {n_simulations}")
        logger.info(
            f"  终值均值: {np.mean(final_values):.4f}, 中位数: {np.median(final_values):.4f}"
        )
        logger.info(f"  破产概率(终值<{bankruptcy_threshold}): {bankruptcy_prob:.2%}")
        logger.info(
            f"  稳健性: {'✅稳健 (Sharpe@5%>0)' if is_robust else '⚠️不稳健 (Sharpe@5%≤0)'}"
        )

        # 兼容旧字段：per-sim final_value & max_drawdown 表格
        # 从 path 重新计算 max_drawdown（与旧版口径一致：每条路径的最大回撤）
        if mc_result.paths is not None:
            peak = np.maximum.accumulate(mc_result.paths, axis=1)
            safe_peak = np.where(peak > 0, peak, 1.0)
            per_sim_max_dd = np.min(mc_result.paths / safe_peak - 1.0, axis=1)
        else:
            per_sim_max_dd = np.zeros(n_simulations)

        mc_results = pd.DataFrame(
            {
                "sim_id": range(n_simulations),
                "final_value": final_values,
                "max_drawdown": per_sim_max_dd,
            }
        )
        save_csv(mc_results, output_dir / "e9_monte_carlo_results.csv")

        # 模拟统计数据（兼容旧版字段 + 新增 is_robust / sharpe_quantiles）
        stats_data: Dict[str, list] = {
            "statistic": ["mean", "median", "std", "p5", "p95"],
            "final_value": [
                float(np.mean(final_values)),
                float(np.median(final_values)),
                float(np.std(final_values)),
                float(np.percentile(final_values, 5)),
                float(np.percentile(final_values, 95)),
            ],
            "max_drawdown": [
                float(np.mean(per_sim_max_dd)),
                float(np.median(per_sim_max_dd)),
                float(np.std(per_sim_max_dd)),
                float(np.percentile(per_sim_max_dd, 5)),
                float(np.percentile(per_sim_max_dd, 95)),
            ],
        }
        # P0 整改：把 MonteCarloResult 公共输出落到 stats csv，便于外部分析
        stats_data["sharpe_quantile_05"] = [mc_result.sharpe_quantiles.get(0.05, 0.0)] * 5
        stats_data["sharpe_quantile_50"] = [mc_result.sharpe_quantiles.get(0.50, 0.0)] * 5
        stats_data["is_robust"] = [is_robust] + [None] * 4  # 仅第一行有意义
        save_csv(pd.DataFrame(stats_data), output_dir / "e9_monte_carlo_stats.csv")

        # 保存完整模拟路径（npz 压缩），供路径分布图 / 历史回放
        if mc_result.paths is not None:
            np.savez_compressed(
                output_dir / "e9_monte_carlo_paths.npz",
                paths=mc_result.paths,
                initial_dates=eq_sorted["date"].iloc[1:].to_numpy(),
                n_simulations=n_simulations,
                random_seed=random_seed,
                trading_days_per_year=trading_days_per_year,
            )
            logger.info(
                f"  路径已保存: e9_monte_carlo_paths.npz "
                f"shape={mc_result.paths.shape}"
            )

        return mc_results
    except Exception as e:
        logger.error(f"  蒙特卡洛模拟失败: {e}")
        return None


@handle_backtest_errors()
def run_e10_html_report(
    config: Dict[str, Any],
    results: Dict[str, Any],
    output_dir: Path,
    optimization_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    E10：生成完整的量化回测分析 HTML 报告。

    直接调用 runner/report/html_report.py 的 generate_html_report，
    删除重复转换代码。

    Args:
        config: 配置字典
        results: 实验结果字典
        output_dir: 输出目录
        optimization_info: 优化信息
    """
    logger.info("E10：生成完整 HTML 分析报告")

    # 直接调用 html_report 模块
    try:
        from runner.report.html_report import generate_html_report

        report_path = generate_html_report(
            config=config,
            results=results,
            output_dir=output_dir,
            optimization_info=optimization_info,
        )

        if report_path:
            logger.info(f"E10：报告已保存至 {report_path}")
        else:
            logger.warning("E10：报告生成失败")
    except Exception as e:
        logger.error(f"E10 报告生成失败: {e}")


def _run_factor_analysis_for_symbol(
    sym: str,
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    factor_names: List[str],
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    对单个品种执行因子分析（E11 子函数，P0 整改）。

    整改前：
      - 使用 RollingICWeightEngine（私有状态对象）做滚动 IC 加权
      - 使用 FactorDecayMonitor（私有状态对象）做衰减检测
      - 访问 decay_monitor._ic_history 私有属性
    整改后：
      - 委托 FactorEvaluator.compute_ic_weights 计算动态权重（公开 API）
      - 委托 FactorEvaluator.detect_decay 检测衰减（公开 API）
      - 通过 factor_scores_history 字典公开维护 IC 历史，规避私有属性

    Args:
        sym: 品种代码
        data_source: 数据源
        config: 配置字典
        factor_names: 因子名称列表
        output_dir: 输出目录

    Returns:
        (ic_df, decay_df, summary) 元组
    """
    # P0 整改：评估器改为单例（无状态），IC 状态由调用方用字典维护
    evaluator = FactorEvaluator()

    ic_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    from core.factors.alpha_futures.sub_strategy_aggregator import (
        compute_sub_strategy_scores_from_ohlcv,
    )

    sym_df = data_source.query(
        data_source.date_range[0], data_source.date_range[1], symbols=[sym]
    )
    if sym_df is None or len(sym_df) < 60:
        logger.warning(f"  {sym}: 数据不足，跳过")
        return pd.DataFrame(), pd.DataFrame(), summary

    scored = compute_sub_strategy_scores_from_ohlcv(sym_df)

    # P0 整改：用公开数据结构维护 IC 历史（替代私有属性访问）
    factor_scores_history: Dict[str, np.ndarray] = {n: np.array([]) for n in factor_names}
    forward_returns_arr: List[float] = []
    prev_weights: Optional[Dict[str, float]] = None
    last_sample_date: Optional[str] = None
    current_ic_snapshot: Dict[str, float] = {}

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

        # 累积每个因子的得分历史
        for name, score in factor_scores.items():
            factor_scores_history[name] = np.append(
                factor_scores_history[name], score
            )
        forward_returns_arr.append(forward_ret)
        last_sample_date = str(row["date"])[:10]

        # 每 10 步采样一次（与旧版节奏一致：避免 csv 过大）
        if i % 10 == 0 and len(forward_returns_arr) >= 20:
            fwd_arr = np.asarray(forward_returns_arr)
            current_weights = evaluator.compute_ic_weights(
                factor_scores_history=factor_scores_history,
                forward_returns=fwd_arr,
                ema_alpha=0.1,
                prev_weights=prev_weights,
            )
            prev_weights = current_weights

            # 当前 IC = 因子得分与 forward_return 的 Pearson（仅取最近 60 步）
            recent_window = min(60, len(forward_returns_arr))
            recent_fwd = fwd_arr[-recent_window:]
            current_ic_snapshot = {}
            for name, scores in factor_scores_history.items():
                if len(scores) < recent_window:
                    continue
                recent_scores = scores[-recent_window:]
                if np.std(recent_scores) < 1e-10 or np.std(recent_fwd) < 1e-10:
                    current_ic_snapshot[name] = 0.0
                else:
                    current_ic_snapshot[name] = float(
                        np.corrcoef(recent_scores, recent_fwd)[0, 1]
                    )

            ic_row: Dict[str, Any] = {"date": last_sample_date}
            for name, ic_val in current_ic_snapshot.items():
                ic_row[f"ic_{name}"] = round(ic_val, 6)
            for name, w in current_weights.items():
                ic_row[f"w_{name}"] = round(w, 4)
            ic_rows.append(ic_row)

    ic_df = pd.DataFrame(ic_rows)

    # P0 整改：使用 FactorEvaluator.detect_decay 公开接口
    # 衰减检测需要的是**历史 IC 序列**，由 factor_scores_history 计算得到
    ic_history_for_decay: Dict[str, List[float]] = {}
    fwd_arr = np.asarray(forward_returns_arr) if forward_returns_arr else np.array([])
    if len(fwd_arr) >= 2:
        for name, scores in factor_scores_history.items():
            if len(scores) < 2 or len(scores) != len(fwd_arr):
                continue
            # 滚动 20 步 IC：每步 Pearson(score[:-1], fwd)
            window = 20
            ic_series: List[float] = []
            for j in range(window, len(scores) + 1):
                s_window = scores[j - window : j]
                f_window = fwd_arr[j - window : j]
                if np.std(s_window) < 1e-10 or np.std(f_window) < 1e-10:
                    continue
                ic_series.append(float(np.corrcoef(s_window, f_window)[0, 1]))
            if ic_series:
                ic_history_for_decay[name] = ic_series

    decay_alerts: Dict[str, Dict[str, Any]] = {}
    if ic_history_for_decay:
        decay_alerts = evaluator.detect_decay(
            ic_history=ic_history_for_decay,
            trend_window=40,
            ic_healthy_threshold=0.03,
            ic_dead_threshold=0.01,
            max_consecutive_decline=5,
            decay_slope_threshold=-0.001,
        )

    # 构造 decay_df（沿用旧版字段格式）
    decay_rows: List[Dict[str, Any]] = []
    decay_status_map: Dict[str, str] = {
        "healthy": "healthy",
        "warning": "warning",
        "decay": "decay",
        "dead": "dead",
    }
    for name in factor_names:
        ic_series = ic_history_for_decay.get(name, [])
        status = decay_alerts.get(name, {}).get("status", "healthy")
        decay_rows.append(
            {
                "date": last_sample_date or "",
                "factor": name,
                "current_ic": round(ic_series[-1], 6) if ic_series else 0.0,
                "mean_ic": round(float(np.mean(ic_series)), 6) if ic_series else 0.0,
                "status": decay_status_map.get(status, "healthy"),
            }
        )
    decay_df = pd.DataFrame(decay_rows)

    # 构造 summary（沿用旧版字段：mean_ic / std_ic / ir / current_ic / current_weight）
    summary_rows: List[Dict[str, Any]] = []
    for name in factor_names:
        ic_series = ic_history_for_decay.get(name, [])
        if ic_series:
            mean_ic = float(np.mean(ic_series))
            std_ic = float(np.std(ic_series))
            ir = mean_ic / std_ic if std_ic > 1e-10 else 0.0
            current_ic = ic_series[-1]
        else:
            mean_ic = std_ic = ir = current_ic = 0.0
        current_weight = float(prev_weights.get(name, 0.0)) if prev_weights else 0.0
        summary_rows.append(
            {
                "symbol": sym,
                "factor": name,
                "mean_ic": round(mean_ic, 6),
                "std_ic": round(std_ic, 6),
                "ir": round(ir, 4),
                "current_ic": round(current_ic, 6),
                "current_weight": round(current_weight, 4),
            }
        )
    summary = {
        "ic_summary": summary_rows,
        "alerts": decay_alerts,
        "final_weights": prev_weights or {},
    }

    return ic_df, decay_df, summary


@handle_backtest_errors(return_value={})
def run_e11_factor_analysis(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    E11：滚动 IC 加权与因子衰减分析。

    对每个品种独立计算因子得分、滚动 IC、动态权重和衰减状态。
    从配置读取 factors 列表，拆分为多个子函数，将绘图移到外部。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        {symbol: {ic_df, decay_df, summary}} 字典
    """
    logger.info("E11：滚动IC加权与因子衰减分析")
    symbols: List[str] = config.get("symbols", [])
    factor_names = get_factors_list(config)
    logger.info(f"  因子列表: {factor_names}")

    all_results: Dict[str, Any] = {}
    all_summary_rows: List[Dict[str, Any]] = []

    for sym in symbols:
        logger.info(f"  分析品种: {sym}")
        try:
            ic_df, decay_df, summary = _run_factor_analysis_for_symbol(
                sym, data_source, config, factor_names, output_dir
            )

            if not ic_df.empty:
                save_csv(
                    ic_df,
                    output_dir
                    / f"e11_ic_{sanitize_filename(sym.replace('.', '_'))}.csv",
                )

            if not decay_df.empty:
                save_csv(
                    decay_df,
                    output_dir
                    / f"e11_decay_{sanitize_filename(sym.replace('.', '_'))}.csv",
                )

            if "ic_summary" in summary:
                all_summary_rows.extend(summary["ic_summary"])

            all_results[sym] = {
                "ic_df": ic_df,
                "decay_df": decay_df,
                "alerts": summary.get("alerts", {}),
                "final_weights": summary.get("final_weights", {}),
            }

            final_weights = summary.get("final_weights", {})
            logger.info(
                f"  {sym}: 最终权重={ ({k: round(v, 4) for k, v in final_weights.items()}) }"
            )
        except Exception as e:
            logger.error(f"  {sym} 因子分析失败: {e}")

    if all_summary_rows:
        summary_df = pd.DataFrame(all_summary_rows)
        save_csv(summary_df, output_dir / "e11_ic_summary.csv")
        logger.info("\n  因子IC汇总:")
        for _, row in summary_df.iterrows():
            logger.info(
                f"    {row['symbol']}/{row['factor']}: "
                f"mean_IC={row['mean_ic']:.4f}, IR={row['ir']:.2f}, "
                f"weight={row['current_weight']:.4f}"
            )

    return all_results
