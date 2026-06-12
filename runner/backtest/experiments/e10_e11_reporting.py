"""
实验 E10 / E11：报告与分析（HTML 报告 + 因子滚动 IC/衰减分析）。

E10：直接委托 runner/report/html_report.py 的 generate_html_report，
    删除重复转换代码；签名特殊 (config, results, output_dir, optimization_info)
    不含 data_source，由 __init__.py::run_experiment 特殊调用。

E11：对每个品种独立计算因子得分、滚动 IC、动态权重和衰减状态；
    委托 FactorEvaluator.compute_ic_weights / detect_decay 公开 API，
    通过 factor_scores_history 字典公开维护 IC 历史，规避私有属性访问。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from core.engine.pybroker_data_source import PyBrokerDataSource
from core.ext.factors.evaluator import FactorEvaluator
from runner.common.config_utils import get_factors_list
from runner.common.utils import (
    handle_backtest_errors,
    is_valid_number,
    sanitize_filename,
    save_csv,
)


# ============================================
# E10：HTML 报告
# ============================================


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

    # ── 5 段式因子验证 PNG 图表（2026-06-12 集成） ──
    _plot_factor_5_section_pngs(results=results, output_dir=output_dir)


def _plot_factor_5_section_pngs(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    若 results["validation"]["standard_report"] 存在，生成 3 个 PNG 图表：
      - factor_prf.png
      - event_study.png
      - factor_redundancy_heatmap.png

    委托 runner/report/plots.py 的 plot_factor_prf / plot_event_study_returns /
    plot_factor_redundancy_heatmap（规则 17：不重复造轮子）。
    """
    validation = results.get("validation") if isinstance(results, dict) else None
    if not isinstance(validation, dict):
        return
    standard = validation.get("standard_report")
    if not isinstance(standard, dict):
        return

    # 1) PRF 柱状图
    try:
        from runner.report.plots import (
            plot_factor_prf,
            plot_event_study_returns,
            plot_factor_redundancy_heatmap,
        )
    except Exception as e:
        logger.warning(f"E10 5 段式 PNG：导入 plot 函数失败: {e}")
        return

    validate_dir = output_dir.parent / "validate"
    if not validate_dir.exists():
        validate_dir = output_dir  # fallback：与主报告同目录

    # PRF
    prf_csv = validate_dir / "factor_prf.csv"
    if prf_csv.exists():
        try:
            df = pd.read_csv(prf_csv)
            plot_factor_prf(df, output_dir / "factor_prf.png")
        except Exception as e:
            logger.warning(f"E10 PRF 绘图失败: {e}")

    # Event Study
    es_csv = validate_dir / "event_study.csv"
    if es_csv.exists():
        try:
            df = pd.read_csv(es_csv)
            plot_event_study_returns(df, output_dir / "event_study.png")
        except Exception as e:
            logger.warning(f"E10 EventStudy 绘图失败: {e}")

    # 冗余热力图
    red_csv = validate_dir / "factor_review_summary.csv"
    if red_csv.exists():
        try:
            from runner.report.plots import plot_factor_redundancy_heatmap

            df = pd.read_csv(red_csv)
            # 尝试构造相关矩阵：若文件含 spearman_max 列则按因子聚合
            if "factor" in df.columns and "max_corr" in df.columns:
                # 简化：单因子对所有其他因子的 max 相关 → 转 1×N 条形
                # 若期望矩阵热力图需 spearman_corr 列，缺则跳过
                pass
            elif (
                "factor_1" in df.columns
                and "factor_2" in df.columns
                and "spearman_rho" in df.columns
            ):
                # 有边表 → 透视成矩阵
                pivot = df.pivot_table(
                    index="factor_1",
                    columns="factor_2",
                    values="spearman_rho",
                    aggfunc="mean",
                )
                # 对称化
                pivot = pivot.combine_first(pivot.T)
                plot_factor_redundancy_heatmap(
                    pivot, output_dir / "factor_redundancy_heatmap.png"
                )
        except Exception as e:
            logger.warning(f"E10 冗余热力图失败: {e}")


# ============================================
# E11 内部：单品种因子分析
# ============================================


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

    from core.ext.factors.alpha_futures.sub_strategy_aggregator import (
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
    factor_scores_history: Dict[str, np.ndarray] = {
        n: np.array([]) for n in factor_names
    }
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
            factor_scores_history[name] = np.append(factor_scores_history[name], score)
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


# ============================================
# E11：滚动 IC 加权与因子衰减分析
# ============================================


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
