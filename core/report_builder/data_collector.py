"""
报告生成模块 — 数据收集。

从输出目录自动扫描策略数据、净值曲线、调仓日志等。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from loguru import logger

from core.config.strategy_profiles import StrategyLibrary
from core.report_builder.metrics import (
    safe_float,
    read_csv,
    read_equity_csv,
    compute_daily_returns,
)


# ---------------------------------------------------------------------------
# 策略数据加载（自动扫描 + 程序化传入）
# ---------------------------------------------------------------------------


def load_strategies_data(
    out_path: Path,
    strategies_data: Optional[Dict[str, Any]],
    out_sample_metrics: Optional[Dict[str, Any]],
    in_sample_dates: Optional[List[str]],
    in_sample_equity: Optional[List[float]],
    out_sample_dates: Optional[List[str]],
    out_sample_equity: Optional[List[float]],
    rebalance_analysis: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """加载策略数据，支持自动扫描和程序化传入两种方式。

    Returns:
        包含所有数据字段的字典。
    """
    data_source_note = ""
    missing_files_note = ""
    diversification_data: Dict[str, Any] = {}

    if strategies_data is None:
        logger.info("报告生成: 从输出目录自动扫描数据...")
        collected = collect_from_directory(out_path)
        strategies_data = collected["strategies"]
        rebalance_analysis = rebalance_analysis or collected.get(
            "rebalance_analysis", {}
        )
        diversification_data = collected.get("diversification_data", {})

        if not strategies_data:
            parent_dir = out_path.parent
            parent_metrics = parent_dir / "all_metrics.csv"
            if parent_metrics.exists():
                logger.info("尝试从父目录扫描标准数据...")
                collected = collect_from_directory(parent_dir)
                strategies_data = collected["strategies"]
                rebalance_analysis = rebalance_analysis or collected.get(
                    "rebalance_analysis", {}
                )

        if not strategies_data:
            logger.warning("标准数据未找到，降级为验证任务文件数据...")
            validation_dir = out_path
            if not list(validation_dir.glob("task1_grid_*.csv")):
                validation_dir = out_path / "validation"
                if not validation_dir.exists() or not list(
                    validation_dir.glob("task1_grid_*.csv")
                ):
                    validation_dir = out_path
            collected = collect_from_validation(validation_dir)
            strategies_data = collected["strategies"]
            data_source_note = "数据来源: 验证任务文件 (task1_grid_*.csv + E7净值曲线)"
            missing = collected.get("missing_files", [])
            if missing:
                missing_files_note = f"缺失文件: {', '.join(missing)}"

        out_sample_metrics = out_sample_metrics or collected.get(
            "out_sample_metrics", {}
        )
        in_sample_dates = in_sample_dates or collected.get("in_sample_dates", [])
        in_sample_equity = in_sample_equity or collected.get("in_sample_equity", [])
        out_sample_dates = out_sample_dates or collected.get("out_sample_dates", [])
        out_sample_equity = out_sample_equity or collected.get("out_sample_equity", [])
    else:
        out_sample_metrics = out_sample_metrics or {}
        in_sample_dates = in_sample_dates or []
        in_sample_equity = in_sample_equity or []
        out_sample_dates = out_sample_dates or []
        out_sample_equity = out_sample_equity or []
        if not in_sample_equity:
            is_d, is_e = read_equity_csv(out_path / "e7_equity_in_sample.csv")
            if is_e:
                in_sample_dates = is_d
                in_sample_equity = is_e
        if not out_sample_equity:
            os_d, os_e = read_equity_csv(out_path / "e7_equity_out_sample.csv")
            if os_e:
                out_sample_dates = os_d
                out_sample_equity = os_e
        if not out_sample_metrics:
            for row in read_csv(out_path / "e7_out_of_sample_metrics.csv"):
                split = row.get("split", "")
                if split:
                    out_sample_metrics[split] = dict(row)

    return {
        "strategies_data": strategies_data,
        "out_sample_metrics": out_sample_metrics,
        "in_sample_dates": in_sample_dates,
        "in_sample_equity": in_sample_equity,
        "out_sample_dates": out_sample_dates,
        "out_sample_equity": out_sample_equity,
        "rebalance_analysis": rebalance_analysis,
        "diversification_data": diversification_data,
        "data_source_note": data_source_note,
        "missing_files_note": missing_files_note,
    }


def _short_label_from_description(description: str) -> str:
    """从 StrategyProfile.description 中提取简短标签。"""
    if not description:
        return ""
    sep = description.find("。")
    return description[:sep] if sep > 0 else description


def _build_strategy_label(name: str, library: StrategyLibrary) -> str:
    """动态构造策略标签。"""
    if not name:
        return ""

    sub = name.split("_", 1)[1] if "_" in name else name
    profile = library.get_profile(sub)
    if profile is not None:
        return _short_label_from_description(profile.description)

    return name


def get_strategy_label(name: str, library: Optional[StrategyLibrary] = None) -> str:
    """对外的薄封装：缺省构造一个 StrategyLibrary。"""
    lib = library or StrategyLibrary()
    return _build_strategy_label(name, lib)


def _analyze_rebalance_decisions(
    switch_log: pd.DataFrame, equity_curve: pd.DataFrame
) -> Dict[str, Any]:
    """分析调仓决策及其后收益表现。"""
    if switch_log.empty or equity_curve.empty:
        return {"decisions": [], "analysis": {}}

    equity_curve = equity_curve.copy()
    if "date" in equity_curve.columns:
        equity_curve["date"] = pd.to_datetime(equity_curve["date"])
        equity_curve = equity_curve.set_index("date")

    decisions = []
    total_decisions = len(switch_log)
    winning_decisions = 0
    total_return = 0.0

    for idx, row in switch_log.iterrows():
        try:
            decision_date = pd.to_datetime(row.get("日期", row.get("timestamp", "")))
            direction = row.get("方向", "")
            composite_score = row.get("综合得分", 0.0)
            position_pct = row.get("仓位比例", 0.0)

            if decision_date in equity_curve.index:
                start_idx = equity_curve.index.get_loc(decision_date)
                end_idx = min(start_idx + 5, len(equity_curve) - 1)

                start_equity = equity_curve.iloc[start_idx]["equity"]
                end_equity = equity_curve.iloc[end_idx]["equity"]

                pct_return = ((end_equity - start_equity) / start_equity) * 100

                is_winning = False
                if direction == "多" and pct_return > 0:
                    is_winning = True
                elif direction == "空" and pct_return < 0:
                    is_winning = True
                elif direction == "平":
                    pass

                if is_winning:
                    winning_decisions += 1

                total_return += pct_return

                decisions.append(
                    {
                        "date": decision_date.strftime("%Y-%m-%d"),
                        "direction": direction,
                        "composite_score": round(float(composite_score), 4),
                        "position_pct": round(float(position_pct), 4),
                        "return_5d": round(pct_return, 2),
                        "winning": is_winning,
                    }
                )
        except Exception as e:
            logger.debug(f"分析调仓决策时出错: {e}")
            continue

    win_rate = (winning_decisions / total_decisions * 100) if total_decisions > 0 else 0
    avg_return = (total_return / total_decisions) if total_decisions > 0 else 0

    analysis = {
        "total_decisions": total_decisions,
        "winning_decisions": winning_decisions,
        "win_rate": round(win_rate, 2),
        "avg_return_5d": round(avg_return, 2),
    }

    return {"decisions": decisions, "analysis": analysis}


def _collect_diversification_data(output_dir: Path) -> dict:
    """收集 E5 多品种分散化数据。"""
    data = {}

    e5_equity_file = output_dir / "e5_multi_symbol_equity.csv"
    if e5_equity_file.exists():
        dates, equity = read_equity_csv(e5_equity_file)
        data["multi_symbol_equity"] = {"dates": dates, "equity": equity}

    single_symbol_equities = {}
    for equity_file in sorted(output_dir.glob("e*_equity_*.csv")):
        stem = equity_file.stem
        parts = stem.replace("_equity_", "|").split("|")
        if len(parts) < 2:
            continue
        symbol = parts[1].replace("_", ".")
        dates, equity = read_equity_csv(equity_file)
        if dates and equity:
            single_symbol_equities[symbol] = {"dates": dates, "equity": equity}
    data["single_symbol_equities"] = single_symbol_equities

    corr_file = output_dir / "e5_correlation_matrix.csv"
    if corr_file.exists():
        try:
            data["correlation_matrix"] = pd.read_csv(corr_file)
        except Exception:
            pass

    for exp in [
        "e1_baseline_metrics.csv",
        "e2_equal_weight_metrics.csv",
        "e3_dynamic_weight_metrics.csv",
    ]:
        file_path_exp = output_dir / exp
        if file_path_exp.exists():
            try:
                df = pd.read_csv(file_path_exp)
                key = exp.replace(".csv", "").replace("e", "E")
                data[f"{key}_metrics"] = df
            except Exception:
                pass

    return data


def collect_from_directory(output_dir: Path) -> Dict[str, Any]:
    """
    从输出目录自动扫描并收集所有策略数据。

    Returns:
        {strategies, out_sample_metrics, in_sample_dates, in_sample_equity,
         out_sample_dates, out_sample_equity, rebalance_analysis, diversification_data}
    """
    metrics_path = output_dir / "all_metrics.csv"
    all_metrics = read_csv(metrics_path)

    strategies = {}
    for row in all_metrics:
        exp = row.get("experiment", "")
        if not exp or exp in strategies:
            continue
        strategies[exp] = {"metrics": dict(row)}

    valid_experiments = {
        "trend",
        "term_structure",
        "mean_reversion",
        "vol_breakout",
        "composite_resonance",
        "cross_sectional",
        "fusion",
    }

    strategy_map: Dict[str, str] = {}

    for equity_file in sorted(output_dir.glob("e*_equity_*.csv")):
        stem = equity_file.stem
        # 提取 _equity_ 之后的部分，取最后一个 _ 后的片段作为策略名
        # e.g. e1_equity_CZCE_CF_trend → strategy = "trend"
        # e.g. e2_equity_fusion → strategy = "fusion"
        if "_equity_" not in stem:
            continue
        suffix = stem.split("_equity_", 1)[1]
        strategy_key = suffix.rsplit("_", 1)[-1] if "_" in suffix else suffix
        if strategy_key not in valid_experiments:
            continue
        dates, equity = read_equity_csv(equity_file)
        if not dates:
            continue
        if strategy_key in strategies:
            strategies[strategy_key]["dates"] = dates
            strategies[strategy_key]["equity"] = equity
        else:
            strategies[strategy_key] = {"metrics": {}, "dates": dates, "equity": equity}

    rebalance_analysis = {}
    for switch_log_file in sorted(output_dir.glob("e*_switch_log_*.csv")):
        try:
            stem = switch_log_file.stem
            parts = stem.replace("_switch_log_", "|").split("|")
            if len(parts) < 2:
                continue

            exp_name = parts[0].upper()
            symbol = parts[1].replace("_", ".")

            switch_log = pd.read_csv(switch_log_file)

            equity_pattern = f"{parts[0]}_equity_{parts[1]}.csv"
            equity_file = output_dir / equity_pattern
            if equity_file.exists():
                equity_curve = pd.read_csv(equity_file)
                analysis = _analyze_rebalance_decisions(switch_log, equity_curve)
                key = f"{exp_name}_{symbol}"
                rebalance_analysis[key] = analysis
        except Exception as e:
            logger.debug(f"收集调仓决策分析时出错: {e}")
            continue

    out_sample = {}
    for oos_file in sorted(output_dir.glob("e7_out_of_sample*.csv")):
        rows = read_csv(oos_file)
        for row in rows:
            split = row.get("split", "")
            if split:
                out_sample[split] = dict(row)

    is_dates, is_eq = read_equity_csv(output_dir / "e7_equity_in_sample.csv")
    os_dates, os_eq = read_equity_csv(output_dir / "e7_equity_out_sample.csv")

    diversification_data = _collect_diversification_data(output_dir)

    return {
        "strategies": strategies,
        "out_sample_metrics": out_sample,
        "in_sample_dates": is_dates,
        "in_sample_equity": is_eq,
        "out_sample_dates": os_dates,
        "out_sample_equity": os_eq,
        "rebalance_analysis": rebalance_analysis,
        "diversification_data": diversification_data,
    }


def collect_from_validation(output_dir: Path) -> Dict[str, Any]:
    """
    从验证目录的任务文件中提取策略数据。

    当标准 all_metrics.csv 和 e1_equity_*.csv 文件不存在时，作为降级数据源。
    """
    strategies = {}
    missing_files = []
    data_source = "validation"

    strategy_map = {
        "trend": "E1_trend",
        "term_structure": "E1_term_structure",
        "mean_reversion": "E1_mean_reversion",
        "vol_breakout": "E1_vol_breakout",
    }

    grid_files = list(output_dir.glob("task1_grid_*.csv"))
    if not grid_files:
        parent_dir = output_dir.parent
        grid_files = list(parent_dir.glob("task1_grid_*.csv"))
        if not grid_files:
            missing_files.append("task1_grid_*.csv")

    for grid_file in grid_files:
        strategy_key = grid_file.stem.replace("task1_grid_", "")
        mapped = strategy_map.get(strategy_key)
        if mapped is None:
            continue
        rows = read_csv(grid_file)
        if not rows:
            continue
        best_row = max(rows, key=lambda r: safe_float(r.get("sharpe", -999)))
        strategies[mapped] = {"metrics": dict(best_row)}

    is_dates, is_equity = read_equity_csv(output_dir / "e7_equity_in_sample.csv")
    if not is_equity:
        is_dates, is_equity = read_equity_csv(
            output_dir.parent / "e7_equity_in_sample.csv"
        )
        if not is_equity:
            missing_files.append("e7_equity_in_sample.csv")

    os_dates, os_equity = read_equity_csv(output_dir / "e7_equity_out_sample.csv")
    if not os_equity:
        os_dates, os_equity = read_equity_csv(
            output_dir.parent / "e7_equity_out_sample.csv"
        )
        if not os_equity:
            missing_files.append("e7_equity_out_sample.csv")

    if is_dates and is_equity:
        for name in strategies:
            strategies[name]["dates"] = list(is_dates)
            strategies[name]["equity"] = list(is_equity)

    out_sample = {}
    oos_metrics_file = output_dir / "e7_out_of_sample_metrics.csv"
    if not oos_metrics_file.exists():
        oos_metrics_file = output_dir.parent / "e7_out_of_sample_metrics.csv"
    if oos_metrics_file.exists():
        for row in read_csv(oos_metrics_file):
            split = row.get("split", "")
            if split:
                out_sample[split] = dict(row)
    else:
        missing_files.append("e7_out_of_sample_metrics.csv")

    validation_data = {}

    wf_file = output_dir / "task1_wf_compare.csv"
    if not wf_file.exists():
        wf_file = output_dir.parent / "task1_wf_compare.csv"
    if wf_file.exists():
        validation_data["wf_compare"] = read_csv(wf_file)
    else:
        missing_files.append("task1_wf_compare.csv")

    mc_file = output_dir / "task3_monte_carlo_summary.csv"
    if not mc_file.exists():
        mc_file = output_dir.parent / "task3_monte_carlo_summary.csv"
    if mc_file.exists():
        validation_data["mc_summary"] = read_csv(mc_file)

    yearly_file = output_dir / "task2_yearly_validation.csv"
    if not yearly_file.exists():
        yearly_file = output_dir.parent / "task2_yearly_validation.csv"
    if yearly_file.exists():
        validation_data["yearly"] = read_csv(yearly_file)

    env_file = output_dir / "task2_env_stats.csv"
    if not env_file.exists():
        env_file = output_dir.parent / "task2_env_stats.csv"
    if env_file.exists():
        validation_data["env_stats"] = read_csv(env_file)

    param_file = output_dir / "task2_param_comparison.csv"
    if not param_file.exists():
        param_file = output_dir.parent / "task2_param_comparison.csv"
    if param_file.exists():
        validation_data["param_comparison"] = read_csv(param_file)

    return {
        "strategies": strategies,
        "out_sample_metrics": out_sample,
        "in_sample_dates": is_dates,
        "in_sample_equity": is_equity,
        "out_sample_dates": os_dates,
        "out_sample_equity": os_equity,
        "validation_data": validation_data,
        "data_source": data_source,
        "missing_files": missing_files,
    }
