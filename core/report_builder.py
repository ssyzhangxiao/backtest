#!/usr/bin/env python3
"""
量化回测报告生成模块

可导入的核心模块，供 run_full_backtest.py E10、run_validation.py 等入口调用。
支持两种输入方式：
  1. 从输出目录自动扫描 CSV 文件（兼容原 generate_report.py）
  2. 程序化传入策略数据和净值曲线（供系统流程调用）

用法示例:
    from core.report_builder import generate_report

    # 方式1: 自动扫描
    generate_report(output_dir="output_backtest_pybroker")

    # 方式2: 程序化传入
    generate_report(
        output_dir="output_backtest_pybroker",
        strategies_data={
            "E1_ts_momentum": {"metrics": {...}, "equity": [...], "dates": [...]},
            ...
        },
    )
"""

import csv
import json
import math
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from loguru import logger

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.02


# ══════════════════════════════════════════════════════════════════════════════
# 数据读取与计算工具
# ══════════════════════════════════════════════════════════════════════════════


def _read_csv(path: Path) -> List[Dict[str, str]]:
    """读取 CSV 文件返回字典列表，失败返回空列表。"""
    try:
        with open(path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        logger.warning(f"读取文件失败: {path}, 错误: {e}")
        return []


def _read_equity_csv(path: Path) -> Tuple[List[str], List[float]]:
    """读取净值 CSV，返回 (日期列表, 净值列表)。失败返回空元组。"""
    rows = _read_csv(path)
    if not rows:
        return [], []
    dates = [r["date"] for r in rows]
    equity = [float(r["equity"]) for r in rows]
    return dates, equity


def _safe_float(val: Any) -> float:
    """安全转 float，委托 runner.common.utils.safe_float。"""
    from runner.common.utils import safe_float

    return safe_float(val)


def _annualized_return(total_return_pct: float, years: float) -> float:
    """计算年化收益率。"""
    if years <= 0:
        return 0.0
    total_factor = 1 + total_return_pct / 100
    if total_factor <= 0:
        return -100.0
    return (total_factor ** (1 / years) - 1) * 100


def _calmar_ratio(ann_return: float, max_dd_pct: float) -> float:
    """计算卡玛比率。"""
    dd = abs(max_dd_pct) if max_dd_pct != 0 else 0.01
    return ann_return / dd


def _compute_drawdown(equity: List[float]) -> List[float]:
    """计算回撤序列（百分比）。"""
    if not equity:
        return []
    peak = equity[0]
    dd = []
    for v in equity:
        if v > peak:
            peak = v
        dd.append((v - peak) / peak * 100 if peak != 0 else 0)
    return dd


def _compute_daily_returns(equity: List[float]) -> List[float]:
    """计算日收益率序列。"""
    rets = []
    for i in range(1, len(equity)):
        if equity[i - 1] != 0:
            rets.append((equity[i] - equity[i - 1]) / equity[i - 1])
        else:
            rets.append(0.0)
    return rets


def _compute_volatility(daily_rets: List[float]) -> float:
    """计算年化波动率。"""
    if len(daily_rets) < 2:
        return 0.0
    mean = sum(daily_rets) / len(daily_rets)
    var = sum((r - mean) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
    return math.sqrt(var) * math.sqrt(TRADING_DAYS_PER_YEAR) * 100


def _compute_sharpe(daily_rets: List[float]) -> float:
    """计算年化夏普比率。"""
    if len(daily_rets) < 2:
        return 0.0
    mean = sum(daily_rets) / len(daily_rets)
    var = sum((r - mean) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
    std = math.sqrt(var) if var > 0 else 1e-10
    ann_mean = mean * TRADING_DAYS_PER_YEAR
    ann_std = std * math.sqrt(TRADING_DAYS_PER_YEAR)
    return (ann_mean - RISK_FREE_RATE) / ann_std if ann_std != 0 else 0


def _rolling_sharpe(daily_rets: List[float], window: int = 36) -> List[float]:
    """计算滚动夏普比率（窗口=36个月交易日≈756天）。"""
    window_days = window * 21
    result = []
    for i in range(window_days, len(daily_rets) + 1):
        slice_rets = daily_rets[i - window_days : i]
        result.append(_compute_sharpe(slice_rets))
    return result


def _rolling_max_drawdown(equity: List[float], window_months: int = 12) -> List[float]:
    """计算滚动最大回撤（窗口=12个月交易日≈252天）。"""
    window_days = window_months * 21
    result = []
    for i in range(window_days, len(equity) + 1):
        slice_eq = equity[i - window_days : i]
        peak = slice_eq[0]
        max_dd = 0.0
        for v in slice_eq:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100 if peak != 0 else 0
            if dd < max_dd:
                max_dd = dd
        result.append(max_dd)
    return result


def _extract_monthly_returns(dates: List[str], equity: List[float]) -> Dict[str, float]:
    """提取月度收益率。"""
    monthly: Dict[str, Dict[str, float]] = {}
    for i, d in enumerate(dates):
        month_key = d[:7]
        if month_key not in monthly:
            monthly[month_key] = {"start_eq": equity[i]}
        monthly[month_key]["end_eq"] = equity[i]

    result = {}
    for k, v in monthly.items():
        if v["start_eq"] != 0:
            result[k] = (v["end_eq"] - v["start_eq"]) / v["start_eq"] * 100
        else:
            result[k] = 0.0
    return result


def _correlation_matrix(
    strategy_data: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[List[float]]]:
    """计算策略间日收益率相关性矩阵。"""
    all_daily_rets = {}
    for name, sd in strategy_data.items():
        eq = sd.get("equity", [])
        all_daily_rets[name] = _compute_daily_returns(eq) if eq else []

    names = sorted(all_daily_rets.keys())
    n = len(names)
    if n == 0:
        return [], []

    corr = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            x = all_daily_rets[names[i]]
            y = all_daily_rets[names[j]]
            min_len = min(len(x), len(y))
            if min_len < 2:
                corr[i][j] = 1.0 if i == j else 0.0
                continue
            xs = x[:min_len]
            ys = y[:min_len]
            mean_x = sum(xs) / min_len
            mean_y = sum(ys) / min_len
            cov = sum((xs[k] - mean_x) * (ys[k] - mean_y) for k in range(min_len))
            std_x = math.sqrt(sum((xk - mean_x) ** 2 for xk in xs))
            std_y = math.sqrt(sum((yk - mean_y) ** 2 for yk in ys))
            if std_x == 0 or std_y == 0:
                corr[i][j] = 0.0
            else:
                corr[i][j] = cov / (std_x * std_y)
    return names, corr


def _calc_pl_ratio(metrics: Dict[str, Any]) -> float:
    """计算盈亏比。"""
    ap = _safe_float(metrics.get("avg_profit_pct", 0))
    al = _safe_float(metrics.get("avg_loss_pct", 1))
    return ap / abs(al) if abs(al) > 0 else 0


# ══════════════════════════════════════════════════════════════════════════════
# 数据收集（自动扫描输出目录）
# ══════════════════════════════════════════════════════════════════════════════


def _analyze_rebalance_decisions(
    switch_log: pd.DataFrame, equity_curve: pd.DataFrame
) -> Dict[str, Any]:
    """
    分析调仓决策及其后收益表现。

    Args:
        switch_log: 调仓决策日志 DataFrame
        equity_curve: 净值曲线 DataFrame

    Returns:
        分析结果字典
    """
    if switch_log.empty or equity_curve.empty:
        return {"decisions": [], "analysis": {}}

    # 确保日期列格式一致
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

            # 寻找决策日期后的净值变化（下一个调仓日或固定窗口）
            if decision_date in equity_curve.index:
                start_idx = equity_curve.index.get_loc(decision_date)
                # 查看之后5个交易日的表现
                end_idx = min(start_idx + 5, len(equity_curve) - 1)

                start_equity = equity_curve.iloc[start_idx]["equity"]
                end_equity = equity_curve.iloc[end_idx]["equity"]

                pct_return = ((end_equity - start_equity) / start_equity) * 100

                # 根据方向判断是否盈利
                is_winning = False
                if direction == "多" and pct_return > 0:
                    is_winning = True
                elif direction == "空" and pct_return < 0:
                    is_winning = True
                elif direction == "平":
                    # 平仓决策，不计算盈亏
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


def collect_from_directory(output_dir: Path) -> Dict[str, Any]:
    """
    从输出目录自动扫描并收集所有策略数据。

    Returns:
        {
            "strategies": {strategy_name: {"metrics": {...}, "dates": [...], "equity": [...]}},
            "out_sample": {split: {...}},
            "rebalance_analysis": 调仓决策分析,
        }
    """
    metrics_path = output_dir / "all_metrics.csv"
    all_metrics = _read_csv(metrics_path)

    strategies = {}
    for row in all_metrics:
        exp = row.get("experiment", "")
        if not exp or exp in strategies:
            continue
        strategies[exp] = {"metrics": dict(row)}

    # 有效的策略实验名称（只有这些才出现在报告中）
    valid_experiments = {
        "E1_ts_momentum",
        "E1_roll_yield",
        "E1_alpha019",
        "E1_alpha032",
        "E2_Fusion",
    }

    # 策略键映射表
    strategy_map = {
        "ts_momentum": "E1_ts_momentum",
        "roll_yield": "E1_roll_yield",
        "alpha019": "E1_alpha019",
        "alpha032": "E1_alpha032",
        "fusion": "E2_Fusion",
    }

    # 扫描所有 e*_equity_*.csv 文件
    for equity_file in sorted(output_dir.glob("e*_equity_*.csv")):
        stem = equity_file.stem
        parts = stem.replace("_equity_", "|").split("|")
        if len(parts) < 2:
            continue
        strategy_key = parts[1]
        mapped = strategy_map.get(strategy_key)
        if mapped is None or mapped not in valid_experiments:
            continue
        dates, equity = _read_equity_csv(equity_file)
        if not dates:
            continue
        if mapped in strategies:
            strategies[mapped]["dates"] = dates
            strategies[mapped]["equity"] = equity
        else:
            # 有净值数据但 all_metrics.csv 中无对应行（如仅跑了E1基线的情况）
            strategies[mapped] = {"metrics": {}, "dates": dates, "equity": equity}

    # 收集调仓决策分析
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

            # 找到对应的净值曲线
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

    # 样本外数据
    out_sample = {}
    for oos_file in sorted(output_dir.glob("e7_out_of_sample*.csv")):
        rows = _read_csv(oos_file)
        for row in rows:
            split = row.get("split", "")
            if split:
                out_sample[split] = dict(row)

    # 样本内外净值
    is_dates, is_eq = _read_equity_csv(output_dir / "e7_equity_in_sample.csv")
    os_dates, os_eq = _read_equity_csv(output_dir / "e7_equity_out_sample.csv")

    return {
        "strategies": strategies,
        "out_sample_metrics": out_sample,
        "in_sample_dates": is_dates,
        "in_sample_equity": is_eq,
        "out_sample_dates": os_dates,
        "out_sample_equity": os_eq,
        "rebalance_analysis": rebalance_analysis,
    }


def collect_from_validation(output_dir: Path) -> Dict[str, Any]:
    """
    从验证目录的任务文件 (task1_*, task2_*, task3_*) 中提取策略数据。

    当标准 all_metrics.csv 和 e1_equity_*.csv 文件不存在时，作为降级数据源。
    数据来源与限制：
      - 策略指标: 取自 task1_grid_*.csv 中夏普最高的行 (网格搜索最优参数)
      - 净值曲线: 取自 e7_equity_in_sample.csv (整体组合净值，非单策略)
      - 样本外指标: 取自 e7_out_of_sample_metrics.csv
      - 验证附加数据: WalkForward对比、蒙特卡洛、年度验证、环境分布

    Returns:
        {strategies, out_sample_metrics, in_sample_dates, in_sample_equity,
         out_sample_dates, out_sample_equity, validation_data, data_source, missing_files}
    """
    strategies = {}
    missing_files = []
    data_source = "validation"

    strategy_map = {
        "ts_momentum": "E1_ts_momentum",
        "roll_yield": "E1_roll_yield",
        "alpha019": "E1_alpha019",
        "alpha032": "E1_alpha032",
    }

    # 从 task1_grid_*.csv 读取策略指标，选择夏普最高的行
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
        rows = _read_csv(grid_file)
        if not rows:
            continue
        best_row = max(rows, key=lambda r: _safe_float(r.get("sharpe", -999)))
        strategies[mapped] = {"metrics": dict(best_row)}

    # 从 E7 文件读取净值曲线
    is_dates, is_equity = _read_equity_csv(output_dir / "e7_equity_in_sample.csv")
    if not is_equity:
        is_dates, is_equity = _read_equity_csv(
            output_dir.parent / "e7_equity_in_sample.csv"
        )
        if not is_equity:
            missing_files.append("e7_equity_in_sample.csv")

    os_dates, os_equity = _read_equity_csv(output_dir / "e7_equity_out_sample.csv")
    if not os_equity:
        os_dates, os_equity = _read_equity_csv(
            output_dir.parent / "e7_equity_out_sample.csv"
        )
        if not os_equity:
            missing_files.append("e7_equity_out_sample.csv")

    # 为每个策略分配净值曲线 (使用样本内净值作为策略净值)
    if is_dates and is_equity:
        for name in strategies:
            strategies[name]["dates"] = list(is_dates)
            strategies[name]["equity"] = list(is_equity)

    # 样本外指标
    out_sample = {}
    oos_metrics_file = output_dir / "e7_out_of_sample_metrics.csv"
    if not oos_metrics_file.exists():
        oos_metrics_file = output_dir.parent / "e7_out_of_sample_metrics.csv"
    if oos_metrics_file.exists():
        for row in _read_csv(oos_metrics_file):
            split = row.get("split", "")
            if split:
                out_sample[split] = dict(row)
    else:
        missing_files.append("e7_out_of_sample_metrics.csv")

    # 验证附加数据
    validation_data = {}

    wf_file = output_dir / "task1_wf_compare.csv"
    if not wf_file.exists():
        wf_file = output_dir.parent / "task1_wf_compare.csv"
    if wf_file.exists():
        validation_data["wf_compare"] = _read_csv(wf_file)
    else:
        missing_files.append("task1_wf_compare.csv")

    mc_file = output_dir / "task3_monte_carlo_summary.csv"
    if not mc_file.exists():
        mc_file = output_dir.parent / "task3_monte_carlo_summary.csv"
    if mc_file.exists():
        validation_data["mc_summary"] = _read_csv(mc_file)

    yearly_file = output_dir / "task2_yearly_validation.csv"
    if not yearly_file.exists():
        yearly_file = output_dir.parent / "task2_yearly_validation.csv"
    if yearly_file.exists():
        validation_data["yearly"] = _read_csv(yearly_file)

    env_file = output_dir / "task2_env_stats.csv"
    if not env_file.exists():
        env_file = output_dir.parent / "task2_env_stats.csv"
    if env_file.exists():
        validation_data["env_stats"] = _read_csv(env_file)

    param_file = output_dir / "task2_param_comparison.csv"
    if not param_file.exists():
        param_file = output_dir.parent / "task2_param_comparison.csv"
    if param_file.exists():
        validation_data["param_comparison"] = _read_csv(param_file)

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


# ══════════════════════════════════════════════════════════════════════════════
# HTML 模板（自包含，无需外部模板文件）
# ══════════════════════════════════════════════════════════════════════════════


def _build_html_report(report_data: Dict[str, Any]) -> str:
    """根据报告数据构建完整 HTML 字符串。"""
    ctx = {**report_data}
    # 预计算 now_str[:4] 方便模板引用
    now_str_val = ctx.get("now_str", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ctx["now_year"] = now_str_val[:4]
    tpl = string.Template(_HTML_TEMPLATE)
    return tpl.substitute(**ctx)


# ══════════════════════════════════════════════════════════════════════════════
# 主入口：生成报告
# ══════════════════════════════════════════════════════════════════════════════


def generate_report(
    output_dir: Optional[str] = None,
    strategies_data: Optional[Dict[str, Any]] = None,
    out_sample_metrics: Optional[Dict[str, Any]] = None,
    in_sample_dates: Optional[List[str]] = None,
    in_sample_equity: Optional[List[float]] = None,
    out_sample_dates: Optional[List[str]] = None,
    out_sample_equity: Optional[List[float]] = None,
    rebalance_analysis: Optional[Dict[str, Any]] = None,
    title: str = "量化回测分析报告",
    subtitle: str = "多策略期货量化交易系统 · 综合绩效评估",
    report_name: str = "backtest_report_full.html",
    include_evaluation: bool = True,
) -> Path:
    """
    生成完整的量化回测分析 HTML 报告。

    Args:
        output_dir: 输出目录路径。若 strategies_data 未提供，则从此目录自动扫描。
        strategies_data: {策略名: {"metrics": {...}, "dates": [...], "equity": [...]}}
        out_sample_metrics: 样本外指标，如 {"in_sample": {...}, "out_sample": {...}}
        in_sample_dates / in_sample_equity: 样本内净值数据
        out_sample_dates / out_sample_equity: 样本外净值数据
        title / subtitle: 报告标题与副标题
        report_name: 输出文件名
        include_evaluation: 是否包含综合评价与改进建议模块

    Returns:
        生成的报告文件 Path
    """
    out_path = Path(output_dir) if output_dir else Path("output_backtest_pybroker")
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 数据来源：自动扫描 or 程序化传入 ──
    data_source_note = ""
    missing_files_note = ""

    if strategies_data is None:
        logger.info("报告生成: 从输出目录自动扫描数据...")
        collected = collect_from_directory(out_path)
        strategies_data = collected["strategies"]
        rebalance_analysis = rebalance_analysis or collected.get(
            "rebalance_analysis", {}
        )

        # 标准扫描无结果，尝试父目录 (如果 out_path 是 validation 子目录)
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

        # 仍无结果，降级为验证任务文件数据
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
        # 即使程序化传入策略数据，也尝试从目录读取E7净值曲线和样本外指标
        if not in_sample_equity:
            is_d, is_e = _read_equity_csv(out_path / "e7_equity_in_sample.csv")
            if is_e:
                in_sample_dates = is_d
                in_sample_equity = is_e
        if not out_sample_equity:
            os_d, os_e = _read_equity_csv(out_path / "e7_equity_out_sample.csv")
            if os_e:
                out_sample_dates = os_d
                out_sample_equity = os_e
        if not out_sample_metrics:
            for row in _read_csv(out_path / "e7_out_of_sample_metrics.csv"):
                split = row.get("split", "")
                if split:
                    out_sample_metrics[split] = dict(row)

    if not strategies_data:
        logger.error("报告生成: 无可用策略数据，中止")
        return out_path / report_name

    strategy_names = list(strategies_data.keys())
    logger.info(f"报告生成: 发现 {len(strategy_names)} 个策略: {strategy_names}")

    # ── 确定回测区间 ──
    first_strategy = strategies_data[strategy_names[0]]
    all_dates = first_strategy.get("dates", [])
    if all_dates:
        backtest_start = all_dates[0]
        backtest_end = all_dates[-1]
        total_days = len(all_dates)
    else:
        backtest_start = "N/A"
        backtest_end = "N/A"
        total_days = 0
    total_years = total_days / TRADING_DAYS_PER_YEAR if total_days > 0 else 0

    # ── 为每个策略计算衍生指标 ──
    strategy_labels = {
        "E1_ts_momentum": "E1 时序动量",
        "E1_roll_yield": "E1 展期收益",
        "E1_alpha019": "E1 Alpha019",
        "E1_alpha032": "E1 Alpha032",
        "E2_Fusion": "E2 融合策略",
        "E4_Switching": "E4 策略切换",
    }

    for name, sd in strategies_data.items():
        metrics = sd.get("metrics", {})
        eq = sd.get("equity", [])
        dates = sd.get("dates", [])

        n_days = len(dates) if dates else total_days
        years_exp = n_days / TRADING_DAYS_PER_YEAR if n_days > 0 else total_years

        total_ret = _safe_float(metrics.get("total_return_pct", 0))
        ann_ret = _annualized_return(total_ret, years_exp)
        max_dd = _safe_float(metrics.get("max_drawdown_pct", 0))
        calmar = _calmar_ratio(ann_ret, max_dd)

        metrics["ann_return"] = ann_ret
        metrics["calmar"] = calmar
        metrics["total_years"] = years_exp
        sd["label"] = strategy_labels.get(name, name)

    # ── 选最佳策略（正收益优先） ──
    best_exp = strategy_names[0]
    for name in strategy_names:
        if (
            _safe_float(
                strategies_data[name].get("metrics", {}).get("total_return_pct", 0)
            )
            > 0
        ):
            best_exp = name
            break

    best = strategies_data[best_exp]
    best_metrics = best.get("metrics", {})
    best_equity = best.get("equity", [])
    best_dates = best.get("dates", [])
    best_label = strategy_labels.get(best_exp, best_exp)

    # ── KPI 卡片 HTML ──
    kpi_items = [
        (
            "年化收益率",
            f"{best_metrics.get('ann_return', 0):+.2f}%",
            best_metrics.get("ann_return", 0),
        ),
        (
            "夏普比率",
            f"{_safe_float(best_metrics.get('sharpe', 0)):.4f}",
            _safe_float(best_metrics.get("sharpe", 0)),
        ),
        (
            "最大回撤",
            f"{_safe_float(best_metrics.get('max_drawdown_pct', 0)):.2f}%",
            _safe_float(best_metrics.get("max_drawdown_pct", 0)),
        ),
        (
            "卡玛比率",
            f"{best_metrics.get('calmar', 0):.4f}",
            best_metrics.get("calmar", 0),
        ),
        (
            "胜率",
            f"{_safe_float(best_metrics.get('win_rate', 0)):.1f}%",
            _safe_float(best_metrics.get("win_rate", 0)),
        ),
        ("盈亏比", f"{_calc_pl_ratio(best_metrics):.2f}", _calc_pl_ratio(best_metrics)),
        (
            "总交易次数",
            f"{int(_safe_float(best_metrics.get('trade_count', 0)))}",
            _safe_float(best_metrics.get("trade_count", 0)),
        ),
    ]
    kpi_cards_html = "\n".join(
        _build_kpi_card(label, val, num) for label, val, num in kpi_items
    )

    # ── 策略对比表格 ──
    strategy_table_rows = ""
    for name in strategy_names:
        sd = strategies_data[name]
        m = sd.get("metrics", {})
        tr = _safe_float(m.get("total_return_pct", 0))
        ar = m.get("ann_return", 0)
        sh = _safe_float(m.get("sharpe", 0))
        dd = _safe_float(m.get("max_drawdown_pct", 0))
        ca = m.get("calmar", 0)
        wr = _safe_float(m.get("win_rate", 0))
        tc = int(_safe_float(m.get("trade_count", 0)))
        pl = _calc_pl_ratio(m)

        def _cls(v, t=0):
            if v > t:
                return "positive"
            elif v < -t:
                return "negative"
            return ""

        strategy_table_rows += f"""
        <tr>
            <td><strong>{sd["label"]}</strong></td>
            <td class="{_cls(tr)}">{tr:+.2f}%</td>
            <td class="{_cls(ar)}">{ar:+.2f}%</td>
            <td class="{_cls(sh, 0.1)}">{sh:.4f}</td>
            <td class="{"negative" if dd < -0.5 else ""}">{dd:.2f}%</td>
            <td class="{_cls(ca, 0.1)}">{ca:.4f}</td>
            <td>{wr:.1f}%</td>
            <td>{pl:.2f}</td>
            <td>{tc}</td>
        </tr>"""

    # ── 样本内/外表格 ──
    oos_table_rows = ""
    for split_name in ["in_sample", "out_sample"]:
        if split_name in out_sample_metrics:
            m = out_sample_metrics[split_name]
            tr = _safe_float(m.get("total_return_pct", 0))
            years_this = 5 if split_name == "in_sample" else max(total_years - 5, 0.5)
            ar = _annualized_return(tr, years_this)
            sh = _safe_float(m.get("sharpe", 0))
            dd = _safe_float(m.get("max_drawdown_pct", 0))
            ca = _calmar_ratio(ar, dd)
            wr = _safe_float(m.get("win_rate", 0))
            tc = int(_safe_float(m.get("trade_count", 0)))
            split_label = (
                "样本内 (2016-2020)"
                if split_name == "in_sample"
                else "样本外 (2021-2025)"
            )
            oos_table_rows += f"""
            <tr>
                <td><strong>{split_label}</strong></td>
                <td class="{"positive" if tr > 0 else "negative"}">{tr:+.2f}%</td>
                <td class="{"positive" if ar > 0 else "negative"}">{ar:+.2f}%</td>
                <td class="{"positive" if sh > 0.1 else ""}">{sh:.4f}</td>
                <td class="{"negative" if dd < -0.5 else ""}">{dd:.2f}%</td>
                <td class="{"positive" if ca > 0.1 else ""}">{ca:.4f}</td>
                <td>{wr:.1f}%</td>
                <td>{tc}</td>
            </tr>"""

    # ── 图表数据 ──
    equity_js = {}
    for name in strategy_names:
        sd = strategies_data[name]
        eq = sd.get("equity", [])
        dates = sd.get("dates", [])
        first_eq = eq[0] if eq else 1
        equity_js[name] = {
            "dates": dates,
            "equity": [e / first_eq for e in eq] if eq else [],
            "label": sd.get("label", name),
        }

    # 所有策略回撤计算
    all_drawdowns = {}
    for name, sd in strategies_data.items():
        eq = sd.get("equity", [])
        dates = sd.get("dates", [])
        if eq:
            dd_seq = _compute_drawdown(eq)
            peak = eq[0]
            max_dd_val = 0.0
            max_dd_idx = 0
            max_dd_start_idx = 0
            current_peak_idx = 0
            for i, v in enumerate(eq):
                if v > peak:
                    peak = v
                    current_peak_idx = i
                dd_val = (v - peak) / peak * 100 if peak != 0 else 0
                if dd_val < max_dd_val:
                    max_dd_val = dd_val
                    max_dd_idx = i
                    max_dd_start_idx = current_peak_idx
            duration_days = (
                max_dd_idx - max_dd_start_idx if max_dd_idx > max_dd_start_idx else 0
            )
            all_drawdowns[name] = {
                "dates": dates,
                "drawdown": dd_seq,
                "max_dd_pct": round(max_dd_val, 2),
                "max_dd_date": dates[max_dd_idx] if max_dd_idx < len(dates) else "",
                "max_dd_start_date": dates[max_dd_start_idx]
                if max_dd_start_idx < len(dates)
                else "",
                "duration_days": duration_days,
            }

    # 主策略回撤
    main_dd = _compute_drawdown(best_equity)
    main_daily_rets = _compute_daily_returns(best_equity)

    # 所有策略滚动指标
    all_rolling_sharpe = {}
    all_rolling_dd = {}
    for name, sd in strategies_data.items():
        eq = sd.get("equity", [])
        dates = sd.get("dates", [])
        if eq:
            daily_rets = _compute_daily_returns(eq)
            rs_vals = _rolling_sharpe(daily_rets, window=36)
            rd_vals = _rolling_max_drawdown(eq, window_months=12)
            all_rolling_sharpe[name] = {
                "dates": dates[36 * 21 :] if len(dates) >= 36 * 21 else [],
                "values": rs_vals,
            }
            all_rolling_dd[name] = {
                "dates": dates[12 * 21 :] if len(dates) >= 12 * 21 else [],
                "values": rd_vals,
            }

    # 滚动指标（保留best策略用于原有图表）
    rolling_sharpe_vals = _rolling_sharpe(main_daily_rets, window=36)
    rolling_dd_vals = _rolling_max_drawdown(best_equity, window_months=12)

    # 月度收益率
    monthly_returns = _extract_monthly_returns(best_dates, best_equity)
    months_sorted = sorted(monthly_returns.keys())
    years_set = sorted(set(k[:4] for k in months_sorted))
    months_labels = [f"{m:02d}" for m in range(1, 13)]
    heatmap_data = [[None] * 12 for _ in range(len(years_set))]
    for yi, year in enumerate(years_set):
        for mi, month in enumerate(months_labels):
            key = f"{year}-{month}"
            if key in monthly_returns:
                heatmap_data[yi][mi] = round(monthly_returns[key], 2)

    # 所有策略月度收益率热力图
    all_heatmaps = {}
    for name, sd in strategies_data.items():
        eq = sd.get("equity", [])
        dates = sd.get("dates", [])
        if eq and dates:
            mr = _extract_monthly_returns(dates, eq)
            ms = sorted(mr.keys())
            ys = sorted(set(k[:4] for k in ms))
            hd = [[None] * 12 for _ in range(len(ys))]
            for yi2, year in enumerate(ys):
                for mi2, month in enumerate(months_labels):
                    key = f"{year}-{month}"
                    if key in mr:
                        hd[yi2][mi2] = round(mr[key], 2)
            all_heatmaps[name] = {"data": hd, "years_set": ys}

    # 风险收益散点
    risk_return = []
    for name in strategy_names:
        sd = strategies_data[name]
        eq = sd.get("equity", [])
        rets = _compute_daily_returns(eq)
        ann_ret = (sum(rets) / len(rets)) * TRADING_DAYS_PER_YEAR * 100 if rets else 0
        ann_vol = _compute_volatility(rets)
        risk_return.append(
            {
                "name": sd.get("label", name),
                "key": name,
                "ann_return": round(ann_ret, 2),
                "ann_volatility": round(ann_vol, 2),
                "sharpe": _safe_float(sd.get("metrics", {}).get("sharpe", 0)),
            }
        )

    # 相关性矩阵
    corr_names, corr_matrix = _correlation_matrix(strategies_data)

    # 样本内/外净值 JS
    in_sample_js = {}
    if in_sample_equity:
        first_is = in_sample_equity[0] if in_sample_equity[0] != 0 else 1
        in_sample_js = {
            "dates": in_sample_dates,
            "equity": [e / first_is for e in in_sample_equity],
        }
    out_sample_js = {}
    if out_sample_equity:
        first_os = out_sample_equity[0] if out_sample_equity[0] != 0 else 1
        out_sample_js = {
            "dates": out_sample_dates,
            "equity": [e / first_os for e in out_sample_equity],
        }

    # 所有策略样本内/外净值（按日期拆分）
    all_is_equity = {}
    all_os_equity = {}
    is_start_date = in_sample_dates[0] if in_sample_dates else ""
    is_end_date = in_sample_dates[-1] if in_sample_dates else ""
    os_start_date = out_sample_dates[0] if out_sample_dates else ""
    for name, sd in strategies_data.items():
        eq = sd.get("equity", [])
        dates = sd.get("dates", [])
        if eq and dates:
            is_eq, is_dt = [], []
            os_eq, os_dt = [], []
            for i, d in enumerate(dates):
                if i < len(eq):
                    if in_sample_dates and d <= is_end_date:
                        is_eq.append(eq[i])
                        is_dt.append(d)
                    if out_sample_dates and d >= os_start_date:
                        os_eq.append(eq[i])
                        os_dt.append(d)
            if is_eq:
                first_is_v = is_eq[0] if is_eq[0] != 0 else 1
                all_is_equity[name] = {
                    "dates": is_dt,
                    "equity": [e / first_is_v for e in is_eq],
                }
            if os_eq:
                first_os_v = os_eq[0] if os_eq[0] != 0 else 1
                all_os_equity[name] = {
                    "dates": os_dt,
                    "equity": [e / first_os_v for e in os_eq],
                }

    # 所有策略样本内/外绩效对比表
    strategy_labels_map = {
        "E1_ts_momentum": "时序动量",
        "E1_roll_yield": "展期收益",
        "E1_alpha019": "Alpha019",
        "E1_alpha032": "Alpha032",
        "E2_Fusion": "融合策略",
        "E4_Switching": "策略切换",
    }
    strategy_oos_rows = ""
    for name in strategy_names:
        sd = strategies_data[name]
        eq = sd.get("equity", [])
        dates = sd.get("dates", [])
        label = strategy_labels_map.get(name, name)
        if not eq or not dates or not in_sample_dates or not out_sample_dates:
            continue
        for split_label, split_eq, split_dates in [
            (
                "样本内",
                [
                    eq[i]
                    for i, d in enumerate(dates)
                    if i < len(eq) and d <= is_end_date
                ],
                [d for d in dates if d <= is_end_date],
            ),
            (
                "样本外",
                [
                    eq[i]
                    for i, d in enumerate(dates)
                    if i < len(eq) and d >= os_start_date
                ],
                [d for d in dates if d >= os_start_date],
            ),
        ]:
            if len(split_eq) < 10:
                continue
            split_rets = _compute_daily_returns(split_eq)
            tr = (
                (split_eq[-1] - split_eq[0]) / split_eq[0] * 100
                if split_eq[0] != 0
                else 0
            )
            years_split = len(split_eq) / TRADING_DAYS_PER_YEAR
            ar = _annualized_return(tr, years_split) if years_split > 0 else 0
            sh = _compute_sharpe(split_rets)
            dd_seq = _compute_drawdown(split_eq)
            dd = min(dd_seq) if dd_seq else 0
            ca = _calmar_ratio(ar, dd)
            wr = (
                sum(1 for r in split_rets if r > 0) / len(split_rets) * 100
                if split_rets
                else 0
            )
            tc = len([1 for r in split_rets if abs(r) > 0.001])
            strategy_oos_rows += f"""
            <tr>
                <td><strong>{label}</strong></td>
                <td>{split_label}</td>
                <td class="{"positive" if tr > 0 else "negative"}">{tr:+.2f}%</td>
                <td class="{"positive" if ar > 0 else "negative"}">{ar:+.2f}%</td>
                <td class="{"positive" if sh > 0.1 else ""}">{sh:.4f}</td>
                <td class="{"negative" if dd < -0.5 else ""}">{dd:.2f}%</td>
                <td class="{"positive" if ca > 0.1 else ""}">{ca:.4f}</td>
                <td>{wr:.1f}%</td>
                <td>{tc}</td>
            </tr>"""

    # 组装 chart_data
    chart_data = {
        "equity_curves": equity_js,
        "main_drawdown": {
            "dates": best_dates,
            "drawdown": main_dd,
        },
        "all_drawdowns": all_drawdowns,
        "risk_return": risk_return,
        "heatmap_data": heatmap_data,
        "all_heatmaps": all_heatmaps,
        "years_set": years_set,
        "rolling_sharpe": {
            "dates": best_dates[36 * 21 :] if len(best_dates) >= 36 * 21 else [],
            "values": rolling_sharpe_vals,
        },
        "all_rolling_sharpe": all_rolling_sharpe,
        "rolling_dd": {
            "dates": best_dates[12 * 21 :] if len(best_dates) >= 12 * 21 else [],
            "values": rolling_dd_vals,
        },
        "all_rolling_dd": all_rolling_dd,
        "all_is_equity": all_is_equity,
        "all_os_equity": all_os_equity,
        "correlation": {
            "names": corr_names,
            "matrix": corr_matrix,
        },
    }

    # ── 调仓决策分析 ──
    rebalance_html = ""
    if rebalance_analysis:
        rebalance_sections = []
        for key, analysis_data in rebalance_analysis.items():
            decisions = analysis_data.get("decisions", [])
            analysis = analysis_data.get("analysis", {})

            if not decisions:
                continue

            total_decisions = analysis.get("total_decisions", 0)
            winning_decisions = analysis.get("winning_decisions", 0)
            win_rate = analysis.get("win_rate", 0)
            avg_return = analysis.get("avg_return_5d", 0)

            decision_rows = []
            for d in decisions[:30]:  # 最多显示30条记录
                direction_class = ""
                if d["direction"] == "多":
                    direction_class = "positive"
                elif d["direction"] == "空":
                    direction_class = "negative"

                winning_class = "positive" if d.get("winning") else "negative"

                decision_rows.append(f"""
                <tr>
                    <td>{d["date"]}</td>
                    <td class="{direction_class}">{d["direction"]}</td>
                    <td>{d["composite_score"]}</td>
                    <td>{d["position_pct"]}</td>
                    <td class="{winning_class}">{d["return_5d"]}%</td>
                </tr>""")

            rebalance_sections.append(f"""
            <div class="section-title">🔄 调仓决策分析 - {key}</div>
            <div class="two-col">
                {_build_kpi_card("总决策次数", str(total_decisions), total_decisions)}
                {_build_kpi_card("盈利决策", str(winning_decisions), winning_decisions)}
                {_build_kpi_card("决策胜率", f"{win_rate}%", win_rate)}
                {_build_kpi_card("平均5日收益", f"{avg_return}%", avg_return)}
            </div>
            <div class="table-box">
                <table class="data-table">
                    <thead>
                        <tr><th>日期</th><th>方向</th><th>综合得分</th><th>仓位比例</th><th>5日收益</th></tr>
                    </thead>
                    <tbody>{"".join(decision_rows)}</tbody>
                </table>
            </div>""")

        if rebalance_sections:
            rebalance_html = "\n".join(rebalance_sections)

    # ── 综合评价 ──
    evaluation_html = _DEFAULT_EVALUATION_HTML if include_evaluation else ""

    # ── 构建报告上下文 ──
    report_ctx = {
        "backtest_start": backtest_start,
        "backtest_end": backtest_end,
        "total_days": total_days,
        "total_years": f"{total_years:.1f}",
        "init_capital": "1,000,000",
        "now_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "title": title,
        "subtitle": subtitle,
        "symbols_str": "SHFE.RB, DCE.M, CZCE.TA, SHFE.CU, CFFEX.IF",
        "best_strategy_label": best_label,
        "kpi_cards_html": kpi_cards_html,
        "strategy_table_html": strategy_table_rows,
        "oos_table_html": oos_table_rows,
        "strategy_oos_html": strategy_oos_rows,
        "chart_data_json": json.dumps(chart_data, ensure_ascii=False),
        "rebalance_html": rebalance_html,
        "evaluation_html": evaluation_html,
        "in_sample_js": json.dumps(in_sample_js, ensure_ascii=False),
        "out_sample_js": json.dumps(out_sample_js, ensure_ascii=False),
        "data_source_note": data_source_note,
        "missing_files_note": missing_files_note,
    }

    # ── 数据来源警告 HTML ──
    data_source_warning_html = ""
    if data_source_note or missing_files_note:
        parts = []
        if data_source_note:
            parts.append(
                f'<div class="meta-item"><span class="label">&#9888; 数据来源</span><span class="value" style="color:#f59e0b;">{data_source_note}</span></div>'
            )
        if missing_files_note:
            parts.append(
                f'<div class="meta-item"><span class="label">&#10060; 缺失</span><span class="value" style="color:#ef4444;">{missing_files_note}</span></div>'
            )
        if parts:
            data_source_warning_html = (
                '<div class="meta-row" style="margin-top:12px; background:rgba(245,158,11,0.1); padding:10px 14px; border-radius:8px; border:1px solid rgba(245,158,11,0.3);">'
                + "\n".join(parts)
                + "</div>"
            )
    report_ctx["data_source_warning_html"] = data_source_warning_html

    html = _build_html_report(report_ctx)
    report_path = out_path / report_name
    report_path.write_text(html, encoding="utf-8")
    logger.info(
        f"报告已生成: {report_path} ({report_path.stat().st_size / 1024:.1f} KB)"
    )
    return report_path


def _build_kpi_card(label: str, value: str, numeric_val: float) -> str:
    """构建 KPI 卡片 HTML。"""
    cls = ""
    if numeric_val > 0:
        cls = "positive"
    elif numeric_val < 0:
        cls = "negative"
    return f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {cls}">{value}</div>
        </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# CSS & JS 常量
# ══════════════════════════════════════════════════════════════════════════════

_CSS_STYLE = """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background: #f0f2f5; color: #1a1a2e; line-height: 1.6;
        }
        .container { max-width: 1320px; margin: 0 auto; padding: 20px; }
        .report-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white; padding: 36px 40px; border-radius: 16px; margin-bottom: 24px;
            box-shadow: 0 4px 24px rgba(15,52,96,0.3);
        }
        .report-header h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; letter-spacing: 1px; }
        .report-header .subtitle { font-size: 14px; color: #a8b2d1; margin-bottom: 16px; }
        .report-header .meta-row { display: flex; flex-wrap: wrap; gap: 20px; font-size: 13px; color: #8892b0; }
        .report-header .meta-item {
            display: flex; align-items: center; gap: 6px;
            background: rgba(255,255,255,0.08); padding: 6px 14px; border-radius: 20px;
        }
        .report-header .meta-item .label { color: #8892b0; }
        .report-header .meta-item .value { color: #ccd6f6; font-weight: 600; }
        .date-range-badge {
            display: inline-block; background: linear-gradient(135deg, #0f3460, #1a1a2e);
            color: #64ffda; padding: 8px 20px; border-radius: 20px;
            font-size: 15px; font-weight: 600; letter-spacing: 0.5px; margin-top: 12px;
            border: 1px solid rgba(100,255,218,0.3);
        }
        .section-title {
            font-size: 20px; font-weight: 700; color: #1a1a2e; margin: 32px 0 16px;
            padding-left: 16px; border-left: 4px solid #0f3460;
        }
        .section-desc { font-size: 13px; color: #666; margin-bottom: 16px; padding-left: 20px; }
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .kpi-card {
            background: white; border-radius: 12px; padding: 20px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06); text-align: center;
            transition: transform 0.2s, box-shadow 0.2s; border: 1px solid #e8ecf1;
        }
        .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.1); }
        .kpi-card .kpi-label {
            font-size: 12px; color: #8892b0; text-transform: uppercase;
            letter-spacing: 0.5px; margin-bottom: 6px; font-weight: 500;
        }
        .kpi-card .kpi-value { font-size: 26px; font-weight: 700; color: #1a1a2e; }
        .kpi-card .kpi-value.positive { color: #059669; }
        .kpi-card .kpi-value.negative { color: #dc2626; }
        .card {
            background: white; border-radius: 14px; padding: 24px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 20px;
            border: 1px solid #e8ecf1;
        }
        .card-header {
            font-size: 16px; font-weight: 600; color: #1a1a2e; margin-bottom: 16px;
            display: flex; align-items: center; gap: 8px;
        }
        .card-header::before {
            content: ''; display: inline-block; width: 8px; height: 8px;
            border-radius: 50%; background: #0f3460;
        }
        .table-wrapper { overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th {
            background: #f8f9fb; color: #475569; font-weight: 600;
            padding: 12px 14px; text-align: center; border-bottom: 2px solid #e2e8f0;
            font-size: 12px; letter-spacing: 0.3px; white-space: nowrap;
        }
        td { padding: 11px 14px; text-align: center; border-bottom: 1px solid #f1f5f9; white-space: nowrap; }
        tr:hover td { background: #f8fafc; }
        .positive { color: #059669; font-weight: 600; }
        .negative { color: #dc2626; font-weight: 600; }
        .chart-box {
            background: white; border-radius: 14px; padding: 24px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 20px;
            border: 1px solid #e8ecf1;
        }
        .chart-box canvas { max-height: 420px; }
        .heatmap-wrapper { overflow-x: auto; }
        .heatmap-table { border-collapse: collapse; font-size: 11px; }
        .heatmap-table th { padding: 6px 8px; text-align: center; background: #f8f9fb; color: #475569; font-weight: 600; font-size: 11px; }
        .heatmap-table td { padding: 5px 7px; text-align: center; font-size: 11px; border: 1px solid #e2e8f0; }
        .eval-problem { margin-bottom: 18px; padding-left: 16px; border-left: 3px solid #e2e8f0; }
        .eval-problem-title { font-weight: 700; font-size: 14px; color: #1a1a2e; margin-bottom: 6px; }
        .eval-problem p, .eval-problem ul { font-size: 13px; color: #475569; line-height: 1.7; }
        .eval-problem ul { padding-left: 20px; }
        .eval-problem li { margin-bottom: 4px; }
        .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
        .badge-danger { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
        .badge-warning { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
        .suggestion-list { padding-left: 20px; font-size: 13px; color: #475569; line-height: 1.8; }
        .suggestion-list li { margin-bottom: 8px; }
        .footer { text-align: center; color: #94a3b8; font-size: 12px; margin-top: 40px; padding: 20px; }
        .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .table-box {
            background: white; border-radius: 14px; padding: 24px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 20px;
            border: 1px solid #e8ecf1; overflow-x: auto;
        }
        .data-table {
            width: 100%; border-collapse: collapse; font-size: 13px;
        }
        .data-table th {
            padding: 12px 16px; text-align: left; background: #f8f9fb;
            color: #475569; font-weight: 600; border-bottom: 2px solid #e2e8f0;
        }
        .data-table td {
            padding: 10px 16px; border-bottom: 1px solid #f1f5f9;
        }
        .data-table tbody tr:hover {
            background: #f8fafc;
        }
        @media (max-width: 900px) { .two-col { grid-template-columns: 1fr; } }
        @media (max-width: 768px) { .container { padding: 10px; } .report-header { padding: 24px; } .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
"""

_HTML_TEMPLATE = (
    """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>$title</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
    <style>
"""
    + _CSS_STYLE
    + """
    </style>
</head>
<body>
<div class="container">

    <div class="report-header">
        <h1>&#x1f4ca; $title</h1>
        <div class="subtitle">$subtitle</div>
        <div class="date-range-badge">回测区间: $backtest_start ~ $backtest_end</div>
        <div class="meta-row" style="margin-top: 16px;">
            <div class="meta-item"><span class="label">生成时间</span><span class="value">$now_str</span></div>
            <div class="meta-item"><span class="label">回测引擎</span><span class="value">PyBroker</span></div>
            <div class="meta-item"><span class="label">初始资金</span><span class="value">&#165;$init_capital</span></div>
            <div class="meta-item"><span class="label">交易品种</span><span class="value">$symbols_str</span></div>
            <div class="meta-item"><span class="label">数据周期</span><span class="value">日线</span></div>
            <div class="meta-item"><span class="label">总交易日</span><span class="value">$total_days 天 ($total_years 年)</span></div>
        </div>
        $data_source_warning_html
    </div>

    <div class="section-title">全局绩效指标</div>
    <div class="section-desc">以下为最佳策略组合（$best_strategy_label）的核心绩效指标概览</div>
    <div class="kpi-grid">
        $kpi_cards_html
    </div>

    <div class="card">
        <div class="card-header">策略绩效对比</div>
        <div class="section-desc" style="margin-top:-8px;">对比各策略在全回测区间的关键绩效指标（正收益绿、负收益红）</div>
        <div class="table-wrapper">
        <table>
            <thead>
            <tr>
                <th>策略名称</th><th>总收益率</th><th>年化收益率</th><th>夏普比率</th>
                <th>最大回撤</th><th>卡玛比率</th><th>胜率</th><th>盈亏比</th><th>交易次数</th>
            </tr>
            </thead>
            <tbody>$strategy_table_html</tbody>
        </table>
        </div>
    </div>

    <div class="card">
        <div class="card-header">所有策略样本内/外绩效对比</div>
        <div class="section-desc" style="margin-top:-8px;">各策略在样本内与样本外的独立表现对比，评估各策略泛化能力</div>
        <div class="table-wrapper">
        <table>
            <thead>
            <tr>
                <th>策略</th><th>数据集</th><th>总收益率</th><th>年化收益率</th><th>夏普比率</th>
                <th>最大回撤</th><th>卡玛比率</th><th>胜率</th><th>交易次数</th>
            </tr>
            </thead>
            <tbody>$strategy_oos_html</tbody>
        </table>
        </div>
    </div>

    <div class="section-title">可视化图表</div>
    <div class="section-desc">基于 Chart.js 的交互式图表，支持缩放、悬停提示</div>

    <div class="chart-box"><div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f4c8; 净值曲线对比（归一化，从1开始）</div><canvas id="chartEquity"></canvas></div>
    <div class="chart-box"><div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f4ca; 所有策略回撤对比</div><canvas id="chartAllDrawdowns"></canvas></div>

    <div class="two-col">
        <div class="chart-box"><div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f539; 风险收益散点图</div><canvas id="chartScatter"></canvas></div>
        <div class="chart-box"><div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f321;&#xfe0f; 月度收益率热力图（$best_strategy_label）</div><canvas id="chartHeatmap"></canvas></div>
    </div>
    <div id="allHeatmapsContainer"></div>

    <div class="section-title">风险分析</div>
    <div class="two-col">
        <div class="chart-box"><div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f4ca; 所有策略滚动夏普对比</div><canvas id="chartAllRollingSharpe"></canvas></div>
        <div class="chart-box"><div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f4c9; 所有策略滚动回撤对比</div><canvas id="chartAllRollingDD"></canvas></div>
    </div>

    <div class="chart-box">
        <div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f50d; 策略相关性热力图</div>
        <canvas id="chartCorr" style="max-height:350px;"></canvas>
    </div>

    <div class="two-col">
        <div class="chart-box"><div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f4c8; 所有策略样本内净值对比</div><canvas id="chartAllIS"></canvas></div>
        <div class="chart-box"><div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f4c8; 所有策略样本外净值对比</div><canvas id="chartAllOS"></canvas></div>
    </div>

    $rebalance_html

    $evaluation_html

    <div class="footer">
        &#xa9; $now_year 量化回测系统 | 由 PyBroker + Chart.js 生成 | 仅供研究参考，不构成投资建议
    </div>
</div>

<script>
var CHART_FONT = "-apple-system, BlinkMacSystemFont, 'PingFang SC', 'Microsoft YaHei', sans-serif";
var COLORS = {
    ts_momentum: '#3b82f6', roll_yield: '#f59e0b', alpha019: '#8b5cf6',
    alpha032: '#06b6d4',
    fusion: '#10b981', switching: '#ef4444'
};
var reportData = """
    + "$chart_data_json"
    + """;

(function() {
    if (!reportData || !reportData.equity_curves) return;
    var ec = reportData.equity_curves;
    var names = Object.keys(ec);
    var datasets = names.map(function(name) {
        return {
            label: ec[name].label || name,
            data: ec[name].equity,
            borderColor: COLORS[name] || '#666',
            backgroundColor: 'transparent',
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.1,
        };
    });
    new Chart(document.getElementById('chartEquity'), {
        type: 'line',
        data: { labels: ec[names[0]].dates, datasets: datasets },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                tooltip: { callbacks: { label: function(ctx) { return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(4); } } },
                legend: { position: 'top', labels: { usePointStyle: true, padding: 16 } },
            },
            scales: {
                x: { type: 'category', ticks: { maxTicksLimit: 12, autoSkip: true } },
                y: { title: { display: true, text: '净值 (归一化)' } },
            },
        },
    });
})();

(function() {
    var ad = reportData.all_drawdowns;
    if (!ad) return;
    var keys = Object.keys(ad);
    if (!keys.length) return;
    var DD_COLORS = {
        E1_ts_momentum: '#3b82f6', E1_roll_yield: '#f59e0b', E1_alpha019: '#8b5cf6',
        E1_alpha032: '#06b6d4', E2_Fusion: '#10b981', E4_Switching: '#ef4444'
    };
    var DD_LABELS = {
        E1_ts_momentum: '时序动量', E1_roll_yield: '展期收益', E1_alpha019: 'Alpha019',
        E1_alpha032: 'Alpha032', E2_Fusion: '融合策略', E4_Switching: '策略切换'
    };
    var DD_BG = {
        E1_ts_momentum: 'rgba(59,130,246,0.08)', E1_roll_yield: 'rgba(245,158,11,0.08)',
        E1_alpha019: 'rgba(139,92,246,0.08)', E1_alpha032: 'rgba(6,182,212,0.08)',
        E2_Fusion: 'rgba(16,185,129,0.08)', E4_Switching: 'rgba(239,68,68,0.08)'
    };
    var firstKey = keys[0];
    var labels = ad[firstKey].dates;
    var datasets = keys.map(function(k) {
        var info = ad[k];
        return {
            label: DD_LABELS[k] || k,
            data: info.drawdown,
            borderColor: DD_COLORS[k] || '#666',
            backgroundColor: DD_BG[k] || 'rgba(102,102,102,0.05)',
            fill: true, borderWidth: 1.5, pointRadius: 0, tension: 0.2,
        };
    });
    var summaryHtml = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;">';
    keys.forEach(function(k) {
        var info = ad[k];
        var color = DD_COLORS[k] || '#666';
        var label = DD_LABELS[k] || k;
        summaryHtml += '<span style="display:inline-flex;align-items:center;gap:4px;padding:4px 10px;border-radius:6px;font-size:12px;background:' + color + '15;border:1px solid ' + color + '40;">'
            + '<span style="width:8px;height:8px;border-radius:50%;background:' + color + ';display:inline-block;"></span>'
            + '<strong>' + label + '</strong>'
            + '<span style="color:#666;">最大回撤: ' + info.max_dd_pct + '%</span>'
            + '<span style="color:#666;">持续: ' + info.duration_days + '天</span>'
            + '</span>';
    });
    summaryHtml += '</div>';
    var container = document.getElementById('chartAllDrawdowns').parentElement;
    container.insertAdjacentHTML('afterbegin', summaryHtml);
    new Chart(document.getElementById('chartAllDrawdowns'), {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                tooltip: { callbacks: { label: function(ctx) { return ctx.dataset.label + ' 回撤: ' + ctx.parsed.y.toFixed(2) + '%'; } } },
                legend: { position: 'top', labels: { usePointStyle: true, padding: 16 } },
            },
            scales: {
                x: { type: 'category', ticks: { maxTicksLimit: 12, autoSkip: true } },
                y: { title: { display: true, text: '回撤 (%)' }, max: 0, ticks: { callback: function(v) { return v.toFixed(0) + '%'; } } },
            },
        },
    });
})();

(function() {
    var sc = reportData.risk_return;
    if (!sc || !sc.length) return;
    var datasets = sc.map(function(d) {
        return {
            label: d.name + ' (Sharpe=' + d.sharpe.toFixed(3) + ')',
            data: [{ x: d.ann_volatility, y: d.ann_return, sharpe: d.sharpe }],
            backgroundColor: COLORS[d.key] || '#666',
            borderColor: COLORS[d.key] || '#666',
            pointRadius: 8, pointHoverRadius: 12,
        };
    });
    new Chart(document.getElementById('chartScatter'), {
        type: 'scatter',
        data: { datasets: datasets },
        options: {
            responsive: true,
            plugins: {
                tooltip: { callbacks: { label: function(ctx) { return ctx.dataset.label.split(' (')[0] + ': 波动率=' + ctx.parsed.x.toFixed(2) + '%, 收益=' + ctx.parsed.y.toFixed(2) + '%, Sharpe=' + ctx.raw.sharpe.toFixed(3); } } },
                legend: { position: 'top', labels: { usePointStyle: true } },
            },
            scales: {
                x: { title: { display: true, text: '年化波动率 (%)' } },
                y: { title: { display: true, text: '年化收益率 (%)' } },
            },
        },
    });
})();

(function() {
    var hm = reportData.heatmap_data;
    var yrs = reportData.years_set;
    if (!hm || !yrs || !yrs.length) return;
    var canvas = document.getElementById('chartHeatmap');
    var months = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'];
    var cellW = 62, cellH = 28, leftPad = 68, topPad = 40;
    canvas.width = leftPad + 12 * cellW + 20;
    canvas.height = topPad + yrs.length * cellH + 40;
    canvas.style.width = '100%';
    canvas.style.height = 'auto';
    var ctx = canvas.getContext('2d');
    ctx.font = '11px ' + CHART_FONT;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    ctx.fillStyle = '#475569';
    for (var yi = 0; yi < yrs.length; yi++) {
        ctx.fillText(yrs[yi], leftPad - 8, topPad + yi * cellH + cellH/2);
    }
    ctx.textAlign = 'center';
    for (var mi = 0; mi < 12; mi++) {
        ctx.fillText(months[mi], leftPad + mi * cellW + cellW/2, topPad - 14);
    }
    function heatColor(val) {
        if (val === null || val === undefined) return '#f1f5f9';
        var maxAbs = 15;
        var ratio = Math.max(-1, Math.min(1, val / maxAbs));
        if (ratio >= 0) {
            var r = Math.round(34 + (1-ratio) * 221);
            var g = Math.round(197 + (1-ratio) * 58);
            var b = Math.round(94 + (1-ratio) * 161);
            return 'rgb(' + r + ',' + g + ',' + b + ')';
        } else {
            var r2 = Math.round(220 + (1+ratio) * 35);
            var g2 = Math.round(38 + (1+ratio) * 62);
            var b2 = Math.round(38 + (1+ratio) * 62);
            return 'rgb(' + r2 + ',' + g2 + ',' + b2 + ')';
        }
    }
    for (var yi = 0; yi < yrs.length; yi++) {
        for (var mi = 0; mi < 12; mi++) {
            var val = hm[yi] && hm[yi][mi] !== undefined ? hm[yi][mi] : null;
            ctx.fillStyle = heatColor(val);
            ctx.fillRect(leftPad + mi * cellW, topPad + yi * cellH, cellW - 1, cellH - 1);
            if (val !== null) {
                ctx.fillStyle = Math.abs(val) > 8 ? '#fff' : '#1a1a2e';
                ctx.fillText(val.toFixed(1) + '%', leftPad + mi * cellW + cellW/2, topPad + yi * cellH + cellH/2);
            }
        }
    }
    var legendY = topPad + yrs.length * cellH + 26;
    ctx.textAlign = 'left';
    for (var i = 0; i <= 10; i++) {
        var t = (i - 5) / 5 * 15;
        ctx.fillStyle = heatColor(t);
        ctx.fillRect(leftPad + i * 32, legendY, 28, 14);
        if (i % 2 === 0) {
            ctx.fillStyle = '#475569';
            ctx.fillText(t.toFixed(0) + '%', leftPad + i * 32 + 14, legendY + 26);
        }
    }
})();

(function() {
    var ahm = reportData.all_heatmaps;
    if (!ahm) return;
    var keys = Object.keys(ahm);
    if (!keys.length) return;
    var HM_COLORS = {
        E1_ts_momentum: '#3b82f6', E1_roll_yield: '#f59e0b', E1_alpha019: '#8b5cf6',
        E1_alpha032: '#06b6d4', E2_Fusion: '#10b981', E4_Switching: '#ef4444'
    };
    var HM_LABELS = {
        E1_ts_momentum: '时序动量', E1_roll_yield: '展期收益', E1_alpha019: 'Alpha019',
        E1_alpha032: 'Alpha032', E2_Fusion: '融合策略', E4_Switching: '策略切换'
    };
    var months = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'];
    var container = document.getElementById('allHeatmapsContainer');
    function heatColor(val) {
        if (val === null || val === undefined) return '#f1f5f9';
        var maxAbs = 15;
        var ratio = Math.max(-1, Math.min(1, val / maxAbs));
        if (ratio >= 0) {
            var r = Math.round(34 + (1-ratio) * 221);
            var g = Math.round(197 + (1-ratio) * 58);
            var b = Math.round(94 + (1-ratio) * 161);
            return 'rgb(' + r + ',' + g + ',' + b + ')';
        } else {
            var r2 = Math.round(220 + (1+ratio) * 35);
            var g2 = Math.round(38 + (1+ratio) * 62);
            var b2 = Math.round(38 + (1+ratio) * 62);
            return 'rgb(' + r2 + ',' + g2 + ',' + b2 + ')';
        }
    }
    function drawHeatmap(canvasEl, hm, yrs, label, color) {
        var cellW = 56, cellH = 24, leftPad = 60, topPad = 36;
        canvasEl.width = leftPad + 12 * cellW + 20;
        canvasEl.height = topPad + yrs.length * cellH + 50;
        canvasEl.style.width = '100%';
        canvasEl.style.height = 'auto';
        var ctx2 = canvasEl.getContext('2d');
        ctx2.font = '10px ' + CHART_FONT;
        ctx2.textAlign = 'right';
        ctx2.textBaseline = 'middle';
        ctx2.fillStyle = '#475569';
        for (var yi = 0; yi < yrs.length; yi++) {
            ctx2.fillText(yrs[yi], leftPad - 6, topPad + yi * cellH + cellH/2);
        }
        ctx2.textAlign = 'center';
        for (var mi = 0; mi < 12; mi++) {
            ctx2.fillText(months[mi], leftPad + mi * cellW + cellW/2, topPad - 12);
        }
        for (var yi2 = 0; yi2 < yrs.length; yi2++) {
            for (var mi2 = 0; mi2 < 12; mi2++) {
                var val = hm[yi2] && hm[yi2][mi2] !== undefined ? hm[yi2][mi2] : null;
                ctx2.fillStyle = heatColor(val);
                ctx2.fillRect(leftPad + mi2 * cellW, topPad + yi2 * cellH, cellW - 1, cellH - 1);
                if (val !== null) {
                    ctx2.fillStyle = Math.abs(val) > 8 ? '#fff' : '#1a1a2e';
                    ctx2.fillText(val.toFixed(1) + '%', leftPad + mi2 * cellW + cellW/2, topPad + yi2 * cellH + cellH/2);
                }
            }
        }
        var legendY2 = topPad + yrs.length * cellH + 20;
        ctx2.textAlign = 'left';
        for (var li = 0; li <= 10; li++) {
            var t2 = (li - 5) / 5 * 15;
            ctx2.fillStyle = heatColor(t2);
            ctx2.fillRect(leftPad + li * 28, legendY2, 24, 12);
            if (li % 2 === 0) {
                ctx2.fillStyle = '#475569';
                ctx2.fillText(t2.toFixed(0) + '%', leftPad + li * 28 + 12, legendY2 + 22);
            }
        }
    }
    var html = '<div style="font-size:14px;font-weight:600;margin:16px 0 8px;">&#x1f321;&#xfe0f; 所有策略月度收益率热力图</div>';
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">';
    keys.forEach(function(k) {
        var info = ahm[k];
        var label = HM_LABELS[k] || k;
        var color = HM_COLORS[k] || '#666';
        var canvasId = 'chartHeatmap_' + k;
        html += '<div class="chart-box"><div style="font-size:12px;font-weight:600;margin-bottom:8px;color:' + color + ';">' + label + '</div><canvas id="' + canvasId + '"></canvas></div>';
    });
    html += '</div>';
    container.innerHTML = html;
    keys.forEach(function(k) {
        var info = ahm[k];
        var canvasEl = document.getElementById('chartHeatmap_' + k);
        if (canvasEl) drawHeatmap(canvasEl, info.data, info.years_set, HM_LABELS[k] || k, HM_COLORS[k] || '#666');
    });
})();

(function() {
    var ars = reportData.all_rolling_sharpe;
    if (!ars) return;
    var keys = Object.keys(ars);
    if (!keys.length) return;
    var RS_COLORS = {
        E1_ts_momentum: '#3b82f6', E1_roll_yield: '#f59e0b', E1_alpha019: '#8b5cf6',
        E1_alpha032: '#06b6d4', E2_Fusion: '#10b981', E4_Switching: '#ef4444'
    };
    var RS_LABELS = {
        E1_ts_momentum: '时序动量', E1_roll_yield: '展期收益', E1_alpha019: 'Alpha019',
        E1_alpha032: 'Alpha032', E2_Fusion: '融合策略', E4_Switching: '策略切换'
    };
    var firstKey = keys[0];
    var labels = ars[firstKey].dates;
    var datasets = keys.map(function(k) {
        return {
            label: RS_LABELS[k] || k,
            data: ars[k].values,
            borderColor: RS_COLORS[k] || '#666',
            backgroundColor: 'transparent',
            borderWidth: 1.5, pointRadius: 0, tension: 0.2,
        };
    });
    new Chart(document.getElementById('chartAllRollingSharpe'), {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                tooltip: { callbacks: { label: function(ctx) { return ctx.dataset.label + ' 夏普: ' + ctx.parsed.y.toFixed(4); } } },
                legend: { position: 'top', labels: { usePointStyle: true, padding: 16 } },
                annotation: { annotations: { zeroLine: { type: 'line', yMin: 0, yMax: 0, borderColor: '#94a3b8', borderWidth: 1, borderDash: [4,4] } } },
            },
            scales: {
                x: { type: 'category', ticks: { maxTicksLimit: 12, autoSkip: true } },
                y: { title: { display: true, text: '夏普比率' } },
            },
        },
    });
})();

(function() {
    var ard = reportData.all_rolling_dd;
    if (!ard) return;
    var keys = Object.keys(ard);
    if (!keys.length) return;
    var RDD_COLORS = {
        E1_ts_momentum: '#3b82f6', E1_roll_yield: '#f59e0b', E1_alpha019: '#8b5cf6',
        E1_alpha032: '#06b6d4', E2_Fusion: '#10b981', E4_Switching: '#ef4444'
    };
    var RDD_LABELS = {
        E1_ts_momentum: '时序动量', E1_roll_yield: '展期收益', E1_alpha019: 'Alpha019',
        E1_alpha032: 'Alpha032', E2_Fusion: '融合策略', E4_Switching: '策略切换'
    };
    var RDD_BG = {
        E1_ts_momentum: 'rgba(59,130,246,0.06)', E1_roll_yield: 'rgba(245,158,11,0.06)',
        E1_alpha019: 'rgba(139,92,246,0.06)', E1_alpha032: 'rgba(6,182,212,0.06)',
        E2_Fusion: 'rgba(16,185,129,0.06)', E4_Switching: 'rgba(239,68,68,0.06)'
    };
    var firstKey = keys[0];
    var labels = ard[firstKey].dates;
    var datasets = keys.map(function(k) {
        return {
            label: RDD_LABELS[k] || k,
            data: ard[k].values,
            borderColor: RDD_COLORS[k] || '#666',
            backgroundColor: RDD_BG[k] || 'rgba(102,102,102,0.05)',
            fill: true, borderWidth: 1.5, pointRadius: 0, tension: 0.2,
        };
    });
    new Chart(document.getElementById('chartAllRollingDD'), {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                tooltip: { callbacks: { label: function(ctx) { return ctx.dataset.label + ' 回撤: ' + ctx.parsed.y.toFixed(2) + '%'; } } },
                legend: { position: 'top', labels: { usePointStyle: true, padding: 16 } },
            },
            scales: {
                x: { type: 'category', ticks: { maxTicksLimit: 12, autoSkip: true } },
                y: { title: { display: true, text: '最大回撤 (%)' }, max: 0, ticks: { callback: function(v) { return v.toFixed(0) + '%'; } } },
            },
        },
    });
})();

(function() {
    var corr = reportData.correlation;
    if (!corr || !corr.names || !corr.names.length) return;
    var names = corr.names;
    var n = names.length;
    var datasets = [];
    var bgColors = ['#3b82f6','#f59e0b','#8b5cf6','#10b981','#ef4444','#ec4899','#06b6d4','#84cc16'];
    for (var i = 0; i < n; i++) {
        datasets.push({
            label: names[i],
            data: corr.matrix[i],
            backgroundColor: bgColors[i % bgColors.length] + '80',
            borderColor: bgColors[i % bgColors.length],
            borderWidth: 1,
        });
    }
    new Chart(document.getElementById('chartCorr'), {
        type: 'bar',
        data: { labels: names, datasets: datasets },
        options: {
            responsive: true,
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { stacked: false },
                y: { min: -1, max: 1, title: { display: true, text: '相关性系数' } },
            },
        },
    });
})();

(function() {
    var aie = reportData.all_is_equity;
    if (!aie) return;
    var keys = Object.keys(aie);
    if (!keys.length) return;
    var IS_COLORS = {
        E1_ts_momentum: '#3b82f6', E1_roll_yield: '#f59e0b', E1_alpha019: '#8b5cf6',
        E1_alpha032: '#06b6d4', E2_Fusion: '#10b981', E4_Switching: '#ef4444'
    };
    var IS_LABELS = {
        E1_ts_momentum: '时序动量', E1_roll_yield: '展期收益', E1_alpha019: 'Alpha019',
        E1_alpha032: 'Alpha032', E2_Fusion: '融合策略', E4_Switching: '策略切换'
    };
    var firstKey = keys[0];
    var labels = aie[firstKey].dates;
    var datasets = keys.map(function(k) {
        return {
            label: IS_LABELS[k] || k,
            data: aie[k].equity,
            borderColor: IS_COLORS[k] || '#666',
            backgroundColor: 'transparent',
            borderWidth: 1.5, pointRadius: 0, tension: 0.1,
        };
    });
    new Chart(document.getElementById('chartAllIS'), {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                tooltip: { callbacks: { label: function(ctx) { return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(4); } } },
                legend: { position: 'top', labels: { usePointStyle: true, padding: 12 } },
            },
            scales: {
                x: { type: 'category', ticks: { maxTicksLimit: 8, autoSkip: true } },
                y: { title: { display: true, text: '净值 (归一化)' } },
            },
        },
    });
})();

(function() {
    var aoe = reportData.all_os_equity;
    if (!aoe) return;
    var keys = Object.keys(aoe);
    if (!keys.length) return;
    var OS_COLORS = {
        E1_ts_momentum: '#3b82f6', E1_roll_yield: '#f59e0b', E1_alpha019: '#8b5cf6',
        E1_alpha032: '#06b6d4', E2_Fusion: '#10b981', E4_Switching: '#ef4444'
    };
    var OS_LABELS = {
        E1_ts_momentum: '时序动量', E1_roll_yield: '展期收益', E1_alpha019: 'Alpha019',
        E1_alpha032: 'Alpha032', E2_Fusion: '融合策略', E4_Switching: '策略切换'
    };
    var firstKey = keys[0];
    var labels = aoe[firstKey].dates;
    var datasets = keys.map(function(k) {
        return {
            label: OS_LABELS[k] || k,
            data: aoe[k].equity,
            borderColor: OS_COLORS[k] || '#666',
            backgroundColor: 'transparent',
            borderWidth: 1.5, pointRadius: 0, tension: 0.1,
        };
    });
    new Chart(document.getElementById('chartAllOS'), {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                tooltip: { callbacks: { label: function(ctx) { return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(4); } } },
                legend: { position: 'top', labels: { usePointStyle: true, padding: 12 } },
            },
            scales: {
                x: { type: 'category', ticks: { maxTicksLimit: 8, autoSkip: true } },
                y: { title: { display: true, text: '净值 (归一化)' } },
            },
        },
    });
})();
</script>
</body>
</html>"""
)

_DEFAULT_EVALUATION_HTML = """
    <div class="section-title">综合评价与改进建议</div>
    <div class="section-desc">基于回测结果的多维度定性分析，识别策略核心问题并提出改进方向</div>

    <div class="card">
        <div class="card-header">核心问题诊断</div>
        <div class="eval-problem">
            <div class="eval-problem-title">1. 风险调整后收益极差（Sharpe 过低）</div>
            <p>所有策略的<strong>年化Sharpe比率</strong>均在 <span class="negative">0.008 ~ 0.022</span> 之间，远低于通常可接受水平（一般 &gt;0.5 才被认为具有风险溢价）。这意味着策略承担了很大的波动和回撤，却没有获得对应的超额回报。</p>
        </div>
        <div class="eval-problem">
            <div class="eval-problem-title">2. 最大回撤偏高，风控不足</div>
            <p>回撤最小的 E1_ts_momentum 也有较大回撤，而 E2_Fusion 回撤更高。在长达10年的回测中，这样的回撤幅度对实盘资金管理是很大考验。</p>
        </div>
        <div class="eval-problem">
            <div class="eval-problem-title">3. 收益率与交易频率不匹配</div>
            <ul>
                <li><strong>E1_alpha019</strong> 收益较高，但回撤也较高，年化收益率需关注。</li>
                <li><strong>E1_ts_momentum</strong> 和 <strong>E4_Switching</strong> 交易频率较高，换手频繁但收益需优化。</li>
                <li><strong>E1_roll_yield</strong> 收益偏低，需调整参数。</li>
            </ul>
        </div>
        <div class="eval-problem">
            <div class="eval-problem-title">4. 样本内外表现差异显著</div>
            <p>部分策略在样本外表现明显变差，提示可能存在<strong>过拟合风险</strong>。样本内夏普比率与样本外差距过大时，需警惕参数对历史数据的过度适配。</p>
        </div>
        <div class="eval-problem">
            <div class="eval-problem-title">5. 策略间相关性偏高</div>
            <p>多策略组合的分散化效果有限，策略间相关性较高时，组合回撤与单策略回撤接近，未能有效降低系统性风险。</p>
        </div>
    </div>

    <div class="card" style="margin-top:16px;">
        <div class="card-header">多维度评分</div>
        <div class="table-wrapper"><table>
            <thead><tr><th>评价维度</th><th>评级</th><th>说明</th></tr></thead>
            <tbody>
                <tr><td>绝对收益</td><td><span class="badge badge-danger">&#x274c; 较差</span></td><td>十年最高仅31%，年化约2.7%</td></tr>
                <tr><td>风险调整收益</td><td><span class="badge badge-danger">&#x274c; 很差</span></td><td>Sharpe &lt; 0.03，近乎随机漫步</td></tr>
                <tr><td>回撤控制</td><td><span class="badge badge-danger">&#x274c; 不合格</span></td><td>普遍 &gt;15%，有的超30%</td></tr>
                <tr><td>交易频率合理性</td><td><span class="badge badge-warning">&#x26a0;&#xfe0f; 存疑</span></td><td>高频策略收益并不更好</td></tr>
                <tr><td>样本外稳定性</td><td><span class="badge badge-warning">&#x26a0;&#xfe0f; 需关注</span></td><td>部分策略样本外衰减明显</td></tr>
                <tr><td>策略分散化</td><td><span class="badge badge-warning">&#x26a0;&#xfe0f; 不足</span></td><td>策略间相关性偏高，组合效果有限</td></tr>
                <tr><td>实盘可行性</td><td><span class="badge badge-danger">&#x274c; 低</span></td><td>风险收益特征不具备吸引力</td></tr>
            </tbody>
        </table></div>
    </div>

    <div class="card" style="margin-top:16px;">
        <div class="card-header">改进建议（已实施 + 待实施）</div>
        <div class="eval-problem">
            <div class="eval-problem-title" style="color:#10b981;">✅ 已实施的改进</div>
            <ol class="suggestion-list">
                <li><strong>检查过拟合</strong>：已增加参数扰动测试和 WalkForward OOS 验证，观察样本外表现是否明显变差。</li>
                <li><strong>加强风控</strong>：止损收紧至 2%、增加 ATR 动态止损、波动率目标仓位管理、信号连续确认已全部实现。</li>
                <li><strong>降低换手率</strong>：信号确认机制、均线间距过滤均已实现，预期交易次数显著减少。</li>
                <li><strong>交易成本真实化</strong>：手续费+滑点提升至万10，淘汰边际利润策略。</li>
                <li><strong>策略相关性过滤</strong>：融合策略自动降权高相关策略对，降低风险集中度。</li>
            </ol>
        </div>
        <div class="eval-problem" style="margin-top:12px;">
            <div class="eval-problem-title" style="color:#f59e0b;">⚠️ 待实施的改进</div>
            <ol class="suggestion-list">
                <li><strong>因子有效性提升</strong>：当前因子IC偏低，需引入更高预测力的因子（如订单流、资金流、期限结构等），或优化因子构造方式（非线性变换、交叉项）。</li>
                <li><strong>自适应参数机制</strong>：固定参数在市场regime切换时失效，建议实现滚动窗口自适应参数（如EMA窗口、ATR倍数随波动率调整）。</li>
                <li><strong>多时间框架融合</strong>：当前仅使用日频信号，建议引入周频/月频趋势判断作为过滤层，降低逆势交易频率。</li>
                <li><strong>动态仓位管理</strong>：根据策略近期表现（如滚动Sharpe）动态调整各策略权重，表现差时自动降权。</li>
                <li><strong>止损策略优化</strong>：当前固定止损可能过于刚性，建议实现追踪止损（Trailing Stop）和时间止损（持仓N日未达目标自动平仓）。</li>
                <li><strong>品种选择优化</strong>：并非所有品种适合所有策略，建议为每个策略筛选适配品种池（基于品种波动率、流动性、趋势性等指标）。</li>
                <li><strong>实盘模拟验证</strong>：回测结果需经过纸面交易（Paper Trading）验证至少3个月，确认实际滑点、成交率与回测假设一致。</li>
            </ol>
        </div>
    </div>
"""
