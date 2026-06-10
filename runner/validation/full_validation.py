"""
全量验证模块：3 阶段串行执行（调参 → 6 品种 EW 横截面 → 蒙特卡洛）。

原根目录 run_full_validation.py 已迁移至此，保留为 Pipeline.full_validation()
编排入口。

时间划分（默认）:
  - in_sample (训练): 2020-01-01 ~ 2022-12-31
  - out_sample (OOS): 2023-01-01 ~ 2024-12-31
  - 蒙特卡洛使用全段: 2020-01-01 ~ 2024-12-31

6 品种: SHFE.AL, SHFE.CU, CZCE.FG, SHFE.RU, DCE.PP, CZCE.CF

三阶段:
  Phase 1 - 全量调参: 在 in_sample 区间对 5 子策略网格搜索 → OOS 优选
  Phase 2 - 6 品种横截面 EW 组合回测: 用 OOS 最优参数在 train/OOS 段分别跑
  Phase 3 - 蒙特卡洛 1000 次鲁棒性测试: 在全段对 OOS 最优参数做扰动
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.config.strategy_profiles import StrategyLibrary
from core.engine.backtest_runner import PyBrokerBacktestRunner
from core.engine.pybroker_data_source import PyBrokerDataSource
from runner.common.utils import safe_float
from runner.validation.monte_carlo import task3_monte_carlo

# 6 品种横截面 EW 组合
DEFAULT_SYMBOLS_6: List[str] = [
    "SHFE.AL", "SHFE.CU", "CZCE.FG", "SHFE.RU", "DCE.PP", "CZCE.CF",
]

# 5 子策略
DEFAULT_STRATEGIES_5: List[str] = [
    "trend",
    "term_structure",
    "mean_reversion",
    "vol_breakout",
    "composite_resonance",
]


# ============================================================
# Phase 1: 调参
# ============================================================
def _phase1_optimize(pipe, in_sample_start: str, in_sample_end: str,
                     oos_start: str, oos_end: str) -> Dict[str, Any]:
    """Phase 1: 委托 Pipeline.optimize() 在 in_sample 内做网格+OOS 优选。"""
    logger.info("=" * 80)
    logger.info("  [Phase 1] 全量调参 (in_sample: %s ~ %s)", in_sample_start, in_sample_end)
    logger.info("=" * 80)

    pipe = pipe.with_config(
        full_start=in_sample_start,
        full_end=oos_end,
        in_sample_end=in_sample_end,
    )
    pipe = pipe.optimize(tasks=["grid", "oos"], save_to_config=True)

    opt = pipe._results.get("optimization", {})
    best_params = opt.get("best_params", {}) or {}
    logger.info("  [Phase 1] 完成: %d 个策略的最优参数", len(best_params))
    for sname, params in best_params.items():
        logger.info("    - %s: %s", sname, params)
    return {"best_params": best_params, "pipe": pipe}


# ============================================================
# Phase 2: 6 品种横截面 EW
# ============================================================
def _phase2_ew_backtest(
    data_source: PyBrokerDataSource,
    config: BacktestConfig,
    strategies: List[str],
    windows: List[Tuple[str, str, str]],
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Phase 2: 对每个 (window × strategy) 跑全品种回测，汇总为 EW 组合。

    返回: 详细结果 DataFrame (列: window, strategy, sharpe, total_return_pct, ...)
    """
    logger.info("=" * 80)
    logger.info("  [Phase 2] %d 品种横截面 EW 组合回测", len(DEFAULT_SYMBOLS_6))
    logger.info("=" * 80)

    rows: List[Dict[str, Any]] = []
    for window_name, start, end in windows:
        for sname in strategies:
            try:
                runner = PyBrokerBacktestRunner(data_source, config)
                runner.register_strategies([sname])
                if best_params and best_params.get(sname):
                    runner.set_custom_params({sname: best_params[sname]})
                res = runner.run(start_date=start, end_date=end)
                metrics = res.metrics if res and hasattr(res, "metrics") else {}
                rows.append({
                    "window": window_name,
                    "strategy": sname,
                    "sharpe": safe_float(metrics.get("sharpe", 0)),
                    "total_return_pct": safe_float(metrics.get("total_return_pct", 0)),
                    "max_dd_pct": safe_float(metrics.get("max_drawdown_pct", 0)),
                    "total_pnl": safe_float(metrics.get("total_pnl", 0)),
                    "trade_count": int(metrics.get("trade_count", 0) or 0),
                    "win_rate": safe_float(metrics.get("win_rate", 0)),
                })
            except Exception as e:
                logger.error("  %s %s 失败: %s", window_name, sname, e)
                rows.append({
                    "window": window_name,
                    "strategy": sname,
                    "sharpe": 0.0, "total_return_pct": 0.0, "max_dd_pct": 0.0,
                    "total_pnl": 0.0, "trade_count": 0, "win_rate": 0.0, "error": str(e),
                })

    df = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "phase2_ew_results.csv", index=False)

    # 详细结果打印
    logger.info("  [Phase 2] 详细结果 (window × strategy):")
    logger.info("    %-15s %-22s %8s %8s %8s %12s %8s",
                "window", "strategy", "sharpe", "ret%", "dd%", "pnl", "trades")
    for _, r in df.iterrows():
        logger.info(
            "    %-15s %-22s %8.4f %8.4f %8.4f %12.2f %8d",
            r["window"], r["strategy"],
            r["sharpe"], r["total_return_pct"], r["max_dd_pct"],
            r["total_pnl"], int(r["trade_count"]),
        )

    # EW 聚合
    summary_rows: List[Dict[str, Any]] = []
    for window_name in df["window"].unique():
        sub = df[df["window"] == window_name]
        valid = sub[sub["trade_count"] > 0]
        ew_sharpe = safe_float(valid["sharpe"].mean()) if not valid.empty else 0.0
        ew_return_pct = safe_float(valid["total_return_pct"].mean()) if not valid.empty else 0.0
        ew_mdd_pct = safe_float(valid["max_dd_pct"].mean()) if not valid.empty else 0.0
        ew_pnl = safe_float(valid["total_pnl"].sum())
        ew_trades = int(sub["trade_count"].sum())
        summary_rows.append({
            "window": window_name,
            "ew_sharpe": round(ew_sharpe, 4),
            "ew_return_pct": round(ew_return_pct, 4),
            "ew_mdd_pct": round(ew_mdd_pct, 4),
            "ew_pnl": round(ew_pnl, 2),
            "total_trades": ew_trades,
            "valid_strategies": len(valid),
        })
        logger.info(
            "    %-15s EW_Sharpe=%7.4f  EW_Ret%%=%7.4f  EW_DD%%=%7.4f  PnL=%12.2f  trades=%d",
            window_name, ew_sharpe, ew_return_pct, ew_mdd_pct, ew_pnl, ew_trades,
        )

    # Sharpe 衰减率
    train = next((r for r in summary_rows if r["window"] == "TRAIN_2020_2022"), None)
    oos = next((r for r in summary_rows if r["window"] == "OOS_2023_2024"), None)
    if train and oos:
        if abs(train["ew_sharpe"]) > 1e-6:
            decay = round((oos["ew_sharpe"] - train["ew_sharpe"]) / abs(train["ew_sharpe"]), 4)
        else:
            decay = 0.0
        logger.info(
            "    OOS Sharpe 衰减率: %+.1f%% (训练 %.4f → OOS %.4f)",
            decay * 100, train["ew_sharpe"], oos["ew_sharpe"],
        )
        summary_rows.append({
            "window": "OOS_DECAY",
            "ew_sharpe": decay,
            "ew_return_pct": 0, "ew_mdd_pct": 0, "ew_pnl": 0,
            "total_trades": 0, "valid_strategies": 0,
        })

    if output_dir is not None:
        pd.DataFrame(summary_rows).to_csv(output_dir / "phase2_ew_summary.csv", index=False)
    return df


# ============================================================
# Phase 3: 蒙特卡洛
# ============================================================
def _phase3_monte_carlo(
    data_source: PyBrokerDataSource,
    config: BacktestConfig,
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Phase 3: 委托 runner.validation.monte_carlo.task3_monte_carlo 执行 1000 次扰动。
    """
    logger.info("=" * 80)
    logger.info("  [Phase 3] 蒙特卡洛 1000 次鲁棒性测试")
    logger.info("=" * 80)

    if output_dir is None:
        output_dir = Path("output_backtest_pybroker/full_validation/phase3_mc")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lib = StrategyLibrary()
    result = task3_monte_carlo(
        data_source=data_source,
        config=config,
        lib=lib,
        output_dir=output_dir,
        best_params=best_params,
        cross_sectional=False,
    )
    summary = result.get("summary")
    if summary is not None and isinstance(summary, pd.DataFrame):
        logger.info("  蒙特卡洛汇总 (1000 次模拟):")
        logger.info("\n%s", summary.to_string(index=False))
        return summary
    return pd.DataFrame()


# ============================================================
# 报告生成
# ============================================================
def _write_report(
    output_dir: Path,
    best_params: Dict[str, Dict[str, Any]],
    phase2_df: pd.DataFrame,
    phase3_df: pd.DataFrame,
    in_sample_start: str, in_sample_end: str,
    oos_start: str, oos_end: str,
    symbols: List[str],
) -> Path:
    """生成 markdown 报告。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 全量验证报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**时间划分**: train={in_sample_start} ~ {in_sample_end}, OOS={oos_start} ~ {oos_end}",
        f"**品种**: {', '.join(symbols)}",
        "",
        "---",
        "",
        "## Phase 1: 调参最优参数",
        "",
    ]
    for sname, params in best_params.items():
        lines.append(f"- **{sname}**: `{params}`")
    lines.append("")

    if not phase2_df.empty:
        lines += [
            "## Phase 2: 6 品种横截面 EW 组合回测",
            "",
            "| Window | Strategy | Sharpe | Ret% | DD% | PnL | Trades |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in phase2_df.iterrows():
            lines.append(
                f"| {r['window']} | {r['strategy']} | {r['sharpe']:.4f} | "
                f"{r['total_return_pct']:.4f} | {r['max_dd_pct']:.4f} | "
                f"{r['total_pnl']:.2f} | {int(r['trade_count'])} |"
            )
        lines.append("")

    if not phase3_df.empty:
        lines += [
            "## Phase 3: 蒙特卡洛 1000 次鲁棒性测试",
            "",
            "| Strategy | Final Mean | Final Median | 5% CI | 95% CI | Bankruptcy | Avg MaxDD | Calmar |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, r in phase3_df.iterrows():
            lines.append(
                f"| {r.get('strategy', '')} | {r.get('final_mean', 0):.4f} | "
                f"{r.get('final_median', 0):.4f} | {r.get('final_5pct', 0):.4f} | "
                f"{r.get('final_95pct', 0):.4f} | {r.get('bankruptcy_prob', 0):.4f} | "
                f"{r.get('avg_max_dd', 0):.4f} | {r.get('calmar_mean', 0):.4f} |"
            )
        lines.append("")

    out = output_dir / "full_validation_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("  [报告] %s", out)
    return out


# ============================================================
# 主入口
# ============================================================
def run_full_validation(
    pipe,
    in_sample_start: str = "2020-01-01",
    in_sample_end: str = "2023-01-01",
    oos_start: str = "2023-01-01",
    oos_end: str = "2024-12-31",
    full_start: str = "2020-01-01",
    full_end: str = "2024-12-31",
    strategies: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    全量验证主入口：3 阶段串行。

    Args:
        pipe: 已加载数据的 Pipeline 实例
        in_sample_start: 训练区间起始
        in_sample_end: 训练区间结束
        oos_start: OOS 区间起始
        oos_end: OOS 区间结束
        full_start: 全段起始（蒙特卡洛用）
        full_end: 全段结束（蒙特卡洛用）
        strategies: 子策略列表，默认 5 子策略
        output_dir: 输出目录，默认 output_backtest_pybroker/full_validation

    Returns:
        {phase1: {best_params}, phase2: DataFrame, phase3: DataFrame, report: Path}
    """
    if strategies is None:
        strategies = DEFAULT_STRATEGIES_5
    if output_dir is None:
        output_dir = Path("output_backtest_pybroker/full_validation")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 写运行日志
    log_path = output_dir / "run.log"
    logger.add(str(log_path), level="INFO", enqueue=True)

    windows = [
        ("TRAIN_2020_2022", in_sample_start, in_sample_end),
        ("OOS_2023_2024", oos_start, oos_end),
    ]

    # Phase 1: 调参
    p1 = _phase1_optimize(pipe, in_sample_start, in_sample_end, oos_start, oos_end)
    best_params = p1.get("best_params", {})

    # 保存 best_params
    with open(output_dir / "phase1_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    # Phase 2: 6 品种横截面 EW
    pipe = p1["pipe"]
    p2_df = _phase2_ew_backtest(
        data_source=pipe._data,
        config=pipe._config,
        strategies=strategies,
        windows=windows,
        best_params=best_params,
        output_dir=output_dir,
    )

    # Phase 3: 蒙特卡洛
    mc_pipe = pipe.with_config(full_start=full_start, full_end=full_end)
    p3_df = _phase3_monte_carlo(
        data_source=mc_pipe._data,
        config=mc_pipe._config,
        best_params=best_params,
        output_dir=output_dir / "phase3_mc",
    )

    # 汇总报告
    report_path = _write_report(
        output_dir=output_dir,
        best_params=best_params,
        phase2_df=p2_df,
        phase3_df=p3_df,
        in_sample_start=in_sample_start, in_sample_end=in_sample_end,
        oos_start=oos_start, oos_end=oos_end,
        symbols=DEFAULT_SYMBOLS_6,
    )

    return {
        "phase1": p1,
        "phase2": p2_df,
        "phase3": p3_df,
        "report": report_path,
    }


__all__ = ["run_full_validation", "DEFAULT_SYMBOLS_6", "DEFAULT_STRATEGIES_5"]
