"""
多窗口 OOS 验证（子策略版）。

对 5 子策略在多个 OOS 窗口内回测，提取 Sharpe/Return/MaxDD/Trades，
并计算等权组合（EW_BLEND）的平均 Sharpe。绕过 cross_sectional（其 E11
依赖过重），直接验证 5 子策略 + 等权组合。

原根目录 run_multi_oos.py 已迁移至此，保留为 Pipeline.multi_oos() 编排入口。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from core.config import BacktestConfig
from core.engine.backtest_runner import PyBrokerBacktestRunner
from core.engine.pybroker_data_source import PyBrokerDataSource
from runner.common.utils import safe_float

# 5 子策略
DEFAULT_STRATEGIES: List[str] = [
    "trend",
    "term_structure",
    "mean_reversion",
    "vol_breakout",
    "composite_resonance",
]

# 默认 3 个 OOS 窗口
DEFAULT_WINDOWS: List[Tuple[str, str, str]] = [
    ("OOS_2022", "2022-01-01", "2022-12-31"),
    ("OOS_2023", "2023-01-01", "2023-12-31"),
    ("OOS_2024", "2024-01-01", "2024-12-31"),
]


def _run_window(
    data_source: PyBrokerDataSource,
    config: BacktestConfig,
    strategies: List[str],
    test_start: str,
    test_end: str,
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    在 OOS 窗口内对各子策略做回测，提取 Sharpe/Return/MaxDD/Trades。

    Args:
        data_source: PyBrokerDataSource
        config: BacktestConfig
        strategies: 子策略名称列表
        test_start: 窗口起始日
        test_end: 窗口结束日
        best_params: 优化后的最优参数（可选）

    Returns:
        {策略名: {sharpe, total_return, max_drawdown, trade_count, [error]}}
    """
    results: Dict[str, Dict[str, Any]] = {}
    for sname in strategies:
        try:
            runner = PyBrokerBacktestRunner(data_source, config)
            runner.register_strategies([sname])
            if best_params and best_params.get(sname):
                runner.set_custom_params({sname: best_params[sname]})
            res = runner.run(start_date=test_start, end_date=test_end)
            metrics = res.metrics if res and hasattr(res, "metrics") else {}
            results[sname] = {
                "sharpe": safe_float(metrics.get("sharpe", 0.0)),
                "total_return": safe_float(metrics.get("total_return", 0.0)),
                "max_drawdown": safe_float(metrics.get("max_drawdown", 0.0)),
                "trade_count": int(metrics.get("trade_count", 0) or 0),
            }
        except Exception as e:
            logger.error(f"{sname} OOS 回测失败: {e}")
            results[sname] = {
                "sharpe": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "trade_count": 0,
                "error": str(e),
            }
    return results


def _summarize_ew(window_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算等权组合的 Sharpe 平均值与总交易次数。

    Args:
        window_results: 单窗口各策略结果

    Returns:
        EW_BLEND 汇总字典
    """
    valid_sharpes = [
        v["sharpe"]
        for v in window_results.values()
        if v.get("trade_count", 0) > 0 and "error" not in v
    ]
    return {
        "sharpe": round(sum(valid_sharpes) / len(valid_sharpes), 4) if valid_sharpes else 0.0,
        "total_trades": sum(v.get("trade_count", 0) for v in window_results.values()),
        "valid_strategies": len(valid_sharpes),
    }


def run_multi_oos(
    data_source: PyBrokerDataSource,
    config: BacktestConfig,
    output_dir: Path,
    strategies: Optional[List[str]] = None,
    windows: Optional[List[Tuple[str, str, str]]] = None,
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    save_json: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    多窗口 OOS 验证：5 子策略 + 等权组合，按窗口汇总。

    Args:
        data_source: PyBrokerDataSource
        config: BacktestConfig
        output_dir: 输出目录
        strategies: 子策略列表，默认 5 子策略
        windows: (窗口名, 起始日, 结束日) 元组列表，默认 2022/2023/2024
        best_params: 优化后的最优参数
        save_json: 是否保存 JSON 汇总

    Returns:
        {窗口名: {策略名或 EW_BLEND: {指标字典}}}
    """
    strategies = strategies or DEFAULT_STRATEGIES
    windows = windows or DEFAULT_WINDOWS

    summary: Dict[str, Dict[str, Any]] = {}
    for window_name, test_start, test_end in windows:
        logger.info(f"多窗口 OOS: {window_name} ({test_start} ~ {test_end})")
        win_results = _run_window(
            data_source, config, strategies, test_start, test_end, best_params
        )
        win_results["EW_BLEND"] = _summarize_ew(win_results)
        summary[window_name] = win_results

    if save_json:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "multi_window_oos_substrategies.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"多窗口 OOS 汇总已保存: {out_path}")

    return summary


__all__ = ["run_multi_oos", "DEFAULT_STRATEGIES", "DEFAULT_WINDOWS"]
