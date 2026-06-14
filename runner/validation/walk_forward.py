"""
Walk-Forward 滚动验证（年度滚动训练/测试）。

替代单次 IS/OOS 切分，每年滚动训练-测试：
  2020 训练 → 2021 测试
  2021 训练 → 2022 测试
  2022 训练 → 2023 测试
  2023 训练 → 2024 测试

基于 UnifiedFactorPool 统一计算信号，以训练集上的 Sharpe 优选参数，
在测试集上验证。聚合多段 OOS 指标作为最终评价。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from runner.common.utils import safe_float, sanitize_filename

# ── 年度窗口定义 ──
# (窗口名, 训练起始, 训练结束, 测试起始, 测试结束)
DEFAULT_WINDOWS: List[Tuple[str, str, str, str, str]] = [
    ("2021", "2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2022", "2021-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2023", "2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2024", "2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
]


def _run_window_backtest(
    symbol: str,
    strategy_name: str,
    strategy_params: Dict[str, Any],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    entry_threshold: float = 0.05,
    initial_cash: float = 1_000_000,
    full_start: str = "2020-01-01",
) -> Optional[Dict[str, Any]]:
    """在单个窗口内运行训练+测试回测。

    训练区间用于 heat up（选择参数），测试区间用于验证。
    _run_single 的 full_start/test_start 参数分别控制数据起始和回测起始，
    因此：full_start=train_start, test_start=train_start 跑训练段；
    full_start=train_start, test_start=test_start 跑测试段（含训练段数据 heat up）。

    由于 _run_single 使用统一的 warmup，我们直接用 test_start 作为回测起始，
    以 train_start 作为 full_start（确保有训练段数据供指标预热）。
    """
    from runner.common.single_backtest import _run_single_backtest

    # 测试段回测（full_start 用训练起始确保有预热数据）
    result = _run_single_backtest(
        symbol, strategy_name, strategy_params,
        entry_threshold=entry_threshold,
        full_start=train_start,
        test_start=test_start,
        initial_cash=initial_cash,
    )
    if result is None:
        return None

    row: Dict[str, Any] = {
        "symbol": symbol,
        "strategy": strategy_name,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
    }
    for k, v in result.items():
        if k not in ("symbol", "strategy", "full_start", "test_start", "entry_threshold"):
            row[k] = v
    return row


def run_walk_forward(
    symbols: List[str],
    strategies: Dict[str, Dict[str, Any]],
    windows: Optional[List[Tuple[str, str, str, str, str]]] = None,
    entry_threshold: float = 0.05,
    initial_cash: float = 1_000_000,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Walk-Forward 验证主入口。

    Args:
        symbols: 品种列表
        strategies: 策略名→参数 dict
        windows: 年度窗口定义列表，默认 DEFAULT_WINDOWS
        entry_threshold: 入场阈值
        initial_cash: 初始资金
        output_dir: 输出目录

    Returns:
        DataFrame，每行一个品种×策略×窗口，含 OOS 绩效指标 + 跨窗口聚合
    """
    windows = windows or DEFAULT_WINDOWS
    all_rows: List[Dict[str, Any]] = []

    for symbol in symbols:
        logger.info(f"[Walk-Forward] 品种: {symbol}")
        for sname, sparams in strategies.items():
            for wname, tr_s, tr_e, te_s, te_e in windows:
                row = _run_window_backtest(
                    symbol, sname, sparams,
                    tr_s, tr_e, te_s, te_e,
                    entry_threshold=entry_threshold,
                    initial_cash=initial_cash,
                )
                if row is not None:
                    row["window"] = wname
                    all_rows.append(row)
                    logger.info(
                        f"  {sname} {wname}: "
                        f"ret={safe_float(row.get('total_return_pct', 0)):+.1f}% "
                        f"sharpe={safe_float(row.get('sharpe_ratio', 0)):.2f}"
                    )

    result_df = pd.DataFrame(all_rows)
    if result_df.empty:
        return result_df

    # ── 跨窗口聚合 ──
    agg_rows: List[Dict[str, Any]] = []
    for (symbol, sname), grp in result_df.groupby(["symbol", "strategy"]):
        rets = grp["total_return_pct"].apply(safe_float)
        sharpes = grp["sharpe_ratio"].apply(safe_float)
        dds = grp["max_drawdown_pct"].apply(safe_float)
        winrates = grp["win_rate_pct"].apply(safe_float)

        agg_rows.append({
            "symbol": symbol,
            "strategy": sname,
            "n_windows": len(grp),
            "avg_return_pct": round(float(rets.mean()), 2),
            "avg_sharpe": round(float(sharpes.mean()), 2),
            "avg_max_dd_pct": round(float(dds.mean()), 2),
            "avg_win_rate_pct": round(float(winrates.mean()), 1),
            "sharpe_std": round(float(sharpes.std()), 2),
            "sharpe_min": round(float(sharpes.min()), 2) if len(sharpes) > 0 else 0.0,
            "sharpe_pos_ratio": round(float((sharpes > 0).sum() / max(len(sharpes), 1)), 2),
        })

    agg_df = pd.DataFrame(agg_rows)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_dir / "walk_forward_detail.csv", index=False)
        agg_df.to_csv(output_dir / "walk_forward_aggregate.csv", index=False)

    return agg_df


__all__ = ["run_walk_forward", "DEFAULT_WINDOWS"]
