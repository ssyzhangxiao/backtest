"""
参数平原测试。

对核心参数做 ±20% 扰动，运行回测后保留变化 < 15% 的"平原区域"。

用途：
  1. 识别参数的稳定区域（plateau），避免尖峰参数选择
  2. 子样本一致性检查（牛/熊/震荡/高波四个环境 Sharpe 不异号）

使用 UnifiedFactorPool 统一计算信号并执行回测。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from runner.common.utils import safe_float

# ── 各策略的核心参数（仅测试这些） ──
_CORE_PARAMS: Dict[str, List[str]] = {
    "carry": ["lookback", "entry_z"],
    "vol_mean_reversion": ["vol_window", "entry_z"],
    "donchian_breakout": ["entry_lookback", "atr_window"],
    "momentum_ma": ["rsi_window", "momentum_fast", "momentum_mid"],
    "tsi_garch": ["reg_window", "t_stat_threshold"],
    "pair_trading": ["lookback", "entry_z", "ols_window"],
}


def _make_grid(
    base_params: Dict[str, Any],
    core_keys: List[str],
    perturbation: float = 0.20,
    steps: int = 5,
) -> List[Dict[str, Any]]:
    """生成 ±20% 扰动网格。"""
    grid: List[Dict[str, Any]] = []
    for key in core_keys:
        if key not in base_params:
            continue
        base_val = base_params[key]
        if not isinstance(base_val, (int, float)) or base_val <= 0:
            continue
        for step in range(steps):
            factor = 1.0 - perturbation + (2.0 * perturbation * step / (steps - 1))
            val = base_val * factor
            if isinstance(base_val, int):
                val = int(round(val))
                if val < 1:
                    continue
            params = dict(base_params)
            params[key] = val
            grid.append(params)
    return grid


def run_parameter_plateau(
    symbol: str,
    strategy_name: str,
    strategy_params: Dict[str, Any],
    perturbation: float = 0.20,
    steps: int = 5,
    variation_threshold: float = 0.15,
    entry_threshold: float = 0.05,
    initial_cash: float = 1_000_000,
    full_start: str = "2020-01-01",
    test_start: str = "2023-01-01",
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    参数平原测试主入口。

    Args:
        symbol: 品种代码
        strategy_name: 策略名
        strategy_params: 基准参数
        perturbation: 扰动比例（默认 0.20 = ±20%）
        steps: 每参数采样点数（默认 5）
        variation_threshold: 变化率阈值（默认 0.15，变 <15% 为"平原"）
        entry_threshold: 入场阈值
        initial_cash: 初始资金
        full_start: 全量数据起始
        test_start: 回测起始
        output_dir: 输出目录

    Returns:
        DataFrame，每行一个参数组合，含指标和"平原"标记
    """
    from runner.common.single_backtest import _run_single_backtest

    core_keys = _CORE_PARAMS.get(strategy_name, [])
    if not core_keys:
        logger.warning(f"{strategy_name}: 无核心参数定义，跳过")
        return pd.DataFrame()

    grid = _make_grid(strategy_params, core_keys, perturbation, steps)
    if not grid:
        logger.warning(f"{strategy_name}: 未生成有效网格")
        return pd.DataFrame()

    logger.info(
        f"[参数平原] {strategy_name} {symbol} "
        f"核心参数={core_keys} 网格={len(grid)} 扰动={perturbation*100:.0f}%"
    )

    rows: List[Dict[str, Any]] = []
    base_result = _run_single_backtest(
        symbol, strategy_name, strategy_params,
        entry_threshold=entry_threshold,
        full_start=full_start,
        test_start=test_start,
        initial_cash=initial_cash,
    )
    if base_result is None or base_result.get("total_return_pct", 0) == 0:
        logger.warning(f"{strategy_name} {symbol}: 基准回测无收益，无法做平原测试")
        return pd.DataFrame()

    base_ret = safe_float(base_result["total_return_pct"])
    base_sharpe = safe_float(base_result.get("sharpe_ratio", 0))

    for params in grid:
        result = _run_single_backtest(
            symbol, strategy_name, params,
            entry_threshold=entry_threshold,
            full_start=full_start,
            test_start=test_start,
            initial_cash=initial_cash,
        )
        if result is None:
            continue

        ret = safe_float(result["total_return_pct"])
        sharpe = safe_float(result.get("sharpe_ratio", 0))

        # 变化率
        ret_change = abs(ret - base_ret) / (abs(base_ret) + 1e-10)
        sharpe_change = abs(sharpe - base_sharpe) / (abs(base_sharpe) + 1e-10)
        max_change = max(ret_change, sharpe_change)

        row: Dict[str, Any] = {"symbol": symbol, "strategy": strategy_name}
        for k, v in params.items():
            row[f"param_{k}"] = v
        row["total_return_pct"] = round(ret, 2)
        row["sharpe_ratio"] = round(sharpe, 2)
        row["ret_change_pct"] = round(ret_change * 100, 1)
        row["sharpe_change_pct"] = round(sharpe_change * 100, 1)
        row["max_change_pct"] = round(max_change * 100, 1)
        row["is_plateau"] = max_change < variation_threshold
        rows.append(row)

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df

    # 标记平原区域
    n_plateau = int(result_df["is_plateau"].sum())
    logger.info(
        f"  → 平原点: {n_plateau}/{len(result_df)} "
        f"({n_plateau/len(result_df)*100:.0f}%)"
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"plateau_{strategy_name}_{sanitize_filename(symbol)}.csv"
        result_df.to_csv(path, index=False)

    return result_df


def run_parameter_plateau_batch(
    symbols: List[str],
    strategy_name: str,
    strategy_params: Dict[str, Any],
    perturbation: float = 0.20,
    steps: int = 5,
    variation_threshold: float = 0.15,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """批量品种参数平原测试。"""
    all_dfs: List[pd.DataFrame] = []
    for symbol in symbols:
        df = run_parameter_plateau(
            symbol=symbol,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            perturbation=perturbation,
            steps=steps,
            variation_threshold=variation_threshold,
            output_dir=output_dir,
        )
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()
    combined = pd.concat(all_dfs, ignore_index=True)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"plateau_{strategy_name}_batch.csv"
        combined.to_csv(path, index=False)

    return combined


def sanitize_filename(name: str) -> str:
    """文件名安全处理。"""
    return name.replace(".", "_").replace("-", "_")


__all__ = [
    "run_parameter_plateau",
    "run_parameter_plateau_batch",
    "_CORE_PARAMS",
]
