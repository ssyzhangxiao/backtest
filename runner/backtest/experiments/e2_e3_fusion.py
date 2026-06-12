"""
实验 E2 / E3：信号融合（等权 + 环境动态加权）。

两个实验共享 _run_weighted_fusion 实现：差异仅在
- use_execute_fusion（True=E3 动态加权；False=E2 等权）
- regime_history 是否保存（仅 E3）

委托 runner/backtest/runner.py 执行回测，runner/common/utils.py 与
runner/strategy/selector.py 处理工具和策略。
"""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from core.engine.pybroker_data_source import PyBrokerDataSource
from runner.backtest.runner import get_pybroker_runner, safe_run_backtest
from runner.common.utils import (
    format_metrics,
    handle_backtest_errors,
    sanitize_filename,
    save_csv,
    save_equity_curve,
)
from runner.strategy.selector import get_strategy_names


# ============================================
# 通用融合实验函数（E2 / E3 共享）
# ============================================


def _run_weighted_fusion(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
    use_dynamic: bool,
) -> pd.DataFrame:
    """
    等权 / 动态加权融合回测通用实现（E2 / E3 共享）。

    P1 整改（2026-06-07）：合并原 _run_equal_weight 与 _run_dynamic_weight，
    差异仅在 use_execute_fusion 与 regime_history 保存。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录
        use_dynamic: True=E3 动态加权；False=E2 等权

    Returns:
        汇总指标 DataFrame
    """
    if use_dynamic:
        label = "E3_动态权重"
        prefix = "e3"
    else:
        label = "E2_等权融合"
        prefix = "e2"
    logger.info(f"执行{label}实验")

    symbols: List[str] = config.get("symbols", [])
    strategy_names = get_strategy_names(config)
    bt_cfg = config["backtest"]
    all_results: List[Dict[str, Any]] = []

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        # 修复 per-symbol 隔离 bug：仅对当前品种做回测，不再用全 30 品种组合
        runner = get_pybroker_runner(
            data_source, config, strategies=strategy_names, target_symbols=[sym]
        )
        result = safe_run_backtest(
            runner,
            bt_cfg["full_start_date"],
            bt_cfg["full_end_date"],
            f"{label[:2]}_{sym}",
            use_execute_fusion=use_dynamic,
        )

        if result is None:
            all_results.append(
                {
                    "symbol": sym,
                    "strategy": None,
                    "experiment": label,
                    "error": "回测失败",
                }
            )
            continue

        m = format_metrics(result.metrics)
        result_row: Dict[str, Any] = {
            "symbol": sym,
            "strategy": None,
            "experiment": label,
            "error": None,
        }
        result_row.update(m)
        all_results.append(result_row)

        logger.info(
            f"  {label}: return={m.get('total_return_pct', 'N/A')} "
            f"sharpe={m.get('sharpe', 'N/A')}"
        )

        eq = result.equity_curve
        if eq is not None and not eq.empty:
            save_equity_curve(
                eq.assign(symbol=sym),
                output_dir,
                f"{prefix}_equity_{sanitize_filename(sym.replace('.', '_'))}",
            )

        if result.switch_log is not None and not result.switch_log.empty:
            save_csv(
                result.switch_log.assign(symbol=sym),
                output_dir
                / f"{prefix}_switch_log_{sanitize_filename(sym.replace('.', '_'))}.csv",
            )
            logger.info(f"  {sym}: 已保存 {len(result.switch_log)} 条调仓决策记录")

        # 仅动态加权（E3）保存 regime_history
        if (
            use_dynamic
            and getattr(result, "regime_history", None) is not None
            and not result.regime_history.empty
        ):
            save_csv(
                result.regime_history,
                output_dir
                / f"{prefix}_regime_{sanitize_filename(sym.replace('.', '_'))}.csv",
            )

    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    # filename 与历史一致（en 短码），不再用中文 label 段
    save_csv(
        df,
        output_dir
        / f"{prefix}_{('equal' if not use_dynamic else 'dynamic')}_weight_metrics.csv",
    )
    return df


# ============================================
# E2：等权信号融合
# ============================================


@handle_backtest_errors(return_value=pd.DataFrame())
def run_e2_equal_weight(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E2：等权信号融合回测。

    所有策略信号等权融合，单一品种组合。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        汇总指标 DataFrame
    """
    logger.info("E2：等权信号融合回测")
    return _run_weighted_fusion(data_source, config, output_dir, use_dynamic=False)


# ============================================
# E3：环境动态加权
# ============================================


@handle_backtest_errors(return_value=pd.DataFrame())
def run_e3_dynamic_weight(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E3：环境动态加权回测（execute 融合模式）。

    根据市场环境动态分配各策略权重。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        汇总指标 DataFrame
    """
    logger.info("E3：环境动态加权回测（execute 融合模式）")
    return _run_weighted_fusion(data_source, config, output_dir, use_dynamic=True)
