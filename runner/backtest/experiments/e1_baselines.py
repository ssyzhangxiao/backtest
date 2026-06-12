"""
实验 E1：单策略多品种基线回测。

对每个品种 × 每个策略单独运行回测，汇总绩效指标，作为后续融合实验（E2-E5）
和稳健性验证（E6-E9）的对比基线。

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
# E1：单策略基线
# ============================================


@handle_backtest_errors(return_value=pd.DataFrame())
def run_e1_single_strategy_baselines(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E1：单策略基线回测。

    对每个品种 × 每个策略单独运行回测，汇总绩效指标。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        汇总指标 DataFrame
    """
    logger.info("E1：单策略基线回测")
    symbols: List[str] = config.get("symbols", [])
    strategy_names = get_strategy_names(config)
    bt_cfg = config["backtest"]
    all_results: List[Dict[str, Any]] = []

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        for sname in strategy_names:
            # 修复 per-symbol 隔离 bug：仅对当前品种做回测
            runner = get_pybroker_runner(
                data_source, config, strategies=[sname], target_symbols=[sym]
            )
            result = safe_run_backtest(
                runner,
                bt_cfg["full_start_date"],
                bt_cfg["full_end_date"],
                f"E1_{sym}_{sname}",
            )

            if result is not None:
                m = format_metrics(result.metrics)
                result_row: Dict[str, Any] = {
                    "symbol": sym,
                    "strategy": sname,
                    "experiment": "E1_单策略基线",
                    "error": None,
                }
                result_row.update(m)
                all_results.append(result_row)
                logger.info(
                    f"  {sname}: return={m.get('total_return_pct', 'N/A')} "
                    f"sharpe={m.get('sharpe', 'N/A')} "
                    f"max_dd={m.get('max_drawdown_pct', 'N/A')}"
                )

                eq = result.equity_curve
                if eq is not None and not eq.empty:
                    save_equity_curve(
                        eq.assign(symbol=sym, strategy=sname),
                        output_dir,
                        f"e1_equity_{sanitize_filename(sym.replace('.', '_'))}_{sname}",
                    )
            else:
                all_results.append(
                    {
                        "symbol": sym,
                        "strategy": sname,
                        "experiment": "E1_单策略基线",
                        "error": "回测失败",
                    }
                )

    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    save_csv(df, output_dir / "e1_baseline_metrics.csv")
    return df
