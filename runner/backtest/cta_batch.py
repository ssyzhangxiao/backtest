"""
CTA 6策略批量回测 — 全量 PyBroker 引擎版。

使用 PyBrokerBacktestRunner 执行完整回测（而非简化信号模拟），
逐品种×CTA策略调用 TqSDK/PyBroker 全链路。

用法:
    from runner.backtest.cta_batch import run_cta_batch
    results = run_cta_batch(data_source, raw_config, symbols, strategies)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from core.engine.pybroker_data_source import PyBrokerDataSource
from runner.backtest.runner import get_pybroker_runner, safe_run_backtest
from runner.common.utils import format_metrics

# CTA 6 策略规范名
CTA_STRATEGIES: List[str] = [
    "carry",
    "vol_mean_reversion",
    "donchian_breakout",
    "momentum_ma",
    "tsi_garch",
    "pair_trading",
]

# 默认 CTA 权重
CTA_WEIGHTS: Dict[str, float] = {
    "carry": 0.30,
    "vol_mean_reversion": 0.30,
    "donchian_breakout": 0.20,
    "momentum_ma": 0.10,
    "tsi_garch": 0.05,
    "pair_trading": 0.05,
}


def _make_cta_config(
    raw_config: Dict[str, Any],
    strategy_name: str,
) -> Dict[str, Any]:
    """构建 CTA 模式的配置 dict（注入 signal_mode=cta）。

    基于原始配置深拷贝，修改 backtest. 段的信号模式，
    确保 PyBrokerBacktestRunner 创建 SignalAbstractionLayer 并运行 CTA 模式。

    Args:
        raw_config: 原始配置字典
        strategy_name: CTA 策略名

    Returns:
        修改后的配置字典
    """
    cfg = deepcopy(raw_config)
    bt = cfg.setdefault("backtest", {})
    bt["use_signal_abstraction"] = True
    bt["signal_mode"] = "cta"
    # 限定只注册当前 CTA 策略对应的因子
    cfg["strategies"] = [{"name": strategy_name}]
    return cfg


def run_cta_batch(
    data_source: PyBrokerDataSource,
    raw_config: Dict[str, Any],
    symbols: List[str],
    strategies: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    CTA 6 策略批量回测 — 全量 PyBroker 引擎版。

    逐品种×策略调用 PyBrokerBacktestRunner 执行完整回测（含滑点、手续费、风控），
    结果基于 PyBroker 计算的真实绩效指标。

    Args:
        data_source: PyBrokerDataSource 实例
        raw_config: 原始配置字典
        symbols: 品种列表
        strategies: CTA 策略名列表（默认 CTA_STRATEGIES）
        start_date: 回测起始（默认 raw_config.backtest.full_start_date）
        end_date: 回测结束（默认 raw_config.backtest.full_end_date）

    Returns:
        DataFrame：每行一个品种×策略组合的完整绩效指标
    """
    strategies = strategies or CTA_STRATEGIES
    bt_cfg = raw_config.get("backtest", {})
    start_date = start_date or bt_cfg.get("full_start_date", "2020-01-01")
    end_date = end_date or bt_cfg.get("full_end_date", "2024-12-31")

    results: List[Dict[str, Any]] = []
    total = len(symbols) * len(strategies)
    count = 0
    ok = 0

    for sym in symbols:
        for strat in strategies:
            count += 1
            cfg = _make_cta_config(raw_config, strat)
            runner = get_pybroker_runner(
                data_source, cfg, strategies=[strat], target_symbols=[sym],
            )
            result = safe_run_backtest(
                runner,
                start_date,
                end_date,
                f"CTA_{sym}_{strat}",
            )

            if result is not None:
                ok += 1
                m = format_metrics(result.metrics)
                row: Dict[str, Any] = {
                    "symbol": sym,
                    "strategy": strat,
                    "status": "OK",
                    "error": None,
                }
                row.update(m)
                logger.info(
                    f"[OK] ({count}/{total}) {sym}/{strat}: "
                    f"return={m.get('total_return_pct', 'N/A')} "
                    f"sharpe={m.get('sharpe', 'N/A')}"
                )
            else:
                row = {
                    "symbol": sym,
                    "strategy": strat,
                    "status": "FAIL",
                    "error": "回测执行失败",
                }
                logger.warning(f"[FAIL] ({count}/{total}) {sym}/{strat}: 回测失败")

            results.append(row)

    df = pd.DataFrame(results)
    logger.info(f"CTA 全量 PyBroker 回测完成: {ok}/{total} 成功")
    return df


def summarize_cta_results(df: pd.DataFrame) -> Dict[str, Any]:
    """汇总 CTA 回测结果。

    Args:
        df: run_cta_batch 返回的 DataFrame

    Returns:
        summary dict
    """
    ok_df = df[df["status"] == "OK"].copy()

    # 各策略平均
    strat_avg = {}
    for s in CTA_STRATEGIES:
        sub = ok_df[ok_df["strategy"] == s]
        if len(sub) > 0:
            strat_avg[s] = {
                "avg_return_pct": float(sub["total_return_pct"].mean()),
                "avg_sharpe": float(sub["sharpe"].mean()),
                "ok_count": int(len(sub)),
                "total_count": len(sub) + len(df[(df["strategy"] == s) & (df["status"] != "OK")]),
            }

    # 加权组合收益
    weighted_return = 0.0
    for s, w in CTA_WEIGHTS.items():
        if s in strat_avg:
            weighted_return += w * strat_avg[s]["avg_return_pct"]

    total = len(df)
    ok = len(ok_df)
    summary = {
        "total_combos": total,
        "succeeded": ok,
        "failed": total - ok,
        "weighted_composite_return_pct": round(weighted_return, 2),
        "strategy_summary": strat_avg,
    }
    return summary


__all__ = ["run_cta_batch", "summarize_cta_results", "CTA_STRATEGIES", "CTA_WEIGHTS"]
