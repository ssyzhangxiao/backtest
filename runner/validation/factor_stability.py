"""
因子IC稳定性分析模块。

对比训练期和验证期的因子IC变化，
委托 core/engine/rolling_ic.py 和 core/engine/factor_decay.py。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.engine.rolling_ic import RollingICWeightEngine, RollingICConfig
from core.engine.factor_decay import (
    FactorDecayMonitor,
    FactorDecayConfig,
    DecayStatus,
)
from core.engine.pybroker_data_source import PyBrokerDataSource
from core.config.strategy_profiles import StrategyLibrary
from runner.common.utils import is_valid_number, save_csv
from core.factors.alpha_futures.sub_strategy_aggregator import (
    compute_sub_strategy_scores_from_ohlcv,
)

_DEFAULT_FACTOR_NAMES = ["trend", "term_structure", "mean_reversion", "vol_breakout", "composite_resonance"]


def factor_ic_stability_analysis(
    data_source: PyBrokerDataSource,
    config: BacktestConfig,
    lib: StrategyLibrary,
    output_dir: Path,
    best_params: Optional[Dict[str, Any]] = None,
    cross_sectional: bool = False,
    factor_names: List[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    因子IC稳定性分析：对比训练期和验证期的因子IC变化。

    对每个品种独立计算因子得分、滚动IC和衰减状态，
    输出IC稳定性CSV和汇总图表。

    委托 core/engine/rolling_ic.py 和 core/engine/factor_decay.py，
    不重复实现IC计算逻辑。

    Args:
        data_source: 数据源
        config: 回测配置（BacktestConfig）
        output_dir: 输出目录
        factor_names: 待分析因子名称列表

    Returns:
        {summary_rows: [...], details: {品种: {...}}}
    """
    if factor_names is None:
        factor_names = _DEFAULT_FACTOR_NAMES

    train_start = config.train_start
    train_end = config.train_end
    test_start = config.test_start
    test_end = config.test_end
    symbols = config.symbols

    logger.info("=" * 60)
    logger.info("因子IC稳定性分析（滚动IC + 衰减监控）")
    logger.info(f"  因子: {factor_names}")
    logger.info(f"  训练期: {train_start} ~ {train_end}")
    logger.info(f"  验证期: {test_start} ~ {test_end}")
    logger.info("=" * 60)

    # IC配置
    ic_config = RollingICConfig(
        window=60,
        forward_period=5,
        ema_alpha=0.1,
        min_observations=30,
    )
    decay_config = FactorDecayConfig(
        trend_window=40,
        ic_healthy_threshold=0.03,
        ic_dead_threshold=0.01,
        max_consecutive_decline=5,
        decay_slope_threshold=-0.001,
    )

    all_results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    for symbol in symbols:
        logger.info(f"  分析品种: {symbol}")
        try:
            sym_result = _analyze_single_symbol(
                symbol,
                ds,
                factor_names,
                ic_config,
                decay_config,
                train_start,
                train_end,
                test_start,
                test_end,
            )
            if sym_result is not None:
                all_results[symbol] = sym_result["details"]
                summary_rows.extend(sym_result["rows"])
        except Exception as e:
            logger.error(f"    {symbol} 因子分析失败: {e}")

    # 汇总CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        save_csv(summary_df, output_dir / "factor_ic_stability.csv")

        logger.info("\n  因子IC稳定性汇总:")
        for row_data in summary_rows:
            logger.info(
                f"    {row_data['symbol']}/{row_data['factor']}: "
                f"mean_IC={row_data['mean_ic']:.4f}, IR={row_data['ir']:.2f}, "
                f"weight={row_data['current_weight']:.4f}, "
                f"status={row_data['decay_status']}"
            )

    return {"summary_rows": summary_rows, "details": all_results}


def _analyze_single_symbol(
    symbol: str,
    data_source: PyBrokerDataSource,
    factor_names: List[str],
    ic_config: RollingICConfig,
    decay_config: FactorDecayConfig,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> Dict[str, Any]:
    """
    分析单个品种的因子IC稳定性。

    Args:
        symbol: 品种代码
        data_source: 数据源
        factor_names: 因子名称列表
        ic_config: 滚动IC配置
        decay_config: 衰减监控配置
        train_start: 训练期开始
        train_end: 训练期结束
        test_start: 验证期开始
        test_end: 验证期结束

    Returns:
        {rows: [...], details: {...}} 或 None
    """
    sym_df = data_source.query(train_start, test_end, symbols=[symbol])
    if sym_df is None or len(sym_df) < 60:
        logger.warning(f"    {symbol}: 数据不足，跳过")
        return None

    # 委托公共因子得分计算（P0 整改：使用新因子引擎聚合，废弃 basic_factors）
    scored = compute_sub_strategy_scores_from_ohlcv(sym_df)

    # 全期滚动IC + 衰减监控
    ic_engine = RollingICWeightEngine(ic_config)
    decay_monitor = FactorDecayMonitor(decay_config)

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

        ic_engine.update(factor_scores, forward_ret)
        current_ic = ic_engine.current_ic
        for name, ic_val in current_ic.items():
            if is_valid_number(ic_val):
                decay_monitor.update(name, ic_val)

    decay_monitor.check_decay()
    ic_summary = ic_engine.get_ic_summary()
    final_weights = ic_engine.get_dynamic_weights()

    # 构建汇总行
    summary_rows = []
    for name, stats in ic_summary.items():
        status = decay_monitor.current_status.get(name, DecayStatus.HEALTHY).value
        summary_rows.append(
            {
                "symbol": symbol,
                "factor": name,
                "mean_ic": round(stats.get("mean", 0.0), 6),
                "std_ic": round(stats.get("std", 0.0), 6),
                "ir": round(stats.get("ir", 0.0), 4),
                "current_ic": round(stats.get("current", 0.0), 6),
                "current_weight": round(final_weights.get(name, 0.0), 4),
                "decay_status": status,
            }
        )

    # 分别统计训练期和验证期IC
    train_ic_summary, test_ic_summary = _compute_period_ic(
        scored,
        factor_names,
        ic_config,
        train_start,
        train_end,
        test_start,
        test_end,
    )

    # 输出衰减对比
    for name in factor_names:
        train_mean = train_ic_summary.get(name, {}).get("mean", 0.0)
        test_mean = test_ic_summary.get(name, {}).get("mean", 0.0)
        ic_drop = (
            (test_mean - train_mean) / (abs(train_mean) + 1e-10)
            if abs(train_mean) > 1e-6
            else 0.0
        )
        logger.info(
            f"    {name}: train_IC={train_mean:.4f}, "
            f"test_IC={test_mean:.4f}, "
            f"衰减={ic_drop:.1%}, "
            f"status={decay_monitor.current_status.get(name, DecayStatus.HEALTHY).value}"
        )

    details = {
        "ic_summary": ic_summary,
        "final_weights": final_weights,
        "decay_status": decay_monitor.current_status,
        "train_ic": train_ic_summary,
        "test_ic": test_ic_summary,
    }

    return {"rows": summary_rows, "details": details}


def _compute_period_ic(
    scored: pd.DataFrame,
    factor_names: List[str],
    ic_config: RollingICConfig,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> tuple:
    """
    分别计算训练期和验证期的因子IC。

    Args:
        scored: 含因子得分的 DataFrame
        factor_names: 因子名称列表
        ic_config: 滚动IC配置
        train_start: 训练期开始
        train_end: 训练期结束
        test_start: 验证期开始
        test_end: 验证期结束

    Returns:
        (train_ic_summary, test_ic_summary) 元组
    """
    train_scored = scored[
        (scored["date"] >= train_start) & (scored["date"] <= train_end)
    ]
    test_scored = scored[(scored["date"] >= test_start) & (scored["date"] <= test_end)]

    train_ic_engine = RollingICWeightEngine(ic_config)
    test_ic_engine = RollingICWeightEngine(ic_config)

    for _, row_data in train_scored.iterrows():
        fwd = float(row_data["forward_return"])
        if not is_valid_number(fwd):
            continue
        fscores = {
            name: float(row_data.get(name, 0.0))
            for name in factor_names
            if is_valid_number(row_data.get(name, 0.0))
        }
        if fscores:
            train_ic_engine.update(fscores, fwd)

    for _, row_data in test_scored.iterrows():
        fwd = float(row_data["forward_return"])
        if not is_valid_number(fwd):
            continue
        fscores = {
            name: float(row_data.get(name, 0.0))
            for name in factor_names
            if is_valid_number(row_data.get(name, 0.0))
        }
        if fscores:
            test_ic_engine.update(fscores, fwd)

    return train_ic_engine.get_ic_summary(), test_ic_engine.get_ic_summary()
