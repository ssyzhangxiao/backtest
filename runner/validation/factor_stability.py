"""
因子IC稳定性分析模块。

对比训练期和验证期的因子IC变化，
委托 core/ext/factors/evaluator.py 的 FactorEvaluator。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.ext.factors.evaluator import FactorEvaluator, FactorEvalResult
from core.engine.pybroker_data_source import PyBrokerDataSource
from core.config.strategy_profiles import StrategyLibrary
from runner.common.utils import is_valid_number, save_csv
from core.ext.factors.alpha_futures.sub_strategy_aggregator import (
    compute_sub_strategy_scores_from_ohlcv,
)

_DEFAULT_FACTOR_NAMES = [
    "trend", "term_structure", "mean_reversion",
    "vol_breakout", "composite_resonance",
]


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
    """因子IC稳定性分析：对比训练期和验证期的因子IC变化。

    对每个品种独立计算因子得分，使用 FactorEvaluator 评估 IC/IR，
    输出IC稳定性CSV和汇总。

    Args:
        data_source: 数据源
        config: 回测配置
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
    logger.info("因子IC稳定性分析（FactorEvaluator）")
    logger.info(f"  因子: {factor_names}")
    logger.info(f"  训练期: {train_start} ~ {train_end}")
    logger.info(f"  验证期: {test_start} ~ {test_end}")
    logger.info("=" * 60)

    evaluator = FactorEvaluator(forward_period=5, ic_window=60, min_observations=30)

    all_results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    for symbol in symbols:
        logger.info(f"  分析品种: {symbol}")
        try:
            sym_result = _analyze_single_symbol(
                symbol=symbol,
                data_source=data_source,
                factor_names=factor_names,
                evaluator=evaluator,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
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
                f"status={row_data['decay_status']}"
            )

    return {"summary_rows": summary_rows, "details": all_results}


def _analyze_single_symbol(
    symbol: str,
    data_source: PyBrokerDataSource,
    factor_names: List[str],
    evaluator: FactorEvaluator,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> Optional[Dict[str, Any]]:
    """分析单个品种的因子IC稳定性。"""
    sym_df = data_source.query(train_start, test_end, symbols=[symbol])
    if sym_df is None or len(sym_df) < 60:
        logger.warning(f"    {symbol}: 数据不足，跳过")
        return None

    # 委托公共因子得分计算
    scored = compute_sub_strategy_scores_from_ohlcv(sym_df)

    # 全期评估
    factor_scores_dict = {}
    forward_returns = None
    for name in factor_names:
        if name in scored.columns:
            vals = scored[name].values.astype(float)
            factor_scores_dict[name] = vals
    if "forward_return" in scored.columns:
        forward_returns = scored["forward_return"].values.astype(float)

    if not factor_scores_dict or forward_returns is None:
        logger.warning(f"    {symbol}: 因子得分或前瞻收益缺失，跳过")
        return None

    # 全期评估
    full_results = evaluator.evaluate_batch(factor_scores_dict, forward_returns)

    # 训练期/验证期分别评估
    train_scored = scored[
        (scored["date"] >= train_start) & (scored["date"] <= train_end)
    ]
    test_scored = scored[
        (scored["date"] >= test_start) & (scored["date"] <= test_end)
    ]

    train_results = _evaluate_period(train_scored, factor_names, evaluator)
    test_results = _evaluate_period(test_scored, factor_names, evaluator)

    # 构建汇总行
    summary_rows = []
    for name in factor_names:
        result = full_results.get(name)
        if result is None:
            continue
        decay_status = _classify_decay(result, train_results.get(name), test_results.get(name))
        summary_rows.append(
            {
                "symbol": symbol,
                "factor": name,
                "mean_ic": round(result.ic_mean, 6),
                "std_ic": round(result.ic_std, 6),
                "ir": round(result.ir, 4),
                "current_ic": round(result.ic_mean, 6),
                "current_weight": 0.0,  # 权重由外部分配
                "decay_status": decay_status,
            }
        )

    # 输出衰减对比
    for name in factor_names:
        train_r = train_results.get(name)
        test_r = test_results.get(name)
        train_mean = train_r.ic_mean if train_r else 0.0
        test_mean = test_r.ic_mean if test_r else 0.0
        ic_drop = (
            (test_mean - train_mean) / (abs(train_mean) + 1e-10)
            if abs(train_mean) > 1e-6
            else 0.0
        )
        decay_status = _classify_decay(
            full_results.get(name), train_r, test_r
        )
        logger.info(
            f"    {name}: train_IC={train_mean:.4f}, "
            f"test_IC={test_mean:.4f}, "
            f"衰减={ic_drop:.1%}, "
            f"status={decay_status}"
        )

    details = {
        "full_results": {k: _result_to_dict(v) for k, v in full_results.items()},
        "train_ic": {k: _result_to_dict(v) for k, v in train_results.items()},
        "test_ic": {k: _result_to_dict(v) for k, v in test_results.items()},
    }

    return {"rows": summary_rows, "details": details}


def _evaluate_period(
    scored: pd.DataFrame,
    factor_names: List[str],
    evaluator: FactorEvaluator,
) -> Dict[str, FactorEvalResult]:
    """评估单个时期的因子IC。"""
    factor_scores_dict = {}
    forward_returns = None
    for name in factor_names:
        if name in scored.columns:
            vals = scored[name].values.astype(float)
            factor_scores_dict[name] = vals
    if "forward_return" in scored.columns:
        forward_returns = scored["forward_return"].values.astype(float)

    if not factor_scores_dict or forward_returns is None:
        return {}

    return evaluator.evaluate_batch(factor_scores_dict, forward_returns)


def _classify_decay(
    full_result: Optional[FactorEvalResult],
    train_result: Optional[FactorEvalResult],
    test_result: Optional[FactorEvalResult],
) -> str:
    """根据训练期/验证期IC变化分类衰减状态。"""
    if full_result is None:
        return "unknown"

    abs_ic = abs(full_result.ic_mean)

    if abs_ic < 0.01:
        return "dead"
    if abs_ic < 0.03:
        return "decaying"

    # 对比训练期和验证期
    if train_result and test_result:
        train_ic = abs(train_result.ic_mean)
        test_ic = abs(test_result.ic_mean)
        if train_ic > 1e-6:
            drop = (train_ic - test_ic) / train_ic
            if drop > 0.5:
                return "decaying"
            if drop > 0.3:
                return "warning"

    if full_result.is_valid:
        return "healthy"
    return "warning"


def _result_to_dict(result: Optional[FactorEvalResult]) -> Dict[str, Any]:
    """将 FactorEvalResult 转为字典。"""
    if result is None:
        return {}
    return {
        "ic_mean": result.ic_mean,
        "ic_std": result.ic_std,
        "ir": result.ir,
        "is_valid": result.is_valid,
        "reject_reason": result.reject_reason,
        "ic_decay_rate": result.ic_decay_rate,
    }
