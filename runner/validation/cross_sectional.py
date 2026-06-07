"""
多策略横截面打分验证。

对 5 子策略横截面打分模式进行验证，
包括各子策略信号分布、综合信号有效性等。

P0 整改（2026-06-07）：core/strategies/ 已整体删除，
子策略信号统一由路径A因子聚合器提供，不再实例化 CrossSectionalStrategy。

P2 整改：形参 data → data_source 与其他验证函数统一；
综合信号（多子策略均值）增加 IC 评估（委托 FactorEvaluator）。
"""
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.factors.alpha_futures.sub_strategy_aggregator import (
    compute_sub_strategy_scores_from_ohlcv,
)
from core.factors.alpha_futures.config import AlphaFuturesConfig
from core.factors.factor_evaluator import FactorEvaluator
from core.config.strategy_profiles import StrategyLibrary
from runner.common.utils import save_csv


def cross_sectional_validation(
    data_source,
    config: BacktestConfig,
    lib: StrategyLibrary,
    output_dir: Path,
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    cross_sectional: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    多策略横截面打分验证。

    验证各子策略信号质量、信号间相关性、综合信号分布与 IC。
    子策略信号通过路径A（sub_strategy_aggregator）计算。

    Args:
        data_source: 数据源
        config: 回测配置
        lib: 策略库
        output_dir: 输出目录
        best_params: 最优参数
        **kwargs: 额外参数

    Returns:
        验证结果字典
    """
    logger.info("=" * 60)
    logger.info("多策略横截面打分验证")
    logger.info("=" * 60)

    results: Dict[str, Any] = {
        "mode": "cross_sectional",
        "sub_strategies": {},
        "composite": {},
    }

    if lib is None:
        lib = StrategyLibrary()

    # 路径A：因子聚合器统一计算 5 子策略得分
    factor_config = AlphaFuturesConfig()
    evaluator = FactorEvaluator(forward_period=5, ic_window=60, min_observations=30)

    # 获取各子策略信号
    for symbol in config.symbols[:3]:  # 取前3个品种做验证
        try:
            ohlcv = data_source.query(
                config.train_start, config.test_end, symbols=[symbol]
            )
            if ohlcv is None or len(ohlcv) < 100:
                continue

            ohlcv = ohlcv.sort_values("date").reset_index(drop=True)

            scored = compute_sub_strategy_scores_from_ohlcv(ohlcv, config=factor_config)
            signals: Dict[str, np.ndarray] = {
                col: scored[col].fillna(0.0).to_numpy()
                for col in scored.columns
            }
            results["sub_strategies"][symbol] = {
                name: {
                    "mean": float(np.nanmean(sig)),
                    "std": float(np.nanstd(sig)),
                    "sharpe": float(np.nanmean(sig) / (np.nanstd(sig) + 1e-8)),
                }
                for name, sig in signals.items()
            }

            # 复合信号：5 子策略横截面均值的归一化
            sub_cols = [c for c in scored.columns
                        if c in ("trend", "term_structure", "mean_reversion",
                                 "vol_breakout", "composite_resonance")]
            if sub_cols:
                composite_score = scored[sub_cols].mean(axis=1).fillna(0.0).to_numpy()
                # 前瞻收益（5 日）
                close = ohlcv["close"].values.astype(float)
                fwd_ret = np.full_like(close, np.nan, dtype=float)
                fwd_ret[:-5] = (close[5:] - close[:-5]) / close[:-5]
                dates_arr = pd.to_datetime(ohlcv["date"]).values

                eval_res = evaluator.evaluate(
                    factor_name="composite",
                    factor_scores=composite_score,
                    forward_returns=fwd_ret,
                    dates=dates_arr,
                )
                results["composite"][symbol] = {
                    "ic_mean": round(eval_res.ic_mean, 6),
                    "ic_std": round(eval_res.ic_std, 6),
                    "ir": round(eval_res.ir, 4),
                    "ic_decay_rate": round(eval_res.ic_decay_rate, 4),
                    "is_valid": eval_res.is_valid,
                }
                logger.info(
                    f"  {symbol}: {len(signals)}个子策略信号 + "
                    f"复合信号 IC={eval_res.ic_mean:.4f} IR={eval_res.ir:.4f}"
                )
            else:
                logger.info(f"  {symbol}: {len(signals)}个子策略信号已计算（路径A）")

        except Exception as e:
            logger.warning(f"横截面验证 {symbol} 失败: {e}")

    logger.info(
        f"横截面验证完成: {len(results['sub_strategies'])}个品种, "
        f"{len(results['composite'])}个复合信号"
    )
    return results
