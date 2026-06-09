"""
因子6项复核验证。

委托 core/factors/alpha_futures/factor_pipeline.py 的 FactorPipeline，
执行数据存活率、缺失值占比、异常值抵抗、参数敏感性、
因子正交性、时序稳定性共6项复核。
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.config.strategy_profiles import StrategyLibrary
from core.factors.alpha_futures.factor_pipeline import FactorPipeline
from core.factors.alpha_futures_24 import AlphaFuturesConfig
from runner.common.utils import save_csv


def factor_review_validation(
    data_source,
    config: BacktestConfig,
    lib: StrategyLibrary,
    output_dir: Path,
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    cross_sectional: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    因子6项复核：对全部已注册因子执行质量检查。

    Args:
        data_source: 数据源
        config: 回测配置
        lib: 策略库（未使用，保持接口一致）
        output_dir: 输出目录
        best_params: 最优参数（未使用）
        **kwargs: 额外参数

    Returns:
        {symbol: PipelineResult} 字典
    """
    logger.info("=" * 60)
    logger.info("因子复核: 6项质量检查")
    logger.info("=" * 60)

    results = {}
    af_cfg = AlphaFuturesConfig()
    symbols = config.symbols

    for symbol in symbols:
        try:
            ohlcv = data_source.query(
                config.train_start, config.test_end, symbols=[symbol]
            )
            if ohlcv is None or len(ohlcv) < 100:
                logger.warning(f"  {symbol}: 数据不足，跳过")
                continue

            ohlcv = ohlcv.sort_values("date").reset_index(drop=True)
            close = ohlcv["close"].values.astype(float)
            forward_ret = np.zeros(len(close), dtype=float)
            fwd = 5
            forward_ret[:-fwd] = (close[fwd:] - close[:-fwd]) / close[:-fwd]

            raw_data = {
                "close": close,
                "open_price": ohlcv["open"].values.astype(float),
                "high": ohlcv["high"].values.astype(float),
                "low": ohlcv["low"].values.astype(float),
                "open_interest": (
                    ohlcv["open_interest"].values.astype(float)
                    if "open_interest" in ohlcv.columns
                    else np.zeros(len(close))
                ),
            }

            pipeline = FactorPipeline(af_cfg)
            pipe_result = pipeline.run(raw_data, forward_ret, run_review=True)
            results[symbol] = pipe_result

            # 输出复核报告摘要
            if pipe_result.review_report:
                stats = pipe_result.review_report.summary_stats
                logger.info(
                    f"  {symbol}: 保留{stats.get('保留',0)} "
                    f"降级{stats.get('降级',0)} "
                    f"待优化{stats.get('待优化',0)} "
                    f"剔除{stats.get('剔除',0)}"
                )
            else:
                logger.info(f"  {symbol}: 复核完成（无报告）")

        except Exception as e:
            logger.warning(f"因子复核 {symbol} 失败: {e}")

    # 保存报告
    if output_dir and results:
        output_dir = Path(output_dir)
        _save_review_report(results, output_dir)

    return results


def _save_review_report(results: dict, output_dir: Path) -> None:
    """汇总保存复核报告为CSV。"""
    rows = []
    for symbol, pipe_result in results.items():
        if not pipe_result.review_report:
            continue
        stats = pipe_result.review_report.summary_stats
        stats["symbol"] = symbol
        rows.append(stats)

    if rows:
        df = pd.DataFrame(rows)
        save_csv(df, output_dir / "factor_review_summary.csv")
        logger.info(f"复核报告已保存: {output_dir / 'factor_review_summary.csv'}")