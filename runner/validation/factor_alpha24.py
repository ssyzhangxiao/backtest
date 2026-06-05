"""
AlphaFutures24 因子IC/IR验证。

对24个商品期货Alpha因子进行逐个IC/IR统计测试，
筛选出符合规则9（IC>0.03且IR>0.5）的有效因子。

委托 core/factors/AlphaFutures24 做因子计算。
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.factors import AlphaFutures24, AlphaFuturesConfig
from core.strategy_registry import StrategyLibrary
from runner.common.utils import save_csv


def factor_alpha24_screening(
    data,
    config: BacktestConfig,
    lib: Optional[StrategyLibrary] = None,
    output_dir: Optional[Path] = None,
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    do_winsorize: bool = True,
) -> Dict[str, Any]:
    """
    对AlphaFutures24全部24个因子进行IC/IR统计测试。

    规则9要求：IC > 0.03 且 IR > 0.5 的因子方可保留。

    Args:
        data: 数据源（PyBrokerDataSource）
        config: 回测配置
        lib: 策略库（本方法不使用，保留接口一致）
        output_dir: 输出目录
        best_params: 最优参数（本方法不使用，保留接口一致）
        do_winsorize: 是否对因子值做缩尾后处理

    Returns:
        {
            results_df: 所有品种的测试结果,
            summary_df: 因子汇总,
            pass_count: 通过规则9的因子数,
            best_factors: 通过规则9的因子列表,
        }
    """
    logger.info("=" * 60)
    logger.info("AlphaFutures24 因子IC/IR验证")
    logger.info("=" * 60)

    calc = AlphaFutures24(AlphaFuturesConfig())
    symbols = config.symbols
    all_rows = []

    for symbol in symbols:
        try:
            ohlcv = data.query(
                config.train_start, config.test_end, symbols=[symbol]
            )
            if ohlcv is None or len(ohlcv) < 100:
                logger.warning(f"  {symbol}: 数据不足，跳过")
                continue

            ohlcv = ohlcv.sort_values("date").reset_index(drop=True)
            close = ohlcv["close"].values.astype(float)
            high = ohlcv["high"].values.astype(float)
            low = ohlcv["low"].values.astype(float)
            open_price = ohlcv["open"].values.astype(float)
            oi = (
                ohlcv["open_interest"].values.astype(float)
                if "open_interest" in ohlcv.columns
                else None
            )
            if oi is None:
                logger.warning(f"  {symbol}: 无持仓量数据，跳过")
                continue

            # 计算24个因子
            factors = calc.compute_all(
                close=close,
                open_price=open_price,
                high=high,
                low=low,
                open_interest=oi,
            )

            if do_winsorize:
                factors = calc.post_process(factors, do_winsorize=True)

            # 前瞻收益（5日）
            forward_ret = np.full_like(close, np.nan, dtype=float)
            fwd_period = 5
            forward_ret[:-fwd_period] = (
                close[fwd_period:] - close[:-fwd_period]
            ) / close[:-fwd_period]

            # 逐个因子计算IC
            for fname, fvalues in factors.items():
                valid = ~(np.isnan(fvalues) | np.isnan(forward_ret))
                if valid.sum() < 30:
                    continue
                fv = fvalues[valid]
                fr = forward_ret[valid]
                ic = np.corrcoef(fv, fr)[0, 1] if len(fv) > 2 else np.nan

                # 滚动IC（60天窗口）
                rolling_ic = (
                    pd.Series(fv, dtype=float)
                    .rolling(60, min_periods=30)
                    .corr(pd.Series(fr, dtype=float))
                    .dropna()
                    .values
                )

                mean_ic = np.nanmean(rolling_ic) if len(rolling_ic) > 10 else ic
                std_ic = np.nanstd(rolling_ic) if len(rolling_ic) > 10 else np.nan
                ir = mean_ic / std_ic if std_ic and std_ic > 0 else 0.0

                all_rows.append({
                    "symbol": symbol,
                    "factor": fname,
                    "mean_ic": round(float(mean_ic), 6),
                    "std_ic": round(float(std_ic), 6) if not np.isnan(std_ic) else np.nan,
                    "ir": round(float(ir), 4),
                    "pass_rule9": abs(mean_ic) > 0.03 and abs(ir) > 0.5,
                })

        except Exception as e:
            logger.warning(f"  {symbol}: 因子计算失败 - {e}")

    if not all_rows:
        logger.warning("无有效测试结果")
        return {
            "results_df": pd.DataFrame(),
            "summary_df": pd.DataFrame(),
            "pass_count": 0,
            "best_factors": [],
        }

    df = pd.DataFrame(all_rows)

    # 因子汇总：多品种平均
    summary = df.groupby("factor").agg({
        "mean_ic": "mean",
        "ir": "mean",
        "pass_rule9": "mean",
    }).reset_index()
    summary["abs_ic"] = summary["mean_ic"].abs()
    summary = summary.sort_values("abs_ic", ascending=False)

    # 通过规则9的因子
    pass_count = int((summary["pass_rule9"] > 0.5).sum())
    best_factors = summary[summary["pass_rule9"] > 0.5]["factor"].tolist()

    logger.info(f"\n因子验证完成: {pass_count}/{len(summary)} 通过规则9")
    if best_factors:
        logger.info(f"  有效因子: {best_factors}")
    else:
        logger.warning("  无因子通过规则9，建议检查数据质量或调整阈值")

    # 保存结果
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_csv(df, output_dir / "factor_alpha24_results.csv")
        save_csv(summary, output_dir / "factor_alpha24_summary.csv")
        logger.info(f"  结果已保存到: {output_dir}")

    return {
        "results_df": df,
        "summary_df": summary,
        "pass_count": pass_count,
        "best_factors": best_factors,
    }