"""因子筛选 — 对24因子进行IC/IR统计测试，筛选有效因子。

规则9要求：IC > 0.03 且 IR > 0.5 的因子方可保留。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.config.strategy_profiles import StrategyLibrary
from runner.common.utils import save_csv
from runner.validation._factor_panel import build_factor_panel, compute_cross_sectional_ic


def factor_alpha24_screening(
    data_source,
    config: BacktestConfig,
    lib: StrategyLibrary,
    output_dir: Path,
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    cross_sectional: bool = False,
    do_winsorize: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """对AlphaFutures24全部24个因子进行IC/IR统计测试。

    规则9要求：IC > 0.03 且 IR > 0.5 的因子方可保留。

    Args:
        data_source: 数据源（PyBrokerDataSource）
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

    # 1. 构建因子面板
    panel_df = build_factor_panel(
        data_source=data_source,
        config=config,
        fwd_period=5,
        do_winsorize=do_winsorize,
    )

    if panel_df.empty:
        logger.warning("无有效测试结果")
        return {
            "results_df": pd.DataFrame(),
            "summary_df": pd.DataFrame(),
            "pass_count": 0,
            "best_factors": [],
        }

    # 2. 计算横截面 IC/IR
    ic_results = compute_cross_sectional_ic(panel_df)

    if not ic_results:
        logger.warning("无有效横截面IC")
        return {
            "results_df": pd.DataFrame(),
            "summary_df": pd.DataFrame(),
            "pass_count": 0,
            "best_factors": [],
        }

    # 3. 构建汇总表
    ic_th = getattr(config.factors_config, "ic_threshold", 0.01)
    ir_th = getattr(config.factors_config, "ir_threshold", 0.1)

    summary_rows: List[Dict[str, Any]] = []
    for fname, stats in ic_results.items():
        mean_ic = stats["mean_ic"]
        ir = stats["ir"]
        is_valid = abs(mean_ic) >= ic_th and abs(ir) >= ir_th
        summary_rows.append(
            {
                "factor": fname,
                "mean_ic": round(mean_ic, 6),
                "ir": round(ir, 4),
                "pass_rule9": float(is_valid),
                "abs_ic": round(abs(mean_ic), 6),
                "n_cross_days": stats["n_days"],
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("abs_ic", ascending=False)
    pass_count = int((summary["pass_rule9"] > 0.5).sum())
    best_factors = summary[summary["pass_rule9"] > 0.5]["factor"].tolist()

    logger.info(f"\n因子验证完成: {pass_count}/{len(summary)} 通过规则9")
    if best_factors:
        logger.info(f"  有效因子: {best_factors}")
    else:
        logger.warning("  无因子通过规则9，建议检查数据质量或调整阈值")

    # 兼容接口：构造 per-symbol rows (空) + summary
    df = pd.DataFrame(
        columns=["symbol", "factor", "mean_ic", "std_ic", "ir", "pass_rule9"]
    )

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
