"""
参数敏感性分析模块。

委托 core/validation/sensitivity.py 的 SensitivityAnalyzer。
"""

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from loguru import logger

from runner.common.utils import safe_float


def print_sensitivity_summary(
    all_grid_results: Dict[str, pd.DataFrame],
    param_spaces: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    输出参数敏感性摘要。

    按参数值分组统计 Sharpe 的均值、标准差、极值。

    Args:
        all_grid_results: 各策略网格搜索结果
        param_spaces: 各策略参数空间
        output_path: 输出目录
    """
    logger.info("\n  参数敏感性摘要:")
    sensitivity_rows = []

    for sname, grid_df in all_grid_results.items():
        if grid_df.empty:
            continue
        param_keys = list(param_spaces.get(sname, {}).keys())

        for pk in param_keys:
            groups = grid_df.groupby(pk)["sharpe"].agg(["mean", "std", "min", "max"])
            for val, row in groups.iterrows():
                sensitivity_rows.append({
                    "strategy": sname,
                    "param": pk,
                    "value": val,
                    "mean_sharpe": round(row["mean"], 4),
                    "std_sharpe": round(row["std"], 4) if not pd.isna(row["std"]) else 0.0,
                    "min_sharpe": round(row["min"], 4),
                    "max_sharpe": round(row["max"], 4),
                })
                logger.info(f"    {sname}.{pk}={val}: mean={row['mean']:.4f}, std={row.get('std', 0):.4f}")

    if sensitivity_rows:
        sens_df = pd.DataFrame(sensitivity_rows)
        sens_df.to_csv(output_path / "opt_sensitivity.csv", index=False)


def print_optimization_suggestions(
    best_params_all: Dict[str, Dict[str, Any]],
    best_window_config: Dict[str, Dict[str, int]],
    lib,
) -> None:
    """
    输出最优参数建议和窗口配置。

    Args:
        best_params_all: 各策略最优参数
        best_window_config: 各策略最优窗口配置
        lib: StrategyLibrary 实例
    """
    logger.info("\n" + "=" * 60)
    logger.info("最优参数建议")
    logger.info("=" * 60)

    for sname, params in best_params_all.items():
        profile = lib.get_profile(sname)
        if profile is None:
            continue
        old_params = dict(profile.default_params)
        win_cfg = best_window_config.get(sname, {})
        logger.info(f"\n  {sname}:")
        logger.info(f"    当前默认: {old_params}")
        logger.info(f"    建议更新: {params}")
        logger.info(f"    窗口配置: train={win_cfg.get('train_bars', 252)}, test={win_cfg.get('test_bars', 63)}, step={win_cfg.get('step_bars', 21)}")
        changed = {k: (old_params.get(k), v) for k, v in params.items() if old_params.get(k) != v}
        if changed:
            logger.info(f"    变更项: {changed}")

    logger.info("\n" + "-" * 60)
    logger.info("参数应用方式（3种，按推荐程度排序）：")
    logger.info("-" * 60)

    logger.info("\n  方式1（推荐）: 通过 StrategyLibrary.update_default_params() 运行时更新")
    logger.info("    示例代码:")
    logger.info("      from core.strategy_registry import StrategyLibrary")
    logger.info("      lib = StrategyLibrary()")
    for sname, params in best_params_all.items():
        logger.info(f"      lib.update_default_params('{sname}', {params})")

    logger.info("\n  方式2: 更新 config.yaml 策略参数段")
    for sname, params in best_params_all.items():
        logger.info(f"\n    - name: \"{sname}\"")
        logger.info(f"      params:")
        for k, v in params.items():
            if isinstance(v, str):
                logger.info(f"        {k}: \"{v}\"")
            else:
                logger.info(f"        {k}: {v}")

    logger.info("\n  方式3: 通过 custom_params 参数覆盖（单次回测）")
    logger.info("\n" + "=" * 60)
