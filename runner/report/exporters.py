"""
多格式导出模块。

提供 CSV、PDF 等格式的结果导出功能。
CSV 导出委托 runner/common/utils.save_csv，
PDF 导出使用 matplotlib 的 PdfPages。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from runner.common.utils import save_csv


def export_results_csv(
    results: Dict[str, Any],
    output_dir: Path,
    prefix: str = "",
) -> List[Path]:
    """
    将实验结果导出为 CSV 文件。

    自动识别 DataFrame、Dict、List 等类型，
    分别保存为独立 CSV 文件。

    Args:
        results: 实验结果字典
        output_dir: 输出目录
        prefix: 文件名前缀

    Returns:
        保存的文件路径列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for name, data in results.items():
        if data is None:
            continue

        file_name = f"{prefix}{name}.csv" if prefix else f"{name}.csv"
        path = output_dir / file_name

        if isinstance(data, pd.DataFrame):
            if not data.empty:
                save_csv(data, path)
                saved_paths.append(path)
        elif isinstance(data, dict):
            # 尝试转为 DataFrame
            try:
                df = pd.DataFrame([data] if not _is_nested(data) else _flatten_dict_rows(data))
                if not df.empty:
                    save_csv(df, path)
                    saved_paths.append(path)
            except Exception as e:
                logger.debug(f"跳过字典导出 {name}: {e}")
        elif isinstance(data, list) and len(data) > 0:
            try:
                df = pd.DataFrame(data)
                if not df.empty:
                    save_csv(df, path)
                    saved_paths.append(path)
            except Exception as e:
                logger.debug(f"跳过列表导出 {name}: {e}")

    return saved_paths


def export_metrics_summary(
    metrics_dict: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> Optional[Path]:
    """
    导出策略指标汇总表。

    将多个策略的指标合并为一张表，每行一个策略。

    Args:
        metrics_dict: {策略名: {指标名: 值}} 字典
        output_path: 输出路径

    Returns:
        保存路径，失败返回 None
    """
    rows = []
    for strategy_name, metrics in metrics_dict.items():
        if isinstance(metrics, dict):
            row = {"strategy": strategy_name}
            row.update(metrics)
            rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    save_csv(df, output_path)
    return output_path


def export_validation_summary(
    results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """
    导出验证结果汇总。

    将 task1/task2/task3 的关键指标合并为一张汇总表。

    Args:
        results: 验证结果字典
        output_dir: 输出目录

    Returns:
        汇总文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "validation_summary.txt"

    lines = []
    lines.append(f"验证完成时间: {pd.Timestamp.now()}")
    lines.append("")

    # 任务1 汇总
    if "task1" in results:
        t1 = results["task1"]
        compare = t1.get("compare", pd.DataFrame())
        if not compare.empty:
            lines.append("任务1 - WalkForward 对比:")
            lines.append(compare.to_string(index=False))
            lines.append("")

    # 任务2 汇总
    if "task2" in results:
        t2 = results["task2"]
        yearly = t2.get("yearly", pd.DataFrame())
        if not yearly.empty:
            lines.append("任务2 - 按年验证:")
            for sname in yearly["strategy"].unique():
                sub = yearly[yearly["strategy"] == sname]
                avg_fixed = sub["fixed_sharpe"].mean()
                avg_regime = sub["regime_sharpe"].mean()
                lines.append(f"  {sname}: 固定avg={avg_fixed:.4f}, 环境avg={avg_regime:.4f}")
            lines.append("")

    # 任务3 汇总
    if "task3" in results:
        t3 = results["task3"]
        mc_summary = t3.get("summary", pd.DataFrame())
        if not mc_summary.empty:
            lines.append("任务3 - 蒙特卡洛:")
            lines.append(mc_summary.to_string(index=False))
            lines.append("")

    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"验证汇总已保存: {summary_path}")
    except Exception as e:
        logger.error(f"验证汇总保存失败: {e}")

    return summary_path


def _is_nested(d: dict) -> bool:
    """检查字典是否包含嵌套结构。"""
    for v in d.values():
        if isinstance(v, (dict, list)):
            return True
    return False


def _flatten_dict_rows(d: dict) -> list:
    """将嵌套字典展平为行列表。"""
    rows = []
    for key, value in d.items():
        if isinstance(value, dict):
            row = {"name": key}
            row.update(value)
            rows.append(row)
        else:
            rows.append({"key": key, "value": value})
    return rows
