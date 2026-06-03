"""
HTML 报告生成模块。

委托 core/report_builder.generate_report() 生成专业 HTML 报告，
不重复实现报告构建逻辑。
"""

from dataclasses import is_dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from core.engine.backtest_runner import PyBrokerResult


def generate_html_report(
    config: Dict[str, Any],
    results: Dict[str, Any],
    output_dir: Path,
    optimization_info: Optional[Dict[str, Any]] = None,
    title: str = "量化回测分析报告",
    subtitle: Optional[str] = None,
    report_name: str = "backtest_report_full.html",
) -> Optional[str]:
    """
    生成完整的量化回测分析 HTML 报告。

    委托 core/report_builder.generate_report()，将 PyBrokerResult
    转换为 report_builder 所需格式。

    Args:
        config: 配置字典
        results: 实验结果字典
        output_dir: 输出目录
        optimization_info: 优化信息
        title: 报告标题
        subtitle: 报告副标题
        report_name: 报告文件名

    Returns:
        报告路径，失败返回 None
    """
    from core.report_builder import generate_report as build_report

    logger.info("生成完整 HTML 分析报告")

    # 将 PyBrokerResult 转换为 report_builder 所需格式
    strategies_data = _convert_results(results)

    # 从 validation 结果中提取样本外指标
    out_sample_metrics = None
    if "validation" in results:
        validation = results["validation"]
        if isinstance(validation, dict) and "train_test" in validation:
            out_sample_metrics = validation["train_test"]

    if subtitle is None:
        subtitle = f"PyBroker 多策略回测 · {datetime.now().strftime('%Y-%m-%d')}"

    try:
        report_path = build_report(
            output_dir=str(output_dir),
            strategies_data=strategies_data,
            title=title,
            subtitle=subtitle,
            report_name=report_name,
            config=config,
            out_sample_metrics=out_sample_metrics,
        )
        logger.info(f"报告已保存至 {output_dir / report_name}")
        return report_path
    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        return None


def generate_validation_report(
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    title: str = "量化回测验证分析报告",
    subtitle: Optional[str] = None,
    report_name: str = "validation_report.html",
) -> Optional[str]:
    """
    生成验证分析 HTML 报告。

    委托 core/report_builder.generate_report()。

    Args:
        output_dir: 输出目录
        config: 配置字典（用于动态评价）
        title: 报告标题
        subtitle: 报告副标题
        report_name: 报告文件名

    Returns:
        报告路径，失败返回 None
    """
    from core.report_builder import generate_report as build_report

    logger.info("生成验证分析报告...")

    if subtitle is None:
        subtitle = (
            f"WalkForward + 样本外验证 + 蒙特卡洛 · "
            f"{datetime.now().strftime('%Y-%m-%d')}"
        )

    try:
        report_path = build_report(
            output_dir=str(output_dir),
            title=title,
            subtitle=subtitle,
            report_name=report_name,
            config=config,
        )
        logger.info(f"验证报告已生成: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"验证报告生成失败: {e}")
        return None


def _convert_results(
    results: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    将实验结果转换为 report_builder 所需格式。

    支持 PyBrokerResult、普通字典、DataFrame、dataclass 四种格式。
    特别处理 "all" 键，递归展开其内部内容。

    Args:
        results: 实验结果字典

    Returns:
        {策略名: {metrics: {...}, dates: [...], equity: [...]}} 字典
    """
    import pandas as pd
    import numpy as np

    strategies_data = {}

    for name, res in results.items():
        if res is None:
            continue

        # 处理 "all" 键：递归展开内部内容
        if name == "all" and isinstance(res, dict):
            logger.info("发现 'all' 实验结果，递归展开...")
            sub_data = _convert_results(res)
            strategies_data.update(sub_data)
            continue

        # PyBrokerResult 对象
        if isinstance(res, PyBrokerResult):
            sd = {
                "metrics": dict(res.metrics)
                if hasattr(res, "metrics") and res.metrics
                else {},
            }
            if (
                hasattr(res, "equity_curve")
                and res.equity_curve is not None
                and not res.equity_curve.empty
            ):
                df = res.equity_curve
                sd["dates"] = df["date"].astype(str).tolist()
                sd["equity"] = df["equity"].astype(float).tolist()
            strategies_data[name] = sd
            continue

        # Dataclass（如 BootstrapResult）
        if is_dataclass(res):
            try:
                data_dict = asdict(res)
                # 提取 dataclass 中的数值作为指标
                metrics = {}
                for key, value in data_dict.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        metrics[key] = value
                    elif (
                        isinstance(value, list)
                        and len(value) > 0
                        and isinstance(value[0], (int, float))
                    ):
                        metrics[f"{key}_mean"] = np.mean(value)
                        metrics[f"{key}_std"] = np.std(value)
                        metrics[f"{key}_count"] = len(value)
                if metrics:
                    strategies_data[name] = {"metrics": metrics}
            except Exception as e:
                logger.warning(f"转换 dataclass 失败: {e}")
            continue

        # 普通字典格式（支持 equity 和 equity_curve 两种键名）
        if isinstance(res, dict):
            metrics = res.get("metrics", {})
            if metrics:
                sd = {"metrics": dict(metrics)}
                eq = res.get("equity_curve") or res.get("equity")
                if eq is not None and hasattr(eq, "empty") and not eq.empty:
                    sd["dates"] = eq["date"].astype(str).tolist()
                    sd["equity"] = eq["equity"].astype(float).tolist()
                strategies_data[name] = sd
            continue

        # DataFrame 格式（E1/E2/E3 等实验返回汇总表）
        if isinstance(res, pd.DataFrame) and not res.empty:
            _convert_dataframe_result(name, res, strategies_data)
            continue

    return strategies_data


def _convert_dataframe_result(
    name: str,
    df: "pd.DataFrame",
    strategies_data: Dict[str, Dict[str, Any]],
) -> None:
    """
    将 DataFrame 格式的实验结果展开为报告所需格式。

    按 strategy/experiment 列分组，计算每组的统计指标。

    Args:
        name: 实验名称
        df: 实验结果 DataFrame
        strategies_data: 输出字典（原地修改）
    """
    import numpy as np

    # 筛选数值列
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    group_col = None
    for col in ["strategy", "experiment"]:
        if col in df.columns:
            group_col = col
            break

    if group_col is None:
        # 无分组列，计算整体统计
        stats = {}
        for col in numeric_cols:
            if col in ["date", "time"]:
                continue
            col_data = df[col].dropna()
            if len(col_data) > 0:
                stats[f"{col}_mean"] = col_data.mean()
                stats[f"{col}_std"] = col_data.std()
                stats[f"{col}_min"] = col_data.min()
                stats[f"{col}_max"] = col_data.max()
        # 同时保留第一行数据
        row = df.iloc[0].to_dict()
        clean = {
            k: v
            for k, v in row.items()
            if isinstance(v, (int, float, np.integer, np.floating)) and not np.isnan(v)
        }
        stats.update(clean)
        strategies_data[name] = {"metrics": stats}
        return

    # 有分组列，按组计算统计
    for group_val in df[group_col].dropna().unique():
        if not group_val or (isinstance(group_val, float) and np.isnan(group_val)):
            continue
        subset = df[df[group_col] == group_val]
        if subset.empty:
            continue

        # 计算该组的统计指标
        stats = {}
        for col in numeric_cols:
            if col in ["date", "time"]:
                continue
            col_data = subset[col].dropna()
            if len(col_data) > 0:
                stats[f"{col}_mean"] = col_data.mean()
                stats[f"{col}_std"] = col_data.std()
                stats[f"{col}_min"] = col_data.min()
                stats[f"{col}_max"] = col_data.max()

        # 同时保留第一行数据
        row = subset.iloc[0].to_dict()
        clean = {
            k: v
            for k, v in row.items()
            if isinstance(v, (int, float, np.integer, np.floating)) and not np.isnan(v)
        }
        stats.update(clean)

        entry_name = str(group_val).replace(" ", "_")
        strategies_data[entry_name] = {"metrics": stats}
