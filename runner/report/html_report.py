"""
HTML 报告生成模块。

委托 core/report_builder.generate_report() 生成专业 HTML 报告，
不重复实现报告构建逻辑。
"""

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

    if not strategies_data:
        logger.warning("无策略数据，跳过报告生成")
        return None

    if subtitle is None:
        subtitle = f"PyBroker 多策略回测 · {datetime.now().strftime('%Y-%m-%d')}"

    try:
        report_path = build_report(
            output_dir=str(output_dir),
            strategies_data=strategies_data,
            title=title,
            subtitle=subtitle,
            report_name=report_name,
        )
        logger.info(f"报告已保存至 {output_dir / report_name}")
        return report_path
    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        return None


def generate_validation_report(
    output_dir: Path,
    title: str = "量化回测验证分析报告",
    subtitle: Optional[str] = None,
    report_name: str = "validation_report.html",
) -> Optional[str]:
    """
    生成验证分析 HTML 报告。

    委托 core/report_builder.generate_report()。

    Args:
        output_dir: 输出目录
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

    支持 PyBrokerResult 对象和普通字典两种输入格式。

    Args:
        results: 实验结果字典

    Returns:
        {策略名: {metrics: {...}, dates: [...], equity: [...]}} 字典
    """
    strategies_data = {}

    for name, res in results.items():
        if res is None:
            continue

        # PyBrokerResult 对象
        if isinstance(res, PyBrokerResult):
            sd = {
                "metrics": dict(res.metrics) if hasattr(res, "metrics") and res.metrics else {},
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

        # 普通字典格式
        if isinstance(res, dict):
            metrics = res.get("metrics", {})
            if metrics:
                sd = {"metrics": dict(metrics)}
                eq = res.get("equity_curve")
                if eq is not None and hasattr(eq, "empty") and not eq.empty:
                    sd["dates"] = eq["date"].astype(str).tolist()
                    sd["equity"] = eq["equity"].astype(float).tolist()
                strategies_data[name] = sd

    return strategies_data
