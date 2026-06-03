"""
报告层：绘图、HTML报告与多格式导出。

委托 core/report_builder 和 utils/plots 的公共接口，
不重复实现报告构建逻辑。
"""

from typing import Any, Dict, Optional

from pathlib import Path

from runner.report.html_report import generate_html_report, generate_validation_report
from runner.report.exporters import (
    export_results_csv,
    export_metrics_summary,
    export_validation_summary,
)


def generate(
    fmt: str,
    results: Dict[str, Any],
    config,
    output_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    统一报告生成入口，供 Pipeline.report() 调用。

    Args:
        fmt: 报告格式（"html", "csv"）
        results: 实验结果字典
        config: 配置对象
        output_dir: 输出目录

    Returns:
        报告路径，失败返回 None
    """
    if output_dir is None:
        output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt_lower = fmt.lower()

    raw_config = config if isinstance(config, dict) else {}

    if fmt_lower == "html":
        return generate_html_report(raw_config, results, output_dir)
    elif fmt_lower == "csv":
        paths = export_results_csv(results, output_dir)
        return str(paths[0]) if paths else None
    elif fmt_lower == "validation":
        return generate_validation_report(output_dir, config=raw_config)
    else:
        from loguru import logger

        logger.warning(f"未知报告格式: {fmt}，可用: html, csv, validation")
        return None


__all__ = [
    "generate",
    "generate_html_report",
    "generate_validation_report",
    "export_results_csv",
    "export_metrics_summary",
    "export_validation_summary",
]
