"""
HTML 报告生成模块。

委托 core/report_builder.generate_report() 生成专业 HTML 报告，
不重复实现报告构建逻辑。
"""

from dataclasses import is_dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.execution._result_types import PyBrokerResult
from utils.metrics import MetricsCalculator


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

    新增（2026-06-12）：若 results["validation"]["standard_report"] 已跑过，
    自动生成 5 段式因子验证 HTML 片段（factor_5_section_report.html），
    并在主报告 evaluation 段追加链接。委托 `build_factor_5_section_html()`。

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

    # ── 5 段式因子验证片段（2026-06-12 集成） ──
    factor_5_section_link: Optional[str] = None
    factor_5_section_path: Optional[Path] = None
    try:
        factor_5_section_path = _maybe_build_factor_5_section(
            results=results, output_dir=output_dir
        )
        if factor_5_section_path is not None:
            factor_5_section_link = factor_5_section_path.name
            logger.info(f"5 段式因子验证片段已生成: {factor_5_section_link}")
    except Exception as e:
        logger.warning(f"5 段式因子验证片段生成失败（非致命）: {e}")

    # 将 PyBrokerResult 转换为 report_builder 所需格式
    strategies_data = _convert_results(results)

    # 从 validation 结果中提取样本外指标
    out_sample_metrics = None
    if "validation" in results:
        validation = results["validation"]
        if isinstance(validation, dict) and "train_test" in validation:
            out_sample_metrics = validation["train_test"]

    # 把 5 段式片段链接注入到主报告评价段（独立 HTML 片段，不嵌入主模板）
    if factor_5_section_link:
        # 通过 evaluation_html 注入：主报告评价模块下方追加一行链接
        link_html = (
            f'<div class="factor5-section-link" style="margin:18px 0 0 0;'
            f"padding:14px 18px;background:#eef5fb;border-left:4px solid #2b6cb0;"
            f'border-radius:4px;">'
            f"<strong>5 段式因子验证报告（ADF + IC + PRF + 事件研究 + 冗余）已生成：</strong>"
            f'<a href="{factor_5_section_link}" target="_blank" '
            f'style="color:#2b6cb0;text-decoration:underline;">'
            f"打开因子验证明细 → {factor_5_section_link}</a>"
            f'<div style="margin-top:6px;font-size:13px;color:#555;">'
            f"每品种 × 每因子的 ADF/IC/PRF/EventStudy/Redundancy 5 段明细 + 通过状态，"
            f"详见独立 HTML 报告。"
            f"</div></div>"
        )
        # 若 evaluation_html 未指定，则生成空串再注入
        # 这里不能直接给 build_report 传 evaluation_html（破坏默认值 build_dynamic_evaluation_html），
        # 改为在主报告生成后追加锚点 div。
        _pending_factor5_link = link_html
    else:
        _pending_factor5_link = None

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

        # 5 段式链接追加：若主报告是 HTML，注入锚点 div 到 </body> 前
        if _pending_factor5_link and report_path:
            _inject_html_anchor(report_path, _pending_factor5_link)

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
        # 无分组列，计算整体统计（委托 MetricsCalculator.aggregate_stats，规则17）
        stats = {}
        for col in numeric_cols:
            if col in ["date", "time"]:
                continue
            col_stats = MetricsCalculator.aggregate_stats(df[col])
            for k, v in col_stats.items():
                stats[f"{col}_{k}"] = v
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


# ══════════════════════════════════════════════════════════════════════════════
# 5 段式因子验证 HTML 片段（2026-06-12 集成）
# ══════════════════════════════════════════════════════════════════════════════

FACTOR_5_SECTION_FILENAME = "factor_5_section_report.html"


def _maybe_build_factor_5_section(
    results: Dict[str, Any],
    output_dir: Path,
) -> Optional[Path]:
    """
    若 results["validation"]["standard_report"] 存在，生成 5 段式 HTML 片段。

    输入契约（2026-06-12 整改）：pipe.validate("standard_report") 完成后会写入
    results["validation"]["standard_report"]，结构为：
      {
        "output_path": Path | None,    # factor_standard_report.csv
        "summary_path": Path | None,   # factor_standard_summary.csv
        "n_factors": int,
        "n_fully_validated": int,
        "fully_validated_rate": float,
      }

    或者保留旧格式（直接 dict 包含 results key）也兼容。

    Args:
        results: 实验结果字典
        output_dir: HTML 报告输出目录（与主报告并列存放）

    Returns:
        生成的 5 段式 HTML 片段路径，未跑过 standard_report 返回 None
    """
    validation = results.get("validation")
    if not isinstance(validation, dict):
        return None
    standard = validation.get("standard_report")
    if not isinstance(standard, dict):
        return None

    # 提取 standard_report 实际产物路径
    full_path = standard.get("output_path")
    summary_path = standard.get("summary_path")

    # 兼容旧格式：standard_report 直接是 {results: {symbol: DataFrame}}
    if full_path is None and "results" in standard:
        # 旧格式：从 results 字典临时拼一个 DataFrame
        frames = []
        for sym, val in standard["results"].items():
            if isinstance(val, pd.DataFrame):
                tmp = val.copy()
                tmp["symbol"] = sym
                frames.append(tmp)
        if not frames:
            return None
        full_df = pd.concat(frames, ignore_index=True)
        full_path = output_dir / "factor_standard_report.csv"
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(full_path, index=False, encoding="utf-8-sig")
        summary_path = None  # 旧格式无独立 summary

    if full_path is None or not Path(full_path).exists():
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    full_df = pd.read_csv(full_path)
    summary_df: Optional[pd.DataFrame] = None
    if summary_path and Path(summary_path).exists():
        try:
            summary_df = pd.read_csv(summary_path)
        except Exception:
            summary_df = None

    html_path = output_dir / FACTOR_5_SECTION_FILENAME
    html_content = build_factor_5_section_html(
        full_df=full_df,
        summary_df=summary_df,
        meta={
            "n_factors": standard.get("n_factors"),
            "n_fully_validated": standard.get("n_fully_validated"),
            "fully_validated_rate": standard.get("fully_validated_rate"),
        },
    )
    html_path.write_text(html_content, encoding="utf-8")
    return html_path


def build_factor_5_section_html(
    full_df: pd.DataFrame,
    summary_df: Optional[pd.DataFrame] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    构建 5 段式因子验证 HTML 片段（独立文件，可被主报告链接）。

    内容结构：
      1) 顶部摘要卡片（n_factors / n_fully_validated / fully_validated_rate）
      2) 按因子汇总表（summary_df 存在时）
      3) 每品种 × 每因子 5 段明细表（full_df）
      4) 通过条件说明（5 段 + 阈值）

    Args:
        full_df: factor_standard_report.csv 内容
        summary_df: factor_standard_summary.csv 内容（可选）
        meta: 元数据字典（可选）

    Returns:
        HTML 字符串
    """
    meta = meta or {}
    n_factors = meta.get("n_factors")
    n_fully = meta.get("n_fully_validated")
    fvr = meta.get("fully_validated_rate")

    # 摘要卡片
    cards_html = (
        '<div class="factor5-cards" style="display:flex;gap:14px;margin:18px 0;flex-wrap:wrap;">'
        f'<div style="flex:1;min-width:200px;padding:14px 18px;background:#eef5fb;'
        f'border-left:4px solid #2b6cb0;border-radius:4px;">'
        f'<div style="font-size:13px;color:#555;">已验证因子数</div>'
        f'<div style="font-size:24px;font-weight:600;color:#2b6cb0;">{n_factors if n_factors is not None else "-"}</div></div>'
        f'<div style="flex:1;min-width:200px;padding:14px 18px;background:#eaf6ee;'
        f'border-left:4px solid #2f855a;border-radius:4px;">'
        f'<div style="font-size:13px;color:#555;">完整通过 (5/5 段)</div>'
        f'<div style="font-size:24px;font-weight:600;color:#2f855a;">{n_fully if n_fully is not None else "-"}</div></div>'
        f'<div style="flex:1;min-width:200px;padding:14px 18px;background:#fef5e7;'
        f'border-left:4px solid #c05621;border-radius:4px;">'
        f'<div style="font-size:13px;color:#555;">通过率</div>'
        f'<div style="font-size:24px;font-weight:600;color:#c05621;">'
        f"{f'{fvr:.1%}' if isinstance(fvr, (int, float)) else '-'}</div></div>"
        "</div>"
    )

    # 因子汇总表
    summary_html = ""
    if summary_df is not None and not summary_df.empty:
        summary_html = (
            '<h3 style="margin-top:24px;color:#2c5282;">按因子汇总</h3>'
            + _df_to_html_table(summary_df, table_id="factor5-summary")
        )

    # 明细表（限制行数避免过大）
    MAX_ROWS = 200
    full_table_html = ""
    if full_df is not None and not full_df.empty:
        # 关键列优先
        preferred = [
            "symbol",
            "factor",
            "n_pass_sections",
            "n_present_sections",
            "fully_validated",
            "is_complete",
            "pass_adf",
            "value_adf",
            "pass_ic",
            "value_ic",
            "pass_prf",
            "value_prf",
            "pass_event_study",
            "value_event_study",
            "pass_redundancy",
            "value_redundancy",
        ]
        cols = [c for c in preferred if c in full_df.columns]
        if not cols:
            cols = list(full_df.columns)
        sub_df = full_df[cols].head(MAX_ROWS)
        truncated_note = (
            (
                f'<p style="color:#888;font-size:13px;">（仅显示前 {MAX_ROWS} 行，'
                f"完整 {len(full_df)} 行请查看 factor_standard_report.csv）</p>"
            )
            if len(full_df) > MAX_ROWS
            else ""
        )

        full_table_html = (
            '<h3 style="margin-top:24px;color:#2c5282;">每品种 × 每因子 5 段明细</h3>'
            + truncated_note
            + _df_to_html_table(sub_df, table_id="factor5-detail")
        )

    # 通过条件说明
    thresholds_html = (
        '<h3 style="margin-top:24px;color:#2c5282;">通过条件（5 段式）</h3>'
        '<ul style="line-height:1.8;color:#444;">'
        "<li><strong>ADF 平稳性</strong>：p_value &lt; 0.05（拒绝单位根）</li>"
        "<li><strong>IC 绝对值</strong>：abs(IC) &gt; 0.03</li>"
        "<li><strong>PRF 离散信号</strong>：Precision &gt; 0.55 且 Lift &gt; 0</li>"
        "<li><strong>事件研究</strong>：T+5 ~ T+10 累计收益 p_value &lt; 0.01</li>"
        "<li><strong>Spearman 冗余</strong>：与高 IC 因子 Spearman ρ &lt; 0.7</li>"
        "<li><strong>完整验证</strong>：5 段全部存在 且 ≥ 4/5 段通过</li>"
        "</ul>"
    )

    return (
        '<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n'
        '<meta charset="utf-8">\n'
        "<title>5 段式因子验证报告</title>\n"
        '<style>body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,'
        '"Helvetica Neue",Arial,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",'
        "sans-serif;max-width:1200px;margin:30px auto;padding:0 24px;color:#1a202c;"
        "background:#fafbfc;line-height:1.6;}"
        "h1{color:#1a365d;border-bottom:3px solid #2b6cb0;padding-bottom:12px;}"
        "table{border-collapse:collapse;width:100%;margin:14px 0;"
        "background:#fff;box-shadow:0 1px 3px rgba(0,0,0,0.06);font-size:13px;}"
        "th,td{padding:8px 12px;border:1px solid #e2e8f0;text-align:left;}"
        "th{background:#edf2f7;font-weight:600;color:#2d3748;}"
        "tr:nth-child(even) td{background:#f7fafc;}"
        "</style></head>\n<body>\n"
        "<h1>5 段式因子验证报告</h1>\n"
        f'<p style="color:#555;">ADF 平稳性 + IC 连续检验 + PRF 离散检验 + 事件研究 + Spearman 冗余 '
        f"—— 5 维度互补视角，禁止单 IC 阈值淘汰因子（规则 28 阶段 A）。</p>\n"
        f"{cards_html}\n"
        f"{summary_html}\n"
        f"{full_table_html}\n"
        f"{thresholds_html}\n"
        "</body></html>\n"
    )


def _df_to_html_table(df: pd.DataFrame, table_id: str = "") -> str:
    """
    DataFrame → 简单 HTML 表格（无外部依赖，仅依赖 pandas to_html）。

    处理：
      - 布尔值 → ✅ / ❌ 图标
      - NaN → "—"
      - 长表加滚动条
    """
    if df is None or df.empty:
        return "<p style='color:#888;'>（空）</p>"

    df = df.copy()

    # 布尔列可视化
    bool_cols = [c for c in df.columns if df[c].dtype == bool]
    for c in bool_cols:
        df[c] = df[c].map(lambda v: "✅" if v else "❌" if pd.notna(v) else "—")

    # NaN → "—"
    df = df.fillna("—")

    return df.to_html(
        index=False,
        escape=False,
        border=0,
        table_id=table_id or None,
        classes="factor5-table",
        na_rep="—",
    )


def _inject_html_anchor(report_path: Path, anchor_html: str) -> bool:
    """
    在 HTML 报告 </body> 前注入锚点 div（轻量改写，不重写整个文档）。

    Args:
        report_path: 主报告路径
        anchor_html: 要注入的 HTML 片段

    Returns:
        是否注入成功
    """
    if not Path(report_path).exists():
        return False
    try:
        text = Path(report_path).read_text(encoding="utf-8")
    except Exception:
        return False

    if "</body>" in text.lower():
        # 不区分大小写替换
        lower = text.lower()
        idx = lower.rfind("</body>")
        new_text = text[:idx] + anchor_html + "\n" + text[idx:]
    else:
        # 没有 </body> 标记：直接追加
        new_text = text + "\n" + anchor_html

    try:
        Path(report_path).write_text(new_text, encoding="utf-8")
        return True
    except Exception:
        return False
