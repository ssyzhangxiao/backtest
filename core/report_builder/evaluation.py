"""动态评价 HTML 生成模块。

从 generator.py 拆分，负责根据回测结果动态生成综合评价与改进建议。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from core.report_builder.html_template import FALLBACK_EVALUATION_HTML


def _to_float(x: Any, default: float = 0.0) -> Optional[float]:
    """安全转换为 float。"""
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def _collect_risk_items(config: Dict[str, Any],
                        implemented: List[str],
                        pending: List[str]) -> None:
    """从配置中收集风控相关的已实施/待实施项。"""
    risk = config.get("risk", {})

    stop_loss = risk.get("stop_loss")
    if stop_loss and stop_loss.get("enabled"):
        implemented.append(
            f"<strong>止损机制</strong>：已启用，止损幅度 {stop_loss.get('percent', 'N/A')}%"
        )
    else:
        pending.append(
            "<strong>止损机制</strong>：建议启用固定止损或 ATR 动态止损"
        )

    atr_stop = risk.get("atr_stop")
    if atr_stop and atr_stop.get("enabled"):
        implemented.append(
            f"<strong>ATR 动态止损</strong>：已启用，倍数 {atr_stop.get('multiplier', 'N/A')}x"
        )
    else:
        pending.append(
            "<strong>ATR 动态止损</strong>：建议实现基于波动率的自适应止损"
        )

    costs = config.get("costs", {})
    if costs.get("commission") or costs.get("slippage"):
        commission = costs.get("commission", 0)
        slippage = costs.get("slippage", 0)
        implemented.append(
            f"<strong>交易成本</strong>：已设置，手续费 {commission * 10000:.0f}bps + 滑点 {slippage * 10000:.0f}bps"
        )
    else:
        pending.append(
            "<strong>交易成本</strong>：建议真实化交易成本（手续费+滑点）"
        )

    signal = config.get("signal", {})
    if signal.get("confirm"):
        implemented.append("<strong>信号确认</strong>：已启用连续确认机制")
    else:
        pending.append("<strong>信号确认</strong>：建议增加信号确认以降低假阳性")


def _compute_avg_metrics(strategies_data: Dict[str, Any]) -> Tuple[float, float]:
    """计算平均 Sharpe 和平均最大回撤。"""
    sharpes: List[float] = []
    max_dds: List[float] = []

    for data in (strategies_data or {}).values():
        metrics = data.get("metrics", {})
        sharpe = _to_float(metrics.get("sharpe"))
        max_dd = _to_float(metrics.get("max_drawdown_pct"))
        if isinstance(sharpe, (int, float)):
            sharpes.append(sharpe)
        if isinstance(max_dd, (int, float)):
            max_dds.append(abs(max_dd))

    avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
    avg_max_dd = sum(max_dds) / len(max_dds) if max_dds else 0.0
    return avg_sharpe, avg_max_dd


def _diagnose_sharpe(avg_sharpe: float) -> Dict[str, str]:
    """诊断 Sharpe 比率水平。"""
    if avg_sharpe < 0.3:
        return {
            "title": "1. 风险调整后收益偏低",
            "desc": f'平均 Sharpe 比率仅为 <span class="negative">{avg_sharpe:.3f}</span>，低于一般可接受水平（建议 &gt;0.5）。',
        }
    if avg_sharpe < 0.5:
        return {
            "title": "1. 风险调整后收益有待提高",
            "desc": f"平均 Sharpe 比率为 {avg_sharpe:.3f}，仍有优化空间。",
        }
    return {
        "title": "1. 风险调整后收益良好",
        "desc": f'平均 Sharpe 比率为 <span class="positive">{avg_sharpe:.3f}</span>，表现不错。',
    }


def _diagnose_drawdown(avg_max_dd: float) -> Dict[str, str]:
    """诊断最大回撤水平。"""
    if avg_max_dd > 20:
        return {
            "title": "2. 最大回撤偏高",
            "desc": f'平均最大回撤 <span class="negative">{avg_max_dd:.1f}%</span>，风控需要加强。',
        }
    if avg_max_dd > 10:
        return {
            "title": "2. 回撤控制尚可",
            "desc": f"平均最大回撤 {avg_max_dd:.1f}%，仍有优化空间。",
        }
    return {
        "title": "2. 回撤控制良好",
        "desc": f'平均最大回撤 <span class="positive">{avg_max_dd:.1f}%</span>，表现优秀。',
    }


def _diagnose_oos_decay(out_sample_metrics: Optional[Dict[str, Any]],
                        pending: List[str]) -> Tuple[Dict[str, str], float]:
    """诊断样本外衰减，返回 (问题字典, 衰减率)。"""
    if not out_sample_metrics:
        return {
            "title": "3. 样本外表现稳定",
            "desc": "样本外 Sharpe 衰减很小，策略鲁棒性较好。",
        }, 0.0

    in_sample = out_sample_metrics.get("in_sample", {})
    out_sample = out_sample_metrics.get("out_sample", {})

    is_sharpe = _to_float(in_sample.get("sharpe"))
    oos_sharpe = _to_float(out_sample.get("sharpe"))

    oos_decay = 0.0
    if is_sharpe and is_sharpe != 0:
        oos_decay = (is_sharpe - oos_sharpe) / abs(is_sharpe) if is_sharpe > 0 else 0

    if oos_decay > 0.3:
        pending.append(
            "<strong>过拟合检查</strong>：建议增加参数扰动测试和 WalkForward 验证"
        )
        return {
            "title": "3. 样本外衰减显著",
            "desc": f'样本外 Sharpe 衰减约 <span class="negative">{oos_decay * 100:.0f}%</span>，存在过拟合风险。',
        }, oos_decay
    if oos_decay > 0:
        return {
            "title": "3. 样本外有一定衰减",
            "desc": f"样本外 Sharpe 衰减约 {oos_decay * 100:.0f}%，需要关注。",
        }, oos_decay
    return {
        "title": "3. 样本外表现稳定",
        "desc": "样本外 Sharpe 衰减很小，策略鲁棒性较好。",
    }, oos_decay


def _check_validation_files(output_dir: Optional[Path],
                            implemented: List[str],
                            pending: List[str]) -> None:
    """检查验证分析文件是否存在。"""
    if output_dir and output_dir.exists():
        has_wf = (output_dir / "task1_wf_compare.csv").exists() or (
            output_dir / "e7_equity_out_sample.csv"
        ).exists()
        has_mc = (output_dir / "task3_monte_carlo_summary.csv").exists() or (
            output_dir / "e9_monte_carlo_results.csv"
        ).exists()
        has_corr = (output_dir / "e5_correlation_matrix.csv").exists()
    else:
        has_wf = has_mc = has_corr = False

    if has_wf:
        implemented.append("<strong>WalkForward 验证</strong>：已执行滚动窗口验证")
    else:
        pending.append("<strong>WalkForward 验证</strong>：建议执行滚动窗口验证以评估参数稳定性")

    if has_mc:
        implemented.append("<strong>蒙特卡洛模拟</strong>：已执行蒙特卡洛分析")
    else:
        pending.append("<strong>蒙特卡洛模拟</strong>：建议执行蒙特卡洛模拟以评估策略鲁棒性")

    if has_corr:
        implemented.append("<strong>相关性分析</strong>：已计算策略相关性矩阵")
    else:
        pending.append("<strong>相关性分析</strong>：建议分析策略间相关性以优化组合")


def _append_generic_suggestions(pending: List[str]) -> None:
    """追加通用改进建议。"""
    pending.extend([
        "<strong>因子有效性提升</strong>：引入更高预测力的因子或优化因子构造方式",
        "<strong>自适应参数机制</strong>：实现滚动窗口自适应参数（EMA窗口、ATR倍数等）",
        "<strong>多时间框架融合</strong>：引入周频/月频趋势判断作为过滤层",
        "<strong>动态仓位管理</strong>：根据策略近期表现动态调整权重",
        "<strong>追踪止损优化</strong>：实现 Trailing Stop 和时间止损",
        "<strong>品种选择优化</strong>：为每个策略筛选适配品种池",
        "<strong>实盘模拟验证</strong>：通过 Paper Trading 验证至少3个月",
    ])


def _compute_scores(avg_sharpe: float, avg_max_dd: float,
                    oos_decay: float, output_dir: Optional[Path]) -> List[Tuple[str, str, str]]:
    """计算多维度评分。"""
    scores: List[Tuple[str, str, str]] = []

    # 绝对收益
    if avg_sharpe > 1.0:
        scores.append(("绝对收益", "优秀", "Sharpe > 1.0"))
    elif avg_sharpe > 0.5:
        scores.append(("绝对收益", "良好", f"Sharpe {avg_sharpe:.2f}"))
    else:
        scores.append(("绝对收益", "较差", f"Sharpe {avg_sharpe:.2f}"))

    # 风险调整收益
    if avg_sharpe > 1.0:
        scores.append(("风险调整收益", "优秀", f"Sharpe {avg_sharpe:.2f}"))
    elif avg_sharpe > 0.5:
        scores.append(("风险调整收益", "良好", f"Sharpe {avg_sharpe:.2f}"))
    else:
        scores.append(("风险调整收益", "很差", f"Sharpe {avg_sharpe:.2f}"))

    # 回撤控制
    if avg_max_dd < 10:
        scores.append(("回撤控制", "优秀", f"平均回撤 {avg_max_dd:.1f}%"))
    elif avg_max_dd < 20:
        scores.append(("回撤控制", "合格", f"平均回撤 {avg_max_dd:.1f}%"))
    else:
        scores.append(("回撤控制", "不合格", f"平均回撤 {avg_max_dd:.1f}%"))

    # 样本外稳定性
    if oos_decay < 0.1:
        scores.append(("样本外稳定性", "很好", "衰减 < 10%"))
    elif oos_decay < 0.3:
        scores.append(("样本外稳定性", "需关注", f"衰减 {oos_decay * 100:.0f}%"))
    else:
        scores.append(("样本外稳定性", "风险大", f"衰减 {oos_decay * 100:.0f}%"))

    # 分析完整性
    analysis_count = 0
    if output_dir and output_dir.exists():
        if (output_dir / "task1_wf_compare.csv").exists() or (output_dir / "e7_equity_out_sample.csv").exists():
            analysis_count += 1
        if (output_dir / "task3_monte_carlo_summary.csv").exists() or (output_dir / "e9_monte_carlo_results.csv").exists():
            analysis_count += 1
        if (output_dir / "e5_correlation_matrix.csv").exists():
            analysis_count += 1

    if analysis_count >= 3:
        scores.append(("分析完整性", "完整", "所有分析均执行"))
    elif analysis_count >= 1:
        scores.append(("分析完整性", "部分", f"已执行 {analysis_count}/3 项分析"))
    else:
        scores.append(("分析完整性", "不足", "建议补充验证分析"))

    return scores


def _render_evaluation_html(problems: List[Dict[str, str]],
                            scores: List[Tuple[str, str, str]],
                            implemented: List[str],
                            pending: List[str]) -> str:
    """渲染评价 HTML 字符串。"""
    html_parts: List[str] = []

    html_parts.append("""
    <div class="section-title">综合评价与改进建议</div>
    <div class="section-desc">基于回测结果的多维度定性分析，识别策略核心问题并提出改进方向</div>

    <div class="card">
        <div class="card-header">核心问题诊断</div>
""")

    for problem in problems:
        html_parts.append(f"""
        <div class="eval-problem">
            <div class="eval-problem-title">{problem["title"]}</div>
            <p>{problem["desc"]}</p>
        </div>
""")

    html_parts.append("""
    </div>

    <div class="card" style="margin-top:16px;">
        <div class="card-header">多维度评分</div>
        <div class="table-wrapper"><table>
            <thead><tr><th>评价维度</th><th>评级</th><th>说明</th></tr></thead>
            <tbody>
""")

    for dim, rating, desc in scores:
        badge_class = (
            "badge-success"
            if "优秀" in rating or "很好" in rating or "完整" in rating
            else "badge-warning"
            if "良好" in rating or "合格" in rating or "需关注" in rating or "部分" in rating
            else "badge-danger"
        )
        html_parts.append(f"""
                <tr><td>{dim}</td><td><span class="badge {badge_class}">{rating}</span></td><td>{desc}</td></tr>
""")

    html_parts.append("""
            </tbody>
        </table></div>
    </div>

    <div class="card" style="margin-top:16px;">
        <div class="card-header">改进建议（已实施 + 待实施）</div>
        <div class="eval-problem">
            <div class="eval-problem-title" style="color:#10b981;">已实施的改进</div>
            <ol class="suggestion-list">
""")

    if implemented:
        for item in implemented[:8]:
            html_parts.append(f"""
                <li>{item}</li>
""")
    else:
        html_parts.append("""
                <li>暂无记录，请检查配置或运行完整分析流程。</li>
""")

    html_parts.append("""
            </ol>
        </div>
        <div class="eval-problem" style="margin-top:12px;">
            <div class="eval-problem-title" style="color:#f59e0b;">待实施的改进</div>
            <ol class="suggestion-list">
""")

    for item in pending[:10]:
        html_parts.append(f"""
                <li>{item}</li>
""")

    html_parts.append("""
            </ol>
        </div>
    </div>
""")

    return "".join(html_parts)


def build_dynamic_evaluation_html(
    config: Optional[Dict[str, Any]] = None,
    strategies_data: Optional[Dict[str, Any]] = None,
    out_sample_metrics: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> str:
    """根据输入数据动态生成综合评价与改进建议 HTML。"""
    try:
        implemented_items: List[str] = []
        pending_items: List[str] = []
        problems: List[Dict[str, str]] = []

        # 1. 配置项检查
        if config:
            _collect_risk_items(config, implemented_items, pending_items)

        # 2. 策略指标
        avg_sharpe, avg_max_dd = _compute_avg_metrics(strategies_data)

        # 3. 问题诊断
        problems.append(_diagnose_sharpe(avg_sharpe))
        problems.append(_diagnose_drawdown(avg_max_dd))
        oos_problem, oos_decay = _diagnose_oos_decay(out_sample_metrics, pending_items)
        problems.append(oos_problem)

        # 4. 验证文件检查
        _check_validation_files(output_dir, implemented_items, pending_items)

        # 5. 通用建议
        _append_generic_suggestions(pending_items)

        # 6. 评分
        scores = _compute_scores(avg_sharpe, avg_max_dd, oos_decay, output_dir)

        # 7. 渲染
        return _render_evaluation_html(problems, scores, implemented_items, pending_items)

    except Exception as e:
        logger.warning(f"动态评价生成失败，使用默认模板: {e}")
        return FALLBACK_EVALUATION_HTML
