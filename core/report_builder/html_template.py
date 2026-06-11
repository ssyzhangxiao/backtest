"""报告生成模块 — HTML 模板组装。

从子模块导入 CSS/JS/FALLBACK 常量，组装为最终 HTML_TEMPLATE。
子模块：
- _css_fallback.py  CSS 样式 + 后备评价 HTML
- _chart_js.py      Chart.js 脚本
"""

from core.report_builder._css_fallback import CSS_STYLE, FALLBACK_EVALUATION_HTML
from core.report_builder._chart_js import CHART_JS_SCRIPT

# ══════════════════════════════════════════════════════════════════════════════
# HTML 模板
# ══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = (
    """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>$title</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
    <style>
"""
    + CSS_STYLE
    + """
    </style>
</head>
<body>
<div class="container">

    <div class="report-header">
        <h1>&#x1f4ca; $title</h1>
        <div class="subtitle">$subtitle</div>
        <div class="date-range-badge">回测区间: $backtest_start ~ $backtest_end</div>
        <div class="meta-row" style="margin-top: 16px;">
            <div class="meta-item"><span class="label">生成时间</span><span class="value">$now_str</span></div>
            <div class="meta-item"><span class="label">回测引擎</span><span class="value">PyBroker</span></div>
            <div class="meta-item"><span class="label">初始资金</span><span class="value">&#165;$init_capital</span></div>
            <div class="meta-item"><span class="label">交易品种</span><span class="value">$symbols_str</span></div>
            <div class="meta-item"><span class="label">数据周期</span><span class="value">日线</span></div>
            <div class="meta-item"><span class="label">总交易日</span><span class="value">$total_days 天 ($total_years 年)</span></div>
        </div>
        $data_source_warning_html
    </div>

    <div class="section-title">全局绩效指标</div>
    <div class="section-desc">以下为最佳策略组合（$best_strategy_label）的核心绩效指标概览</div>
    <div class="kpi-grid">
        $kpi_cards_html
    </div>

    <div class="card">
        <div class="card-header">策略绩效对比</div>
        <div class="section-desc" style="margin-top:-8px;">对比各策略在全回测区间的关键绩效指标（正收益绿、负收益红）</div>
        <div class="table-wrapper">
        <table>
            <thead>
            <tr>
                <th>策略名称</th><th>总收益率</th><th>年化收益率</th><th>夏普比率</th>
                <th>最大回撤</th><th>卡玛比率</th><th>胜率</th><th>盈亏比</th><th>交易次数</th>
            </tr>
            </thead>
            <tbody>$strategy_table_html</tbody>
        </table>
        </div>
    </div>

    <div class="card">
        <div class="card-header">所有策略样本内/外绩效对比</div>
        <div class="section-desc" style="margin-top:-8px;">各策略在样本内与样本外的独立表现对比，评估各策略泛化能力</div>
        <div class="table-wrapper">
        <table>
            <thead>
            <tr>
                <th>策略</th><th>数据集</th><th>总收益率</th><th>年化收益率</th><th>夏普比率</th>
                <th>最大回撤</th><th>卡玛比率</th><th>胜率</th><th>交易次数</th>
            </tr>
            </thead>
            <tbody>$strategy_oos_html</tbody>
        </table>
        </div>
    </div>

    <div class="section-title">可视化图表</div>
    <div class="section-desc">基于 Chart.js 的交互式图表，支持缩放、悬停提示</div>

    <div class="chart-box"><div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f4c8; 净值曲线对比（归一化，从1开始）</div><canvas id="chartEquity"></canvas></div>
    <div class="chart-box"><div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f4ca; 所有策略回撤对比</div><canvas id="chartAllDrawdowns"></canvas></div>

    <div class="two-col">
        <div class="chart-box"><div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f539; 风险收益散点图</div><canvas id="chartScatter"></canvas></div>
        <div class="chart-box"><div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f321;&#xfe0f; 月度收益率热力图（$best_strategy_label）</div><canvas id="chartHeatmap"></canvas></div>
    </div>
    <div id="allHeatmapsContainer"></div>

    <div class="section-title">风险分析</div>
    <div class="two-col">
        <div class="chart-box"><div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f4ca; 所有策略滚动夏普对比</div><canvas id="chartAllRollingSharpe"></canvas></div>
        <div class="chart-box"><div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f4c9; 所有策略滚动回撤对比</div><canvas id="chartAllRollingDD"></canvas></div>
    </div>

    <div class="chart-box">
        <div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f50d; 策略相关性热力图</div>
        <canvas id="chartCorr" style="max-height:350px;"></canvas>
    </div>

    <div class="two-col">
        <div class="chart-box"><div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f4c8; 所有策略样本内净值对比</div><canvas id="chartAllIS"></canvas></div>
        <div class="chart-box"><div style="font-size:14px;font-weight:600;margin-bottom:12px;">&#x1f4c8; 所有策略样本外净值对比</div><canvas id="chartAllOS"></canvas></div>
    </div>

    $diversification_html

    $rebalance_html

    $evaluation_html

    <div class="footer">
        &#xa9; $now_year 量化回测系统 | 由 PyBroker + Chart.js 生成 | 仅供研究参考，不构成投资建议
    </div>
</div>

<script>
"""
    + CHART_JS_SCRIPT
    + """
</script>
</body>
</html>"""
)
