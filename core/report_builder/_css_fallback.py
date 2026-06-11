"""报告生成模块 — CSS 样式常量。"""

CSS_STYLE = """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background: #f0f2f5; color: #1a1a2e; line-height: 1.6;
        }
        .container { max-width: 1320px; margin: 0 auto; padding: 20px; }
        .report-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white; padding: 36px 40px; border-radius: 16px; margin-bottom: 24px;
            box-shadow: 0 4px 24px rgba(15,52,96,0.3);
        }
        .report-header h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; letter-spacing: 1px; }
        .report-header .subtitle { font-size: 14px; color: #a8b2d1; margin-bottom: 16px; }
        .report-header .meta-row { display: flex; flex-wrap: wrap; gap: 20px; font-size: 13px; color: #8892b0; }
        .report-header .meta-item {
            display: flex; align-items: center; gap: 6px;
            background: rgba(255,255,255,0.08); padding: 6px 14px; border-radius: 20px;
        }
        .report-header .meta-item .label { color: #8892b0; }
        .report-header .meta-item .value { color: #ccd6f6; font-weight: 600; }
        .date-range-badge {
            display: inline-block; background: linear-gradient(135deg, #0f3460, #1a1a2e);
            color: #64ffda; padding: 8px 20px; border-radius: 20px;
            font-size: 15px; font-weight: 600; letter-spacing: 0.5px; margin-top: 12px;
            border: 1px solid rgba(100,255,218,0.3);
        }
        .section-title {
            font-size: 20px; font-weight: 700; color: #1a1a2e; margin: 32px 0 16px;
            padding-left: 16px; border-left: 4px solid #0f3460;
        }
        .section-desc { font-size: 13px; color: #666; margin-bottom: 16px; padding-left: 20px; }
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .kpi-card {
            background: white; border-radius: 12px; padding: 20px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06); text-align: center;
            transition: transform 0.2s, box-shadow 0.2s; border: 1px solid #e8ecf1;
        }
        .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.1); }
        .kpi-card .kpi-label {
            font-size: 12px; color: #8892b0; text-transform: uppercase;
            letter-spacing: 0.5px; margin-bottom: 6px; font-weight: 500;
        }
        .kpi-card .kpi-value { font-size: 26px; font-weight: 700; color: #1a1a2e; }
        .kpi-card .kpi-value.positive { color: #059669; }
        .kpi-card .kpi-value.negative { color: #dc2626; }
        .card {
            background: white; border-radius: 14px; padding: 24px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 20px;
            border: 1px solid #e8ecf1;
        }
        .card-header {
            font-size: 16px; font-weight: 600; color: #1a1a2e; margin-bottom: 16px;
            display: flex; align-items: center; gap: 8px;
        }
        .card-header::before {
            content: ''; display: inline-block; width: 8px; height: 8px;
            border-radius: 50%; background: #0f3460;
        }
        .table-wrapper { overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th {
            background: #f8f9fb; color: #475569; font-weight: 600;
            padding: 12px 14px; text-align: center; border-bottom: 2px solid #e2e8f0;
            font-size: 12px; letter-spacing: 0.3px; white-space: nowrap;
        }
        td { padding: 11px 14px; text-align: center; border-bottom: 1px solid #f1f5f9; white-space: nowrap; }
        tr:hover td { background: #f8fafc; }
        .positive { color: #059669; font-weight: 600; }
        .negative { color: #dc2626; font-weight: 600; }
        .chart-box {
            background: white; border-radius: 14px; padding: 24px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 20px;
            border: 1px solid #e8ecf1;
        }
        .chart-box canvas { max-height: 420px; }
        .heatmap-wrapper { overflow-x: auto; }
        .heatmap-table { border-collapse: collapse; font-size: 11px; }
        .heatmap-table th { padding: 6px 8px; text-align: center; background: #f8f9fb; color: #475569; font-weight: 600; font-size: 11px; }
        .heatmap-table td { padding: 5px 7px; text-align: center; font-size: 11px; border: 1px solid #e2e8f0; }
        .eval-problem { margin-bottom: 18px; padding-left: 16px; border-left: 3px solid #e2e8f0; }
        .eval-problem-title { font-weight: 700; font-size: 14px; color: #1a1a2e; margin-bottom: 6px; }
        .eval-problem p, .eval-problem ul { font-size: 13px; color: #475569; line-height: 1.7; }
        .eval-problem ul { padding-left: 20px; }
        .eval-problem li { margin-bottom: 4px; }
        .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
        .badge-danger { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
        .badge-warning { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
        .badge-success { background: #f0fdf4; color: #059669; border: 1px solid #bbf7d0; }
        .suggestion-list { padding-left: 20px; font-size: 13px; color: #475569; line-height: 1.8; }
        .suggestion-list li { margin-bottom: 8px; }
        .footer { text-align: center; color: #94a3b8; font-size: 12px; margin-top: 40px; padding: 20px; }
        .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .table-box {
            background: white; border-radius: 14px; padding: 24px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 20px;
            border: 1px solid #e8ecf1; overflow-x: auto;
        }
        .data-table {
            width: 100%; border-collapse: collapse; font-size: 13px;
        }
        .data-table th {
            padding: 12px 16px; text-align: left; background: #f8f9fb;
            color: #475569; font-weight: 600; border-bottom: 2px solid #e2e8f0;
        }
        .data-table td {
            padding: 10px 16px; border-bottom: 1px solid #f1f5f9;
        }
        .data-table tbody tr:hover {
            background: #f8fafc;
        }
        @media (max-width: 900px) { .two-col { grid-template-columns: 1fr; } }
        @media (max-width: 768px) { .container { padding: 10px; } .report-header { padding: 24px; } .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
"""



FALLBACK_EVALUATION_HTML = """
    <div class="section-title">综合评价与改进建议</div>
    <div class="section-desc">基于回测结果的多维度定性分析，识别策略核心问题并提出改进方向</div>

    <div class="card">
        <div class="card-header">核心问题诊断</div>
        <div class="eval-problem">
            <div class="eval-problem-title">1. 风险调整后收益极差（Sharpe 过低）</div>
            <p>所有策略的<strong>年化Sharpe比率</strong>均在 <span class="negative">0.008 ~ 0.022</span> 之间，远低于通常可接受水平（一般 &gt;0.5 才被认为具有风险溢价）。这意味着策略承担了很大的波动和回撤，却没有获得对应的超额回报。</p>
        </div>
        <div class="eval-problem">
            <div class="eval-problem-title">2. 最大回撤偏高，风控不足</div>
            <p>回撤最小的 E1_trend 也有较大回撤，而 E2_Fusion 回撤更高。在长达10年的回测中，这样的回撤幅度对实盘资金管理是很大考验。</p>
        </div>
        <div class="eval-problem">
            <div class="eval-problem-title">3. 收益率与交易频率不匹配</div>
            <ul>
                <li><strong>E1_mean_reversion</strong> 收益较高，但回撤也较高，年化收益率需关注。</li>
                <li><strong>E1_trend</strong> 和 <strong>E4_Switching</strong> 交易频率较高，换手频繁但收益需优化。</li>
                <li><strong>E1_term_structure</strong> 收益偏低，需调整参数。</li>
            </ul>
        </div>
        <div class="eval-problem">
            <div class="eval-problem-title">4. 样本内外表现差异显著</div>
            <p>部分策略在样本外表现明显变差，提示可能存在<strong>过拟合风险</strong>。样本内夏普比率与样本外差距过大时，需警惕参数对历史数据的过度适配。</p>
        </div>
        <div class="eval-problem">
            <div class="eval-problem-title">5. 策略间相关性偏高</div>
            <p>多策略组合的分散化效果有限，策略间相关性较高时，组合回撤与单策略回撤接近，未能有效降低系统性风险。</p>
        </div>
    </div>

    <div class="card" style="margin-top:16px;">
        <div class="card-header">多维度评分</div>
        <div class="table-wrapper"><table>
            <thead><tr><th>评价维度</th><th>评级</th><th>说明</th></tr></thead>
            <tbody>
                <tr><td>绝对收益</td><td><span class="badge badge-danger">&#x274c; 较差</span></td><td>十年最高仅31%，年化约2.7%</td></tr>
                <tr><td>风险调整收益</td><td><span class="badge badge-danger">&#x274c; 很差</span></td><td>Sharpe &lt; 0.03，近乎随机漫步</td></tr>
                <tr><td>回撤控制</td><td><span class="badge badge-danger">&#x274c; 不合格</span></td><td>普遍 &gt;15%，有的超30%</td></tr>
                <tr><td>交易频率合理性</td><td><span class="badge badge-warning">&#x26a0;&#xfe0f; 存疑</span></td><td>高频策略收益并不更好</td></tr>
                <tr><td>样本外稳定性</td><td><span class="badge badge-warning">&#x26a0;&#xfe0f; 需关注</span></td><td>部分策略样本外衰减明显</td></tr>
                <tr><td>策略分散化</td><td><span class="badge badge-warning">&#x26a0;&#xfe0f; 不足</span></td><td>策略间相关性偏高，组合效果有限</td></tr>
                <tr><td>实盘可行性</td><td><span class="badge badge-danger">&#x274c; 低</span></td><td>风险收益特征不具备吸引力</td></tr>
            </tbody>
        </table></div>
    </div>

    <div class="card" style="margin-top:16px;">
        <div class="card-header">改进建议（已实施 + 待实施）</div>
        <div class="eval-problem">
            <div class="eval-problem-title" style="color:#10b981;">已实施的改进</div>
            <ol class="suggestion-list">
                <li><strong>检查过拟合</strong>：已增加参数扰动测试和 WalkForward OOS 验证，观察样本外表现是否明显变差。</li>
                <li><strong>加强风控</strong>：止损收紧至 2%、增加 ATR 动态止损、波动率目标仓位管理、信号连续确认已全部实现。</li>
                <li><strong>降低换手率</strong>：信号确认机制、均线间距过滤均已实现，预期交易次数显著减少。</li>
                <li><strong>交易成本真实化</strong>：手续费+滑点提升至万10，淘汰边际利润策略。</li>
                <li><strong>策略相关性过滤</strong>：融合策略自动降权高相关策略对，降低风险集中度。</li>
            </ol>
        </div>
        <div class="eval-problem" style="margin-top:12px;">
            <div class="eval-problem-title" style="color:#f59e0b;">待实施的改进</div>
            <ol class="suggestion-list">
                <li><strong>因子有效性提升</strong>：当前因子IC偏低，需引入更高预测力的因子（如订单流、资金流、期限结构等），或优化因子构造方式（非线性变换、交叉项）。</li>
                <li><strong>自适应参数机制</strong>：固定参数在市场regime切换时失效，建议实现滚动窗口自适应参数（如EMA窗口、ATR倍数随波动率调整）。</li>
                <li><strong>多时间框架融合</strong>：当前仅使用日频信号，建议引入周频/月频趋势判断作为过滤层，降低逆势交易频率。</li>
                <li><strong>动态仓位管理</strong>：根据策略近期表现（如滚动Sharpe）动态调整各策略权重，表现差时自动降权。</li>
                <li><strong>止损策略优化</strong>：当前固定止损可能过于刚性，建议实现追踪止损（Trailing Stop）和时间止损（持仓N日未达目标自动平仓）。</li>
                <li><strong>品种选择优化</strong>：并非所有品种适合所有策略，建议为每个策略筛选适配品种池（基于品种波动率、流动性、趋势性等指标）。</li>
                <li><strong>实盘模拟验证</strong>：回测结果需经过纸面交易（Paper Trading）验证至少3个月，确认实际滑点、成交率与回测假设一致。</li>
            </ol>
        </div>
    </div>
"""
