"""报告生成模块 — Chart.js 脚本常量。"""

CHART_JS_SCRIPT = """
var CHART_FONT = "-apple-system, BlinkMacSystemFont, 'PingFang SC', 'Microsoft YaHei', sans-serif";
var COLORS = {
    trend: '#3b82f6', term_structure: '#f59e0b', mean_reversion: '#8b5cf6',
    vol_breakout: '#06b6d4',
    fusion: '#10b981', switching: '#ef4444'
};
var reportData = $chart_data_json;

(function() {
    if (!reportData || !reportData.equity_curves) return;
    var ec = reportData.equity_curves;
    var names = Object.keys(ec);
    var datasets = names.map(function(name) {
        return {
            label: ec[name].label || name,
            data: ec[name].equity,
            borderColor: COLORS[name] || '#666',
            backgroundColor: 'transparent',
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.1,
        };
    });
    new Chart(document.getElementById('chartEquity'), {
        type: 'line',
        data: { labels: ec[names[0]].dates, datasets: datasets },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                tooltip: { callbacks: { label: function(ctx) { return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(4); } } },
                legend: { position: 'top', labels: { usePointStyle: true, padding: 16 } },
            },
            scales: {
                x: { type: 'category', ticks: { maxTicksLimit: 12, autoSkip: true } },
                y: { title: { display: true, text: '净值 (归一化)' } },
            },
        },
    });
})();

(function() {
    var ad = reportData.all_drawdowns;
    if (!ad) return;
    var keys = Object.keys(ad);
    if (!keys.length) return;
    var DD_COLORS = {
        trend: '#3b82f6', term_structure: '#f59e0b', mean_reversion: '#8b5cf6',
        vol_breakout: '#06b6d4', cross_sectional: '#10b981', fusion: '#10b981'
    };
    var DD_LABELS = {
        trend: '趋势策略', term_structure: '期限结构', mean_reversion: '均值回归',
        vol_breakout: '波动率突破', cross_sectional: '横截面打分', fusion: '融合策略'
    };
    var DD_BG = {
        trend: 'rgba(59,130,246,0.08)', term_structure: 'rgba(245,158,11,0.08)',
        mean_reversion: 'rgba(139,92,246,0.08)', vol_breakout: 'rgba(6,182,212,0.08)',
        cross_sectional: 'rgba(16,185,129,0.08)', fusion: 'rgba(16,185,129,0.08)'
    };
    var firstKey = keys[0];
    var labels = ad[firstKey].dates;
    var datasets = keys.map(function(k) {
        var info = ad[k];
        return {
            label: DD_LABELS[k] || k,
            data: info.drawdown,
            borderColor: DD_COLORS[k] || '#666',
            backgroundColor: DD_BG[k] || 'rgba(102,102,102,0.05)',
            fill: true, borderWidth: 1.5, pointRadius: 0, tension: 0.2,
        };
    });
    var summaryHtml = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;">';
    keys.forEach(function(k) {
        var info = ad[k];
        var color = DD_COLORS[k] || '#666';
        var label = DD_LABELS[k] || k;
        summaryHtml += '<span style="display:inline-flex;align-items:center;gap:4px;padding:4px 10px;border-radius:6px;font-size:12px;background:' + color + '15;border:1px solid ' + color + '40;">'
            + '<span style="width:8px;height:8px;border-radius:50%;background:' + color + ';display:inline-block;"></span>'
            + '<strong>' + label + '</strong>'
            + '<span style="color:#666;">最大回撤: ' + info.max_dd_pct + '%</span>'
            + '<span style="color:#666;">持续: ' + info.duration_days + '天</span>'
            + '</span>';
    });
    summaryHtml += '</div>';
    var container = document.getElementById('chartAllDrawdowns').parentElement;
    container.insertAdjacentHTML('afterbegin', summaryHtml);
    new Chart(document.getElementById('chartAllDrawdowns'), {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                tooltip: { callbacks: { label: function(ctx) { return ctx.dataset.label + ' 回撤: ' + ctx.parsed.y.toFixed(2) + '%'; } } },
                legend: { position: 'top', labels: { usePointStyle: true, padding: 16 } },
            },
            scales: {
                x: { type: 'category', ticks: { maxTicksLimit: 12, autoSkip: true } },
                y: { title: { display: true, text: '回撤 (%)' }, max: 0, ticks: { callback: function(v) { return v.toFixed(0) + '%'; } } },
            },
        },
    });
})();

(function() {
    var sc = reportData.risk_return;
    if (!sc || !sc.length) return;
    var datasets = sc.map(function(d) {
        return {
            label: d.name + ' (Sharpe=' + d.sharpe.toFixed(3) + ')',
            data: [{ x: d.ann_volatility, y: d.ann_return, sharpe: d.sharpe }],
            backgroundColor: COLORS[d.key] || '#666',
            borderColor: COLORS[d.key] || '#666',
            pointRadius: 8, pointHoverRadius: 12,
        };
    });
    new Chart(document.getElementById('chartScatter'), {
        type: 'scatter',
        data: { datasets: datasets },
        options: {
            responsive: true,
            plugins: {
                tooltip: { callbacks: { label: function(ctx) { return ctx.dataset.label.split(' (')[0] + ': 波动率=' + ctx.parsed.x.toFixed(2) + '%, 收益=' + ctx.parsed.y.toFixed(2) + '%, Sharpe=' + ctx.raw.sharpe.toFixed(3); } } },
                legend: { position: 'top', labels: { usePointStyle: true } },
            },
            scales: {
                x: { title: { display: true, text: '年化波动率 (%)' } },
                y: { title: { display: true, text: '年化收益率 (%)' } },
            },
        },
    });
})();

// 月度收益率热力图（主策略）
(function() {
    var hm = reportData.heatmap_data;
    var yrs = reportData.years_set;
    if (!hm || !yrs || !yrs.length) return;
    var canvas = document.getElementById('chartHeatmap');
    var months = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'];
    var cellW = 62, cellH = 28, leftPad = 68, topPad = 40;
    canvas.width = leftPad + 12 * cellW + 20;
    canvas.height = topPad + yrs.length * cellH + 40;
    canvas.style.width = '100%';
    canvas.style.height = 'auto';
    var ctx = canvas.getContext('2d');
    ctx.font = '11px ' + CHART_FONT;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#475569';
    for (var yi = 0; yi < yrs.length; yi++) {
        ctx.fillText(yrs[yi], leftPad - 8, topPad + yi * cellH + cellH/2);
    }
    ctx.textAlign = 'center';
    for (var mi = 0; mi < 12; mi++) {
        ctx.fillText(months[mi], leftPad + mi * cellW + cellW/2, topPad - 14);
    }
    function heatColor(val) {
        if (val === null || val === undefined) return '#f1f5f9';
        var maxAbs = 15;
        var ratio = Math.max(-1, Math.min(1, val / maxAbs));
        if (ratio >= 0) {
            var r = Math.round(34 + (1-ratio) * 221);
            var g = Math.round(197 + (1-ratio) * 58);
            var b = Math.round(94 + (1-ratio) * 161);
            return 'rgb(' + r + ',' + g + ',' + b + ')';
        } else {
            var r2 = Math.round(220 + (1+ratio) * 35);
            var g2 = Math.round(38 + (1+ratio) * 62);
            var b2 = Math.round(38 + (1+ratio) * 62);
            return 'rgb(' + r2 + ',' + g2 + ',' + b2 + ')';
        }
    }
    for (var yi = 0; yi < yrs.length; yi++) {
        for (var mi = 0; mi < 12; mi++) {
            var val = hm[yi] && hm[yi][mi] !== undefined ? hm[yi][mi] : null;
            ctx.fillStyle = heatColor(val);
            ctx.fillRect(leftPad + mi * cellW, topPad + yi * cellH, cellW - 1, cellH - 1);
            if (val !== null) {
                ctx.fillStyle = Math.abs(val) > 8 ? '#fff' : '#1a1a2e';
                ctx.fillText(val.toFixed(1) + '%', leftPad + mi * cellW + cellW/2, topPad + yi * cellH + cellH/2);
            }
        }
    }
    var legendY = topPad + yrs.length * cellH + 26;
    ctx.textAlign = 'left';
    for (var i = 0; i <= 10; i++) {
        var t = (i - 5) / 5 * 15;
        ctx.fillStyle = heatColor(t);
        ctx.fillRect(leftPad + i * 32, legendY, 28, 14);
        if (i % 2 === 0) {
            ctx.fillStyle = '#475569';
            ctx.fillText(t.toFixed(0) + '%', leftPad + i * 32 + 14, legendY + 26);
        }
    }
})();

// 所有策略月度收益率热力图
(function() {
    var ahm = reportData.all_heatmaps;
    if (!ahm) return;
    var keys = Object.keys(ahm);
    if (!keys.length) return;
    var HM_COLORS = {
        trend: '#3b82f6', term_structure: '#f59e0b', mean_reversion: '#8b5cf6',
        vol_breakout: '#06b6d4', cross_sectional: '#10b981', fusion: '#10b981'
    };
    var HM_LABELS = {
        trend: '趋势策略', term_structure: '期限结构', mean_reversion: '均值回归',
        vol_breakout: '波动率突破', cross_sectional: '横截面打分', fusion: '融合策略'
    };
    var months = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'];
    var container = document.getElementById('allHeatmapsContainer');
    function heatColor(val) {
        if (val === null || val === undefined) return '#f1f5f9';
        var maxAbs = 15;
        var ratio = Math.max(-1, Math.min(1, val / maxAbs));
        if (ratio >= 0) {
            var r = Math.round(34 + (1-ratio) * 221);
            var g = Math.round(197 + (1-ratio) * 58);
            var b = Math.round(94 + (1-ratio) * 161);
            return 'rgb(' + r + ',' + g + ',' + b + ')';
        } else {
            var r2 = Math.round(220 + (1+ratio) * 35);
            var g2 = Math.round(38 + (1+ratio) * 62);
            var b2 = Math.round(38 + (1+ratio) * 62);
            return 'rgb(' + r2 + ',' + g2 + ',' + b2 + ')';
        }
    }
    function drawHeatmap(canvasEl, hm, yrs, label, color) {
        var cellW = 56, cellH = 24, leftPad = 60, topPad = 36;
        canvasEl.width = leftPad + 12 * cellW + 20;
        canvasEl.height = topPad + yrs.length * cellH + 50;
        canvasEl.style.width = '100%';
        canvasEl.style.height = 'auto';
        var ctx2 = canvasEl.getContext('2d');
        ctx2.font = '10px ' + CHART_FONT;
        ctx2.textAlign = 'right';
        ctx2.textBaseline = 'middle';
        ctx2.fillStyle = '#475569';
        for (var yi = 0; yi < yrs.length; yi++) {
            ctx2.fillText(yrs[yi], leftPad - 6, topPad + yi * cellH + cellH/2);
        }
        ctx2.textAlign = 'center';
        for (var mi = 0; mi < 12; mi++) {
            ctx2.fillText(months[mi], leftPad + mi * cellW + cellW/2, topPad - 12);
        }
        for (var yi2 = 0; yi2 < yrs.length; yi2++) {
            for (var mi2 = 0; mi2 < 12; mi2++) {
                var val = hm[yi2] && hm[yi2][mi2] !== undefined ? hm[yi2][mi2] : null;
                ctx2.fillStyle = heatColor(val);
                ctx2.fillRect(leftPad + mi2 * cellW, topPad + yi2 * cellH, cellW - 1, cellH - 1);
                if (val !== null) {
                    ctx2.fillStyle = Math.abs(val) > 8 ? '#fff' : '#1a1a2e';
                    ctx2.fillText(val.toFixed(1) + '%', leftPad + mi2 * cellW + cellW/2, topPad + yi2 * cellH + cellH/2);
                }
            }
        }
        var legendY2 = topPad + yrs.length * cellH + 20;
        ctx2.textAlign = 'left';
        for (var li = 0; li <= 10; li++) {
            var t2 = (li - 5) / 5 * 15;
            ctx2.fillStyle = heatColor(t2);
            ctx2.fillRect(leftPad + li * 28, legendY2, 24, 12);
            if (li % 2 === 0) {
                ctx2.fillStyle = '#475569';
                ctx2.fillText(t2.toFixed(0) + '%', leftPad + li * 28 + 12, legendY2 + 22);
            }
        }
    }
    var html = '<div style="font-size:14px;font-weight:600;margin:16px 0 8px;">&#x1f321;&#xfe0f; 所有策略月度收益率热力图</div>';
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">';
    keys.forEach(function(k) {
        var info = ahm[k];
        var label = HM_LABELS[k] || k;
        var color = HM_COLORS[k] || '#666';
        var canvasId = 'chartHeatmap_' + k;
        html += '<div class="chart-box"><div style="font-size:12px;font-weight:600;margin-bottom:8px;color:' + color + ';">' + label + '</div><canvas id="' + canvasId + '"></canvas></div>';
    });
    html += '</div>';
    container.innerHTML = html;
    keys.forEach(function(k) {
        var info = ahm[k];
        var canvasEl = document.getElementById('chartHeatmap_' + k);
        if (canvasEl) drawHeatmap(canvasEl, info.data, info.years_set, HM_LABELS[k] || k, HM_COLORS[k] || '#666');
    });
})();

// 滚动夏普
(function() {
    var ars = reportData.all_rolling_sharpe;
    if (!ars) return;
    var keys = Object.keys(ars);
    if (!keys.length) return;
    var RS_COLORS = {
        trend: '#3b82f6', term_structure: '#f59e0b', mean_reversion: '#8b5cf6',
        vol_breakout: '#06b6d4', cross_sectional: '#10b981', fusion: '#10b981'
    };
    var RS_LABELS = {
        trend: '趋势策略', term_structure: '期限结构', mean_reversion: '均值回归',
        vol_breakout: '波动率突破', cross_sectional: '横截面打分', fusion: '融合策略'
    };
    var firstKey = keys[0];
    var labels = ars[firstKey].dates;
    var datasets = keys.map(function(k) {
        return {
            label: RS_LABELS[k] || k,
            data: ars[k].values,
            borderColor: RS_COLORS[k] || '#666',
            backgroundColor: 'transparent',
            borderWidth: 1.5, pointRadius: 0, tension: 0.2,
        };
    });
    new Chart(document.getElementById('chartAllRollingSharpe'), {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                tooltip: { callbacks: { label: function(ctx) { return ctx.dataset.label + ' 夏普: ' + ctx.parsed.y.toFixed(4); } } },
                legend: { position: 'top', labels: { usePointStyle: true, padding: 16 } },
                annotation: { annotations: { zeroLine: { type: 'line', yMin: 0, yMax: 0, borderColor: '#94a3b8', borderWidth: 1, borderDash: [4,4] } } },
            },
            scales: {
                x: { type: 'category', ticks: { maxTicksLimit: 12, autoSkip: true } },
                y: { title: { display: true, text: '夏普比率' } },
            },
        },
    });
})();

// 滚动回撤
(function() {
    var ard = reportData.all_rolling_dd;
    if (!ard) return;
    var keys = Object.keys(ard);
    if (!keys.length) return;
    var RDD_COLORS = {
        trend: '#3b82f6', term_structure: '#f59e0b', mean_reversion: '#8b5cf6',
        vol_breakout: '#06b6d4', cross_sectional: '#10b981', fusion: '#10b981'
    };
    var RDD_LABELS = {
        trend: '趋势策略', term_structure: '期限结构', mean_reversion: '均值回归',
        vol_breakout: '波动率突破', cross_sectional: '横截面打分', fusion: '融合策略'
    };
    var RDD_BG = {
        trend: 'rgba(59,130,246,0.06)', term_structure: 'rgba(245,158,11,0.06)',
        mean_reversion: 'rgba(139,92,246,0.06)', vol_breakout: 'rgba(6,182,212,0.06)',
        cross_sectional: 'rgba(16,185,129,0.06)', fusion: 'rgba(16,185,129,0.06)'
    };
    var firstKey = keys[0];
    var labels = ard[firstKey].dates;
    var datasets = keys.map(function(k) {
        return {
            label: RDD_LABELS[k] || k,
            data: ard[k].values,
            borderColor: RDD_COLORS[k] || '#666',
            backgroundColor: RDD_BG[k] || 'rgba(102,102,102,0.05)',
            fill: true, borderWidth: 1.5, pointRadius: 0, tension: 0.2,
        };
    });
    new Chart(document.getElementById('chartAllRollingDD'), {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                tooltip: { callbacks: { label: function(ctx) { return ctx.dataset.label + ' 回撤: ' + ctx.parsed.y.toFixed(2) + '%'; } } },
                legend: { position: 'top', labels: { usePointStyle: true, padding: 16 } },
            },
            scales: {
                x: { type: 'category', ticks: { maxTicksLimit: 12, autoSkip: true } },
                y: { title: { display: true, text: '最大回撤 (%)' }, max: 0, ticks: { callback: function(v) { return v.toFixed(0) + '%'; } } },
            },
        },
    });
})();

// 相关性热力图
(function() {
    var corr = reportData.correlation;
    if (!corr || !corr.names || !corr.names.length) return;
    var names = corr.names;
    var n = names.length;
    var datasets = [];
    var bgColors = ['#3b82f6','#f59e0b','#8b5cf6','#10b981','#ef4444','#ec4899','#06b6d4','#84cc16'];
    for (var i = 0; i < n; i++) {
        datasets.push({
            label: names[i],
            data: corr.matrix[i],
            backgroundColor: bgColors[i % bgColors.length] + '80',
            borderColor: bgColors[i % bgColors.length],
            borderWidth: 1,
        });
    }
    new Chart(document.getElementById('chartCorr'), {
        type: 'bar',
        data: { labels: names, datasets: datasets },
        options: {
            responsive: true,
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { stacked: false },
                y: { min: -1, max: 1, title: { display: true, text: '相关性系数' } },
            },
        },
    });
})();

// 样本内净值
(function() {
    var aie = reportData.all_is_equity;
    if (!aie) return;
    var keys = Object.keys(aie);
    if (!keys.length) return;
    var IS_COLORS = {
        trend: '#3b82f6', term_structure: '#f59e0b', mean_reversion: '#8b5cf6',
        vol_breakout: '#06b6d4', cross_sectional: '#10b981', fusion: '#10b981'
    };
    var IS_LABELS = {
        trend: '趋势策略', term_structure: '期限结构', mean_reversion: '均值回归',
        vol_breakout: '波动率突破', cross_sectional: '横截面打分', fusion: '融合策略'
    };
    var firstKey = keys[0];
    var labels = aie[firstKey].dates;
    var datasets = keys.map(function(k) {
        return {
            label: IS_LABELS[k] || k,
            data: aie[k].equity,
            borderColor: IS_COLORS[k] || '#666',
            backgroundColor: 'transparent',
            borderWidth: 1.5, pointRadius: 0, tension: 0.1,
        };
    });
    new Chart(document.getElementById('chartAllIS'), {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                tooltip: { callbacks: { label: function(ctx) { return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(4); } } },
                legend: { position: 'top', labels: { usePointStyle: true, padding: 12 } },
            },
            scales: {
                x: { type: 'category', ticks: { maxTicksLimit: 8, autoSkip: true } },
                y: { title: { display: true, text: '净值 (归一化)' } },
            },
        },
    });
})();

// 样本外净值
(function() {
    var aoe = reportData.all_os_equity;
    if (!aoe) return;
    var keys = Object.keys(aoe);
    if (!keys.length) return;
    var OS_COLORS = {
        trend: '#3b82f6', term_structure: '#f59e0b', mean_reversion: '#8b5cf6',
        vol_breakout: '#06b6d4', cross_sectional: '#10b981', fusion: '#10b981'
    };
    var OS_LABELS = {
        trend: '趋势策略', term_structure: '期限结构', mean_reversion: '均值回归',
        vol_breakout: '波动率突破', cross_sectional: '横截面打分', fusion: '融合策略'
    };
    var firstKey = keys[0];
    var labels = aoe[firstKey].dates;
    var datasets = keys.map(function(k) {
        return {
            label: OS_LABELS[k] || k,
            data: aoe[k].equity,
            borderColor: OS_COLORS[k] || '#666',
            backgroundColor: 'transparent',
            borderWidth: 1.5, pointRadius: 0, tension: 0.1,
        };
    });
    new Chart(document.getElementById('chartAllOS'), {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                tooltip: { callbacks: { label: function(ctx) { return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(4); } } },
                legend: { position: 'top', labels: { usePointStyle: true, padding: 12 } },
            },
            scales: {
                x: { type: 'category', ticks: { maxTicksLimit: 8, autoSkip: true } },
                y: { title: { display: true, text: '净值 (归一化)' } },
            },
        },
    });
})();

// 多品种分散化净值图表
(function() {
    if (!reportData.diversification || !reportData.diversification.equity_curves) return;
    var divEquities = reportData.diversification.equity_curves;
    var divNames = Object.keys(divEquities);
    if (!divNames.length) return;
    var firstKey = divNames[0];
    var labels = divEquities[firstKey].dates;
    var COLORS = [
        '#3b82f6', '#10b981', '#f59e0b', '#ef4444',
        '#8b5cf6', '#06b6d4', '#ec4899', '#6366f1'
    ];
    var datasets = divNames.map(function(name, idx) {
        return {
            label: name,
            data: divEquities[name].equity,
            borderColor: COLORS[idx % COLORS.length],
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1
        };
    });
    new Chart(document.getElementById('diversificationEquityChart'), {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'top' },
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(4);
                        }
                    }
                }
            },
            scales: {
                x: { type: 'category', ticks: { maxTicksLimit: 12, autoSkip: true } },
                y: { title: { display: true, text: '净值（归一化）' } }
            }
        }
    });
})();
"""

