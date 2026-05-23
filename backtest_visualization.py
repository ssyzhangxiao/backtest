"""
回测结果可视化分析

生成详细的图表和分析报告
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')


def load_results(result_dir: str) -> Dict:
    """
    加载回测结果
    
    Args:
        result_dir: 结果目录
    
    Returns:
        结果字典
    """
    print(f"正在加载结果从: {result_dir}")
    
    results = {}
    
    # 加载对比表格
    comparison_df = pd.read_csv(f"{result_dir}/comparison_table.csv", index_col=0)
    results['comparison'] = comparison_df
    
    # 加载配置
    with open(f"{result_dir}/config.json", 'r', encoding='utf-8') as f:
        results['config'] = json.load(f)
    
    # 加载组合净值
    try:
        combined_portfolio = pd.read_csv(f"{result_dir}/combined_portfolio.csv")
        results['combined_portfolio'] = combined_portfolio
    except:
        pass
    
    # 加载各策略数据
    results['portfolios'] = {}
    results['trades'] = {}
    
    for variant_id in ['original', 'variant_a', 'variant_b', 'variant_c']:
        try:
            portfolio = pd.read_csv(f"{result_dir}/portfolio_{variant_id}.csv")
            results['portfolios'][variant_id] = portfolio
        except:
            pass
        
        try:
            trades = pd.read_csv(f"{result_dir}/trades_{variant_id}.csv")
            results['trades'][variant_id] = trades
        except:
            pass
    
    return results


def plot_equity_curves(results: Dict, output_dir: str):
    """
    绘制净值曲线对比图
    
    Args:
        results: 回测结果
        output_dir: 输出目录
    """
    print("绘制净值曲线...")
    
    if 'combined_portfolio' not in results:
        print("缺少组合净值数据")
        return
    
    df = results['combined_portfolio'].copy()
    df['date'] = pd.to_datetime(df['date'])
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    color_idx = 0
    
    for strategy in df['strategy'].unique():
        strat_data = df[df['strategy'] == strategy].sort_values('date')
        
        # 计算归一化净值（从1开始）
        if 'equity' in strat_data.columns:
            equity = strat_data['equity']
        elif 'market_value' in strat_data.columns:
            equity = strat_data['market_value']
        else:
            continue
        
        equity_normalized = equity / equity.iloc[0]
        
        fig.add_trace(go.Scatter(
            x=strat_data['date'],
            y=equity_normalized,
            name=strategy,
            line=dict(width=2, color=colors[color_idx % len(colors)]),
            mode='lines'
        ))
        color_idx += 1
    
    fig.update_layout(
        title='策略净值曲线对比（归一化）',
        xaxis_title='日期',
        yaxis_title='净值（初始=1）',
        hovermode='x unified',
        template='plotly_white',
        height=600
    )
    
    fig.write_html(f"{output_dir}/equity_curves.html")
    # 跳过 PNG 导出避免依赖问题
    # fig.write_image(f"{output_dir}/equity_curves.png", width=1200, height=600, scale=2)


def plot_drawdown_curves(results: Dict, output_dir: str):
    """
    绘制回撤曲线对比图
    
    Args:
        results: 回测结果
        output_dir: 输出目录
    """
    print("绘制回撤曲线...")
    
    if 'combined_portfolio' not in results:
        print("缺少组合净值数据")
        return
    
    df = results['combined_portfolio'].copy()
    df['date'] = pd.to_datetime(df['date'])
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    color_idx = 0
    
    for strategy in df['strategy'].unique():
        strat_data = df[df['strategy'] == strategy].sort_values('date')
        
        if 'equity' in strat_data.columns:
            equity = strat_data['equity']
        elif 'market_value' in strat_data.columns:
            equity = strat_data['market_value']
        else:
            continue
        
        # 计算回撤
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max * 100
        
        fig.add_trace(go.Scatter(
            x=strat_data['date'],
            y=drawdown,
            name=strategy,
            fill='tozeroy',
            line=dict(width=1.5, color=colors[color_idx % len(colors)]),
            mode='lines'
        ))
        color_idx += 1
    
    fig.update_layout(
        title='策略回撤曲线对比',
        xaxis_title='日期',
        yaxis_title='回撤（%）',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    fig.write_html(f"{output_dir}/drawdown_curves.html")
    # 跳过 PNG 导出避免依赖问题
    # fig.write_image(f"{output_dir}/drawdown_curves.png", width=1200, height=500, scale=2)


def plot_performance_radar(results: Dict, output_dir: str):
    """
    绘制绩效雷达图
    
    Args:
        results: 回测结果
        output_dir: 输出目录
    """
    print("绘制绩效雷达图...")
    
    comparison_df = results['comparison']
    
    # 选择指标
    metrics = ['总收益率(%)', 'Sharpe比率', 'Sortino比率', 'Calmar比率', '胜率(%)']
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    color_idx = 0
    
    for strategy in comparison_df.index:
        # 归一化指标
        values = []
        for metric in metrics:
            val = comparison_df.loc[strategy, metric]
            values.append(val)
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            name=strategy,
            fill='toself',
            line=dict(color=colors[color_idx % len(colors)]),
            marker=dict(color=colors[color_idx % len(colors)], size=8)
        ))
        color_idx += 1
    
    fig.update_layout(
        title='策略绩效雷达图',
        polar=dict(
            radialaxis=dict(showticklabels=True, ticks=''),
            angularaxis=dict(tickfont=dict(size=12))
        ),
        template='plotly_white',
        height=700
    )
    
    fig.write_html(f"{output_dir}/performance_radar.html")
    # 跳过 PNG 导出避免依赖问题
    # fig.write_image(f"{output_dir}/performance_radar.png", width=1000, height=700, scale=2)


def plot_trade_analysis(results: Dict, output_dir: str):
    """
    绘制交易分析图
    
    Args:
        results: 回测结果
        output_dir: 输出目录
    """
    print("绘制交易分析...")
    
    variant_names = {
        'original': '原策略',
        'variant_a': '策略A',
        'variant_b': '策略B',
        'variant_c': '策略C'
    }
    
    # 检查是否有交易数据
    has_trades = False
    for variant_id in variant_names.keys():
        if variant_id in results['trades'] and len(results['trades'][variant_id]) > 0:
            has_trades = True
            break
    
    if not has_trades:
        print("缺少交易数据")
        return
    
    # 交易次数对比
    comparison_df = results['comparison']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('交易次数', '胜率与盈亏比', '年化收益率vs最大回撤', '持仓周期'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # 1. 交易次数
    fig.add_trace(
        go.Bar(
            x=comparison_df.index,
            y=comparison_df['交易次数'],
            name='交易次数',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        ),
        row=1, col=1
    )
    
    # 2. 胜率与盈亏比
    fig.add_trace(
        go.Bar(
            x=comparison_df.index,
            y=comparison_df['胜率(%)'],
            name='胜率(%)',
            marker_color='#1f77b4'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=comparison_df.index,
            y=comparison_df['盈亏比'],
            name='盈亏比',
            yaxis='y2',
            mode='markers+lines',
            marker=dict(size=10, color='#ff7f0e'),
            line=dict(color='#ff7f0e', width=2)
        ),
        row=1, col=2
    )
    
    # 3. 年化收益率vs最大回撤
    fig.add_trace(
        go.Scatter(
            x=comparison_df['最大回撤(%)'].abs(),
            y=comparison_df['年化收益率(%)'],
            text=comparison_df.index,
            mode='markers+text',
            textposition='top center',
            marker=dict(
                size=15,
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                line=dict(color='black', width=1)
            )
        ),
        row=2, col=1
    )
    
    # 4. 持仓周期
    fig.add_trace(
        go.Bar(
            x=comparison_df.index,
            y=comparison_df['平均持仓天数'],
            name='平均持仓天数',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title='交易分析综合图表',
        height=900,
        template='plotly_white',
        showlegend=False
    )
    
    fig.update_xaxes(title_text='策略', row=1, col=1)
    fig.update_xaxes(title_text='策略', row=1, col=2)
    fig.update_xaxes(title_text='最大回撤绝对值(%)', row=2, col=1)
    fig.update_xaxes(title_text='策略', row=2, col=2)
    
    fig.update_yaxes(title_text='交易次数', row=1, col=1)
    fig.update_yaxes(title_text='胜率(%)', row=1, col=2)
    fig.update_yaxes(title_text='年化收益率(%)', row=2, col=1)
    fig.update_yaxes(title_text='持仓天数', row=2, col=2)
    
    # 添加第二个Y轴
    fig.update_layout(
        yaxis2=dict(
            title='盈亏比',
            overlaying='y',
            side='right',
            anchor='x2'
        )
    )
    
    fig.write_html(f"{output_dir}/trade_analysis.html")
    # 跳过 PNG 导出避免依赖问题
    # fig.write_image(f"{output_dir}/trade_analysis.png", width=1200, height=900, scale=2)


def plot_returns_distribution(results: Dict, output_dir: str):
    """
    绘制收益率分布
    
    Args:
        results: 回测结果
        output_dir: 输出目录
    """
    print("绘制收益率分布...")
    
    if 'portfolios' not in results or len(results['portfolios']) == 0:
        print("缺少组合数据")
        return
    
    variant_names = {
        'original': '原策略',
        'variant_a': '策略A',
        'variant_b': '策略B',
        'variant_c': '策略C'
    }
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[variant_names[v] for v in variant_names.keys()]
    )
    
    row, col = 1, 1
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    color_idx = 0
    
    for variant_id, variant_name in variant_names.items():
        if variant_id not in results['portfolios']:
            continue
        
        df = results['portfolios'][variant_id]
        
        if 'equity' in df.columns:
            equity = df['equity']
        elif 'market_value' in df.columns:
            equity = df['market_value']
        else:
            continue
        
        returns = equity.pct_change().dropna() * 100
        
        fig.add_trace(
            go.Histogram(
                x=returns,
                name=variant_name,
                nbinsx=50,
                marker_color=colors[color_idx % len(colors)],
                opacity=0.7
            ),
            row=row, col=col
        )
        
        color_idx += 1
        col += 1
        if col > 2:
            col = 1
            row += 1
    
    fig.update_layout(
        title='日收益率分布直方图',
        height=800,
        template='plotly_white'
    )
    
    for i in range(1, 5):
        fig.update_xaxes(title_text='日收益率(%)', row=(i-1)//2 + 1, col=(i-1)%2 + 1)
        fig.update_yaxes(title_text='频次', row=(i-1)//2 + 1, col=(i-1)%2 + 1)
    
    fig.write_html(f"{output_dir}/returns_distribution.html")
    # 跳过 PNG 导出避免依赖问题
    # fig.write_image(f"{output_dir}/returns_distribution.png", width=1000, height=800, scale=2)


def compute_yearly_returns(portfolios: Dict, variant_names: Dict = None) -> pd.DataFrame:
    """
    计算各策略逐年收益率
    
    Args:
        portfolios: 各策略的 portfolio DataFrame
        variant_names: 策略名称映射
    
    Returns:
        逐年收益率 DataFrame，列为策略名，索引为年份
    """
    if variant_names is None:
        variant_names = {
            'original': '原策略',
            'variant_a': '策略A',
            'variant_b': '策略B',
            'variant_c': '策略C'
        }
    
    yearly_data = {}
    
    for variant_id, variant_name in variant_names.items():
        if variant_id not in portfolios:
            continue
        
        df = portfolios[variant_id].copy()
        
        if 'date' not in df.columns:
            continue
        
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        
        if 'equity' in df.columns:
            equity_col = 'equity'
        elif 'market_value' in df.columns:
            equity_col = 'market_value'
        else:
            continue
        
        yearly_eq = df.groupby('year')[equity_col].agg(['first', 'last'])
        yearly_eq['return_pct'] = (yearly_eq['last'] / yearly_eq['first'] - 1) * 100
        yearly_data[variant_name] = yearly_eq['return_pct']
    
    if not yearly_data:
        return pd.DataFrame()
    
    result = pd.DataFrame(yearly_data)
    result.index.name = '年份'
    
    return result


def plot_yearly_returns(results: Dict, output_dir: str):
    """
    绘制逐年收益率柱状图（含盈亏高亮、趋势线、策略失效区间标记）
    
    Args:
        results: 回测结果
        output_dir: 输出目录
    """
    print("绘制逐年收益率...")
    
    if 'portfolios' not in results or len(results['portfolios']) == 0:
        print("缺少组合数据")
        return
    
    variant_names = {
        'original': '原策略',
        'variant_a': '策略A',
        'variant_b': '策略B',
        'variant_c': '策略C'
    }
    
    yearly_df = compute_yearly_returns(results['portfolios'], variant_names)
    
    if yearly_df.empty:
        print("无法计算逐年收益率")
        return
    
    yearly_df.to_csv(f"{output_dir}/yearly_returns.csv", encoding='utf-8-sig')
    
    strategies = yearly_df.columns.tolist()
    n_strategies = len(strategies)
    years = yearly_df.index.tolist()
    
    fig = make_subplots(
        rows=n_strategies, cols=1,
        subplot_titles=[f'{s} 逐年收益率' for s in strategies],
        vertical_spacing=0.08
    )
    
    colors_map = {
        '原策略': '#1f77b4',
        '策略A': '#ff7f0e',
        '策略B': '#2ca02c',
        '策略C': '#d62728'
    }
    
    for idx, strategy in enumerate(strategies):
        row = idx + 1
        values = yearly_df[strategy].values
        
        bar_colors = ['#2ca02c' if v > 0 else '#d62728' for v in values]
        
        fig.add_trace(
            go.Bar(
                x=years,
                y=values,
                name=strategy,
                marker_color=bar_colors,
                text=[f'{v:.2f}%' for v in values],
                textposition='outside',
                textfont=dict(size=10),
                showlegend=False,
                hovertemplate='%{x}年: %{y:.2f}%<extra></extra>'
            ),
            row=row, col=1
        )
        
        fig.add_hline(y=0, line_color='gray', line_width=0.8, row=row, col=1)
        
        mean_return = values.mean()
        fig.add_hline(
            y=mean_return, line_dash='dash', line_color=colors_map.get(strategy, '#333'),
            line_width=1.5,
            annotation_text=f'均值 {mean_return:.2f}%',
            annotation_position='top right',
            annotation_font=dict(size=10),
            row=row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=values,
                mode='lines',
                line=dict(
                    color=colors_map.get(strategy, '#333'),
                    width=2,
                    dash='dot'
                ),
                name=f'{strategy} 趋势',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=1
        )
        
        consecutive_loss = []
        loss_count = 0
        for v in values:
            if v < 0:
                loss_count += 1
                consecutive_loss.append(loss_count)
            else:
                loss_count = 0
                consecutive_loss.append(0)
        
        for i, count in enumerate(consecutive_loss):
            if count >= 2:
                fig.add_vrect(
                    x0=years[i] - 0.4, x1=years[i] + 0.4,
                    fillcolor='rgba(214, 39, 40, 0.1)',
                    line_width=0,
                    row=row, col=1
                )
    
    fig.update_layout(
        title='策略逐年收益率分析（绿色=盈利年份，红色=亏损年份，浅红底色=连续亏损区间）',
        height=250 * n_strategies + 100,
        template='plotly_white',
        hovermode='x unified'
    )
    
    for idx in range(n_strategies):
        fig.update_yaxes(title_text='收益率(%)', row=idx + 1, col=1)
    fig.update_xaxes(title_text='年份', row=n_strategies, col=1)
    
    fig.write_html(f"{output_dir}/yearly_returns.html")


def plot_yearly_returns_comparison(results: Dict, output_dir: str):
    """
    绘制逐年收益率对比柱状图（所有策略并排对比）
    
    Args:
        results: 回测结果
        output_dir: 输出目录
    """
    print("绘制逐年收益率对比...")
    
    if 'portfolios' not in results or len(results['portfolios']) == 0:
        print("缺少组合数据")
        return
    
    variant_names = {
        'original': '原策略',
        'variant_a': '策略A',
        'variant_b': '策略B',
        'variant_c': '策略C'
    }
    
    yearly_df = compute_yearly_returns(results['portfolios'], variant_names)
    
    if yearly_df.empty:
        print("无法计算逐年收益率")
        return
    
    strategies = yearly_df.columns.tolist()
    years = yearly_df.index.tolist()
    
    colors_map = {
        '原策略': '#1f77b4',
        '策略A': '#ff7f0e',
        '策略B': '#2ca02c',
        '策略C': '#d62728'
    }
    
    fig = go.Figure()
    
    for strategy in strategies:
        values = yearly_df[strategy].values
        fig.add_trace(go.Bar(
            name=strategy,
            x=years,
            y=values,
            marker_color=colors_map.get(strategy, '#333'),
            text=[f'{v:.1f}%' for v in values],
            textposition='outside',
            textfont=dict(size=9),
            hovertemplate=f'{strategy} %{{x}}年: %{{y:.2f}}%<extra></extra>'
        ))
    
    all_values = yearly_df.values.flatten()
    all_negative = all_values[all_values < 0]
    if len(all_negative) > 0:
        worst_year = yearly_df.min(axis=1).idxmin()
        worst_val = yearly_df.min(axis=1).min()
        fig.add_annotation(
            x=worst_year, y=worst_val,
            text=f'最差年份: {worst_year}年 ({worst_val:.1f}%)',
            showarrow=True, arrowhead=2,
            font=dict(color='red', size=12)
        )
    
    all_positive = all_values[all_values > 0]
    if len(all_positive) > 0:
        best_idx = np.unravel_index(np.argmax(all_values), yearly_df.shape)
        best_year = years[best_idx[0]]
        best_val = all_values.max()
        best_strategy = strategies[best_idx[1]]
        fig.add_annotation(
            x=best_year, y=best_val,
            text=f'最佳: {best_year}年 {best_strategy} ({best_val:.1f}%)',
            showarrow=True, arrowhead=2,
            font=dict(color='green', size=12)
        )
    
    fig.add_hline(y=0, line_color='gray', line_width=1)
    
    fig.update_layout(
        title='策略逐年收益率对比（分组柱状图）',
        xaxis_title='年份',
        yaxis_title='收益率(%)',
        barmode='group',
        template='plotly_white',
        height=600,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    fig.write_html(f"{output_dir}/yearly_returns_comparison.html")


def generate_yearly_returns_html_table(results: Dict) -> str:
    """
    生成逐年收益率的HTML表格（含盈亏高亮和统计摘要）
    
    Args:
        results: 回测结果
    
    Returns:
        HTML表格字符串
    """
    variant_names = {
        'original': '原策略',
        'variant_a': '策略A',
        'variant_b': '策略B',
        'variant_c': '策略C'
    }
    
    yearly_df = compute_yearly_returns(results['portfolios'], variant_names)
    
    if yearly_df.empty:
        return '<p>无逐年收益率数据</p>'
    
    strategies = yearly_df.columns.tolist()
    
    html = '<table><thead><tr><th>年份</th>'
    for s in strategies:
        html += f'<th>{s}</th>'
    html += '</tr></thead><tbody>'
    
    for year in yearly_df.index:
        html += f'<tr><td style="text-align:center;font-weight:bold;">{year}</td>'
        for s in strategies:
            val = yearly_df.loc[year, s]
            if val > 0:
                bg = 'background-color:#d4edda;color:#155724;'
            elif val < 0:
                bg = 'background-color:#f8d7da;color:#721c24;'
            else:
                bg = ''
            html += f'<td style="text-align:right;{bg}">{val:.2f}%</td>'
        html += '</tr>'
    
    html += '<tr style="font-weight:bold;border-top:2px solid #333;"><td style="text-align:center;">平均</td>'
    for s in strategies:
        avg = yearly_df[s].mean()
        if avg > 0:
            bg = 'background-color:#d4edda;color:#155724;'
        elif avg < 0:
            bg = 'background-color:#f8d7da;color:#721c24;'
        else:
            bg = ''
        html += f'<td style="text-align:right;{bg}">{avg:.2f}%</td>'
    html += '</tr>'
    
    html += '<tr style="font-weight:bold;"><td style="text-align:center;">盈利年数</td>'
    for s in strategies:
        profit_years = (yearly_df[s] > 0).sum()
        total_years = len(yearly_df)
        pct = profit_years / total_years * 100
        html += f'<td style="text-align:right;">{profit_years}/{total_years} ({pct:.0f}%)</td>'
    html += '</tr>'
    
    html += '<tr style="font-weight:bold;"><td style="text-align:center;">最佳年份</td>'
    for s in strategies:
        best_year = yearly_df[s].idxmax()
        best_val = yearly_df[s].max()
        html += f'<td style="text-align:right;background-color:#d4edda;color:#155724;">{best_year}年 ({best_val:.2f}%)</td>'
    html += '</tr>'
    
    html += '<tr style="font-weight:bold;"><td style="text-align:center;">最差年份</td>'
    for s in strategies:
        worst_year = yearly_df[s].idxmin()
        worst_val = yearly_df[s].min()
        html += f'<td style="text-align:right;background-color:#f8d7da;color:#721c24;">{worst_year}年 ({worst_val:.2f}%)</td>'
    html += '</tr>'
    
    html += '<tr style="font-weight:bold;"><td style="text-align:center;">连续亏损最长</td>'
    for s in strategies:
        max_consecutive = 0
        current = 0
        for v in yearly_df[s]:
            if v < 0:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        html += f'<td style="text-align:right;">{max_consecutive}年</td>'
    html += '</tr>'
    
    html += '</tbody></table>'
    
    return html


def generate_html_report(results: Dict, output_dir: str):
    """
    生成HTML综合报告
    
    Args:
        results: 回测结果
        output_dir: 输出目录
    """
    print("生成HTML综合报告...")
    
    comparison_df = results['comparison']
    config = results['config']
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>双均线策略回测对比分析报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{ color: #1f77b4; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; }}
        h2 {{ color: #2ca02c; margin-top: 30px; }}
        h3 {{ color: #ff7f0e; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: right;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
            text-align: center;
        }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .info-box {{
            background-color: #e7f3ff;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            margin: 20px 0;
        }}
        .strategy-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background-color: #fff;
        }}
        .highlight {{ background-color: #fff3cd; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <h1>📊 双均线策略回测对比分析报告</h1>
    
    <div class="info-box">
        <strong>生成时间:</strong> {config['timestamp']}<br>
        <strong>回测品种:</strong> {', '.join(config['data_info']['symbols'])}<br>
        <strong>时间范围:</strong> {config['data_info']['date_range'][0]} 至 {config['data_info']['date_range'][1]}<br>
        <strong>初始资金:</strong> {config['global_config']['initial_cash']:,.0f} 元<br>
        <strong>手续费率:</strong> {config['global_config']['commission']*10000:.0f}%% + 滑点 {config['global_config']['slippage']*10000:.0f}%%
    </div>
    
    <h2>1. 策略配置</h2>
"""
    
    for variant_id, variant_info in config['strategy_variants'].items():
        html_content += f"""
    <div class="strategy-card">
        <h3>{variant_info['name']}</h3>
        <p><strong>描述:</strong> {variant_info['description']}</p>
        <p><strong>参数:</strong></p>
        <ul>
"""
        for param_name, param_value in variant_info['params'].items():
            display_value = param_value if param_value is not None else '未启用'
            html_content += f"            <li>{param_name}: {display_value}</li>\n"
        
        html_content += "        </ul>\n    </div>\n"
    
    html_content += """
    <h2>2. 绩效对比总览</h2>
    <table>
        <thead>
            <tr>
                <th>策略</th>
                <th>总收益率(%)</th>
                <th>年化收益率(%)</th>
                <th>最大回撤(%)</th>
                <th>回撤天数</th>
                <th>胜率(%)</th>
                <th>盈亏比</th>
                <th>交易次数</th>
                <th>持仓天数</th>
                <th>Sharpe</th>
                <th>Sortino</th>
                <th>Calmar</th>
            </tr>
        </thead>
        <tbody>
"""
    
    for strategy in comparison_df.index:
        row = comparison_df.loc[strategy]
        
        # 高亮最优值
        total_return_class = 'highlight' if row['总收益率(%)'] == comparison_df['总收益率(%)'].max() else ''
        sharpe_class = 'highlight' if row['Sharpe比率'] == comparison_df['Sharpe比率'].max() else ''
        drawdown_class = 'highlight' if abs(row['最大回撤(%)']) == abs(comparison_df['最大回撤(%)']).min() else ''
        
        html_content += f"""
            <tr>
                <td style="text-align:left;"><strong>{strategy}</strong></td>
                <td class="{total_return_class}">{row['总收益率(%)']:.2f}</td>
                <td>{row['年化收益率(%)']:.2f}</td>
                <td class="{drawdown_class}">{row['最大回撤(%)']:.2f}</td>
                <td>{int(row['回撤持续天数'])}</td>
                <td>{row['胜率(%)']:.2f}</td>
                <td>{row['盈亏比']:.4f}</td>
                <td>{int(row['交易次数'])}</td>
                <td>{row['平均持仓天数']:.1f}</td>
                <td class="{sharpe_class}">{row['Sharpe比率']:.4f}</td>
                <td>{row['Sortino比率']:.4f}</td>
                <td>{row['Calmar比率']:.4f}</td>
            </tr>
"""
    
    yearly_table_html = generate_yearly_returns_html_table(results)
    
    html_content += f"""
        </tbody>
    </table>
    
    <h2>3. 逐年收益率分析</h2>
    <div class="info-box">
        <p>下表展示各策略每年的收益率表现。<span style="background-color:#d4edda;padding:2px 6px;border-radius:3px;color:#155724;">绿色</span>表示盈利年份，<span style="background-color:#f8d7da;padding:2px 6px;border-radius:3px;color:#721c24;">红色</span>表示亏损年份。</p>
        <p>连续亏损年份是策略失效的重要信号，需特别关注。</p>
    </div>
    {yearly_table_html}
    
    <div class="info-box">
        <p>点击下方链接查看交互式逐年收益率图表：</p>
        <ul>
            <li><a href="yearly_returns.html" target="_blank">3.1 逐年收益率分策略图（含趋势线与失效区间标记）</a></li>
            <li><a href="yearly_returns_comparison.html" target="_blank">3.2 逐年收益率策略对比图</a></li>
        </ul>
    </div>
    
    <h2>4. 可视化分析</h2>
    <div class="info-box">
        <p>点击下方链接查看交互式图表：</p>
        <ul>
            <li><a href="equity_curves.html" target="_blank">4.1 净值曲线对比</a></li>
            <li><a href="drawdown_curves.html" target="_blank">4.2 回撤曲线对比</a></li>
            <li><a href="performance_radar.html" target="_blank">4.3 绩效雷达图</a></li>
            <li><a href="trade_analysis.html" target="_blank">4.4 交易分析</a></li>
            <li><a href="returns_distribution.html" target="_blank">4.5 收益率分布</a></li>
        </ul>
    </div>
    
    <h2>5. 综合评价</h2>
    <div class="info-box">
        <h3>📈 收益率维度</h3>
        <p><strong>最优策略:</strong> {comparison_df['总收益率(%)'].idxmax()}</p>
        <p><strong>总收益率:</strong> {comparison_df['总收益率(%)'].max():.2f}%</p>
        
        <h3>🛡️ 风险控制维度</h3>
        <p><strong>最小回撤策略:</strong> {comparison_df['最大回撤(%)'].abs().idxmin()}</p>
        <p><strong>最大回撤:</strong> {comparison_df['最大回撤(%)'].abs().min():.2f}%</p>
        
        <h3>⚖️ 风险调整收益维度</h3>
        <p><strong>最高Sharpe策略:</strong> {comparison_df['Sharpe比率'].idxmax()}</p>
        <p><strong>Sharpe比率:</strong> {comparison_df['Sharpe比率'].max():.4f}</p>
    </div>
    
    <h2>6. 各策略适用场景分析</h2>
    <div class="strategy-card">
        <h3>原策略</h3>
        <p>适合强趋势市场，能够完整把握趋势行情，但在震荡市中可能频繁止损。</p>
    </div>
    <div class="strategy-card">
        <h3>策略A（5%追踪+20天时间止损）</h3>
        <p>平衡型策略，既有追踪止损保护利润，又有时间止损避免无效持仓，适合大多数市场环境。</p>
    </div>
    <div class="strategy-card">
        <h3>策略B（仅5%追踪止损）</h3>
        <p>趋势跟踪型，让利润奔跑，同时控制单笔亏损，适合中等强度趋势市场。</p>
    </div>
    <div class="strategy-card">
        <h3>策略C（7%追踪+20天时间止损）</h3>
        <p>更宽松的止损，能够承受更大波动，适合高波动品种或强趋势市场。</p>
    </div>
    
    <hr>
    <p style="text-align:center; color:#666;">报告生成时间: {config['timestamp']}</p>
</body>
</html>
"""
    
    with open(f"{output_dir}/report.html", 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML报告已生成: {output_dir}/report.html")


def main(result_dir: str = None):
    """
    主函数
    
    Args:
        result_dir: 回测结果目录，如果为None则自动查找最新目录
    """
    if result_dir is None:
        # 查找最新的回测结果目录
        dirs = [d for d in os.listdir('.') if d.startswith('backtest_results_')]
        if not dirs:
            print("未找到回测结果目录！请先运行 backtest_comparison.py")
            return
        result_dir = sorted(dirs)[-1]
    
    # 加载结果
    results = load_results(result_dir)
    
    # 生成可视化
    plot_equity_curves(results, result_dir)
    plot_drawdown_curves(results, result_dir)
    plot_performance_radar(results, result_dir)
    plot_trade_analysis(results, result_dir)
    plot_returns_distribution(results, result_dir)
    plot_yearly_returns(results, result_dir)
    plot_yearly_returns_comparison(results, result_dir)
    
    # 生成HTML报告
    generate_html_report(results, result_dir)
    
    print("\n" + "=" * 80)
    print("可视化分析完成！")
    print(f"查看报告: {result_dir}/report.html")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import sys
    result_dir = sys.argv[1] if len(sys.argv) > 1 else None
    main(result_dir)
