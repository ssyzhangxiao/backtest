"""
绩效指标计算模块。

从 PyBroker 回测结果中提取并计算各种绩效指标，
供前端展示和分析使用。
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List


class MetricsCalculator:
    """
    绩效指标计算器。

    从 PyBroker 的 TestResult 中提取指标，并计算额外的衍生指标。
    """

    @staticmethod
    def extract_from_pybroker_result(result) -> Dict:
        """
        从 PyBroker 回测结果中提取指标。

        Args:
            result: PyBroker 的 TestResult 对象

        Returns:
            指标字典
        """
        metrics = {}

        # 方式1: 尝试直接访问属性
        attr_names = [
            'total_return_pct', 'annual_return_pct', 'max_drawdown_pct',
            'sharpe_ratio', 'sharpe', 'sortino_ratio', 'sortino', 'calmar_ratio',
            'win_rate', 'profit_factor', 'trade_count'
        ]
        for attr in attr_names:
            if hasattr(result, attr):
                val = getattr(result, attr)
                metrics[attr] = val

        # 方式2: 尝试 metrics_df (如果存在)
        if hasattr(result, 'metrics_df') and result.metrics_df is not None:
            df = result.metrics_df
            try:
                if 'name' in df.columns and 'value' in df.columns:
                    # 格式: name / value 两列
                    metrics.update(dict(zip(df['name'], df['value'])))
                elif len(df.columns) > 0:
                    # 格式: 指标名列，第一列是指标
                    for col in df.columns:
                        metrics[col] = df.iloc[0][col] if len(df) > 0 else 0
            except Exception:
                pass

        # 方式3: 如果 result 有 __dict__，尝试从那里提取
        if hasattr(result, '__dict__'):
            for k, v in result.__dict__.items():
                if not k.startswith('_') and k not in metrics:
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        metrics[k] = v

        return metrics

    @staticmethod
    def calculate_additional_metrics(portfolio_df: pd.DataFrame,
                                      trades_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        计算额外的绩效指标。

        Args:
            portfolio_df: PyBroker portfolio DataFrame
            trades_df: PyBroker trades DataFrame

        Returns:
            额外指标字典
        """
        metrics = {}

        if portfolio_df is not None and not portfolio_df.empty:
            # 尝试多种可能的净值列名
            equity_col = None
            for col in ['equity', 'market_value', 'port_value', 'total_equity']:
                if col in portfolio_df.columns:
                    equity_col = col
                    break
            
            if equity_col:
                equity = portfolio_df[equity_col]
                returns = equity.pct_change().dropna()

                if len(returns) > 0:
                    annual_factor = 252
                    metrics['annual_return_pct'] = (
                        (1 + returns).prod() ** (annual_factor / len(returns)) - 1
                    ) * 100

                    metrics['annual_volatility_pct'] = returns.std() * np.sqrt(annual_factor) * 100

                    risk_free = 0.02 / annual_factor
                    excess = returns - risk_free
                    if returns.std() > 0:
                        metrics['sharpe_ratio'] = (
                            excess.mean() / returns.std() * np.sqrt(annual_factor)
                        )
                    else:
                        metrics['sharpe_ratio'] = 0.0

                    downside = returns[returns < 0]
                    if len(downside) > 0 and downside.std() > 0:
                        metrics['sortino_ratio'] = (
                            excess.mean() / downside.std() * np.sqrt(annual_factor)
                        )
                    else:
                        metrics['sortino_ratio'] = 0.0

                    peak = equity.cummax()
                    drawdown = (equity - peak) / peak
                    metrics['max_drawdown_pct'] = drawdown.min() * 100

                    in_drawdown = drawdown < 0
                    if in_drawdown.any():
                        dd_groups = (~in_drawdown).cumsum()
                        dd_durations = in_drawdown.groupby(dd_groups).sum()
                        metrics['max_drawdown_duration_days'] = int(dd_durations.max())
                    else:
                        metrics['max_drawdown_duration_days'] = 0

                    metrics['calmar_ratio'] = (
                        metrics['annual_return_pct'] / abs(metrics['max_drawdown_pct'])
                        if metrics['max_drawdown_pct'] != 0 else 0.0
                    )

                    positive_days = (returns > 0).sum()
                    total_days = len(returns)
                    metrics['daily_win_rate'] = positive_days / total_days * 100 if total_days > 0 else 0

        if trades_df is not None and not trades_df.empty:
            if 'pnl' in trades_df.columns:
                winning = trades_df[trades_df['pnl'] > 0]
                losing = trades_df[trades_df['pnl'] < 0]

                metrics['trade_count'] = len(trades_df)
                metrics['winning_trades'] = len(winning)
                metrics['losing_trades'] = len(losing)
                metrics['win_rate'] = len(winning) / len(trades_df) * 100 if len(trades_df) > 0 else 0

                if len(winning) > 0:
                    metrics['avg_win'] = winning['pnl'].mean()
                    metrics['avg_win_pct'] = winning['return_pct'].mean() if 'return_pct' in winning.columns else 0
                else:
                    metrics['avg_win'] = 0
                    metrics['avg_win_pct'] = 0

                if len(losing) > 0:
                    metrics['avg_loss'] = losing['pnl'].mean()
                    metrics['avg_loss_pct'] = losing['return_pct'].mean() if 'return_pct' in losing.columns else 0
                else:
                    metrics['avg_loss'] = 0
                    metrics['avg_loss_pct'] = 0

                total_profit = winning['pnl'].sum() if len(winning) > 0 else 0
                total_loss = abs(losing['pnl'].sum()) if len(losing) > 0 else 1
                metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')

                metrics['expectancy'] = trades_df['pnl'].mean() if len(trades_df) > 0 else 0

        return metrics

    @staticmethod
    def _get_metric(metrics: Dict, *keys) -> float:
        for k in keys:
            if k in metrics:
                return float(metrics[k])
        return 0.0

    @staticmethod
    def format_metrics_card(metrics: Dict) -> List[Dict]:
        """
        将指标格式化为前端卡片数据。

        PyBroker 返回的指标键名可能为 'sharpe', 'total_return_pct' 等，
        而 calculate_additional_metrics 使用 'sharpe_ratio', 'sortino_ratio' 等。
        此方法同时检查多种键名变体。

        Args:
            metrics: 指标字典

        Returns:
            卡片数据列表 [{label, value, format}]
        """
        _g = MetricsCalculator._get_metric
        card_items = [
            {"label": "总收益率", "value": _g(metrics, 'total_return_pct'), "format": "pct"},
            {"label": "年化收益率", "value": _g(metrics, 'annual_return_pct'), "format": "pct"},
            {"label": "最大回撤", "value": _g(metrics, 'max_drawdown_pct'), "format": "pct"},
            {"label": "Sharpe比率", "value": _g(metrics, 'sharpe_ratio', 'sharpe'), "format": "ratio"},
            {"label": "Sortino比率", "value": _g(metrics, 'sortino_ratio', 'sortino'), "format": "ratio"},
            {"label": "Calmar比率", "value": _g(metrics, 'calmar_ratio'), "format": "ratio"},
            {"label": "胜率", "value": _g(metrics, 'win_rate'), "format": "pct"},
            {"label": "盈亏比", "value": _g(metrics, 'profit_factor'), "format": "ratio"},
            {"label": "交易次数", "value": _g(metrics, 'trade_count'), "format": "int"},
            {"label": "日均胜率", "value": _g(metrics, 'daily_win_rate'), "format": "pct"},
        ]
        return card_items

    @staticmethod
    def format_value(value, format_type: str) -> str:
        """
        格式化指标值。

        Args:
            value: 原始值
            format_type: 格式类型 ('pct', 'ratio', 'int', 'money')

        Returns:
            格式化后的字符串
        """
        try:
            if format_type == 'pct':
                return f"{float(value):.2f}%"
            elif format_type == 'ratio':
                return f"{float(value):.4f}"
            elif format_type == 'int':
                return f"{int(value)}"
            elif format_type == 'money':
                return f"{float(value):,.2f}"
            else:
                return str(value)
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def compute_rollover_stats(trades_df: pd.DataFrame,
                                rollover_dates: Optional[pd.DataFrame] = None) -> Dict:
        """
        计算展期相关统计。

        Args:
            trades_df: 交易记录
            rollover_dates: 展期日期记录

        Returns:
            展期统计字典
        """
        stats = {
            "rollover_count": 0,
            "total_rollover_cost": 0,
            "avg_rollover_cost": 0,
        }

        if rollover_dates is not None and not rollover_dates.empty:
            stats['rollover_count'] = len(rollover_dates)
            if 'rollover_cost' in rollover_dates.columns:
                stats['total_rollover_cost'] = rollover_dates['rollover_cost'].sum()
                stats['avg_rollover_cost'] = rollover_dates['rollover_cost'].mean()

        return stats
