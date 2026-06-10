"""
绩效评估与优化系统。

提供全面的策略绩效评估、定期报告生成、参数自适应优化和预警机制。

核心功能:
  - 多维度绩效指标计算
  - 定期（日/周/月）绩效报告
  - 参数自适应优化
  - 策略预警机制
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

import pandas as pd
import numpy as np


class AlertLevel(Enum):
    """预警级别。"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """绩效预警。"""
    timestamp: str
    level: AlertLevel
    strategy_name: str
    metric: str
    current_value: float
    threshold: float
    message: str


@dataclass
class PerformanceConfig:
    """绩效评估配置。"""
    # 预警阈值
    sharpe_warning: float = 0.0
    sharpe_critical: float = -0.5
    max_drawdown_warning: float = -15.0
    max_drawdown_critical: float = -25.0
    win_rate_warning: float = 40.0
    win_rate_critical: float = 30.0
    daily_loss_warning: float = -0.02
    daily_loss_critical: float = -0.05

    # 自适应优化
    adaptation_window: int = 60  # 自适应回看窗口（天）
    adaptation_min_trades: int = 10  # 最少交易次数
    adaptation_interval: int = 30  # 优化间隔（天）

    # 报告
    report_dir: str = "./output_performance"


class PerformanceEvaluator:
    """
    绩效评估器。

    从净值曲线和交易记录计算全面的绩效指标。
    """

    @staticmethod
    def compute_metrics(equity: pd.Series, trades: Optional[pd.DataFrame] = None,
                        risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        计算全面的绩效指标。

        Args:
            equity: 净值序列
            trades: 交易记录（可选）
            risk_free_rate: 无风险利率（年化）

        Returns:
            指标字典
        """
        if len(equity) < 2:
            return {}

        metrics = {}
        returns = equity.pct_change().dropna()
        annual_factor = 252

        # 绝对收益
        metrics["total_return_pct"] = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        days = len(equity)
        metrics["annual_return_pct"] = ((equity.iloc[-1] / equity.iloc[0]) ** (annual_factor / days) - 1) * 100 if days > 0 else 0

        # 波动率
        metrics["annual_volatility_pct"] = returns.std() * np.sqrt(annual_factor) * 100 if len(returns) > 0 else 0

        # 风险调整后收益
        excess = returns - risk_free_rate / annual_factor
        # P0 修复（2026-06-10）：防止 daily std 接近 0 时 Sharpe 爆炸到百万级
        # 原因：equity 水平线（5 个子策略全 0 信号）时 std ≈ 1e-10，mean/std 爆炸
        # 阈值：日波动率 < 1e-6 视为"无交易"，Sharpe 记 0
        MIN_DAILY_STD = 1e-6
        if returns.std() > MIN_DAILY_STD:
            metrics["sharpe"] = excess.mean() / returns.std() * np.sqrt(annual_factor)
        else:
            metrics["sharpe"] = 0.0

        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > MIN_DAILY_STD:
            metrics["sortino"] = excess.mean() / downside.std() * np.sqrt(annual_factor)
        else:
            metrics["sortino"] = 0.0

        # 最大回撤
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        metrics["max_drawdown_pct"] = drawdown.min() * 100

        # Calmar
        metrics["calmar"] = metrics["annual_return_pct"] / abs(metrics["max_drawdown_pct"]) if metrics["max_drawdown_pct"] != 0 else 0

        # 回撤持续期
        in_dd = drawdown < 0
        if in_dd.any():
            dd_groups = (~in_dd).cumsum()
            dd_durations = in_dd.groupby(dd_groups).sum()
            metrics["max_drawdown_duration"] = int(dd_durations.max())
        else:
            metrics["max_drawdown_duration"] = 0

        # 日胜率
        metrics["daily_win_rate"] = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0

        # 交易相关指标
        if trades is not None and not trades.empty and "pnl" in trades.columns:
            winning = trades[trades["pnl"] > 0]
            losing = trades[trades["pnl"] < 0]
            metrics["trade_count"] = len(trades)
            metrics["win_rate"] = len(winning) / len(trades) * 100 if len(trades) > 0 else 0

            total_profit = winning["pnl"].sum() if len(winning) > 0 else 0
            total_loss = abs(losing["pnl"].sum()) if len(losing) > 0 else 1
            metrics["profit_factor"] = total_profit / total_loss if total_loss > 0 else float("inf")

            avg_win = winning["pnl"].mean() if len(winning) > 0 else 0
            avg_loss = abs(losing["pnl"].mean()) if len(losing) > 0 else 1
            metrics["profit_loss_ratio"] = avg_win / avg_loss if avg_loss > 0 else float("inf")

            metrics["expectancy"] = trades["pnl"].mean()
        else:
            metrics["trade_count"] = 0
            metrics["win_rate"] = 0
            metrics["profit_factor"] = 0
            metrics["profit_loss_ratio"] = 0
            metrics["expectancy"] = 0

        return metrics

    @staticmethod
    def compute_rolling_metrics(equity: pd.Series, window: int = 60) -> pd.DataFrame:
        """
        计算滚动窗口绩效指标。

        Args:
            equity: 净值序列
            window: 滚动窗口（天）

        Returns:
            滚动指标DataFrame
        """
        returns = equity.pct_change().dropna()
        result = pd.DataFrame(index=equity.index)

        result["rolling_return"] = equity.pct_change(window)
        result["rolling_volatility"] = returns.rolling(window).std() * np.sqrt(252)
        result["rolling_sharpe"] = (
            returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        )

        # 滚动最大回撤
        rolling_dd = []
        for i in range(len(equity)):
            if i < window:
                rolling_dd.append(0)
            else:
                window_eq = equity.iloc[max(0, i - window):i + 1]
                peak = window_eq.cummax()
                dd = ((window_eq - peak) / peak).min()
                rolling_dd.append(dd)
        result["rolling_max_drawdown"] = rolling_dd

        return result


class PerformanceMonitor:
    """
    绩效监控与预警系统。

    定期评估策略表现，当指标低于阈值时发出预警。
    """

    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self._alerts: List[PerformanceAlert] = []
        self._last_evaluation: Dict[str, str] = {}

    def evaluate(self, strategy_name: str, metrics: Dict[str, float],
                 current_date: str) -> List[PerformanceAlert]:
        """
        评估策略绩效并生成预警。

        Args:
            strategy_name: 策略名称
            metrics: 当前绩效指标
            current_date: 当前日期

        Returns:
            预警列表
        """
        alerts = []
        cfg = self.config

        # Sharpe预警
        sharpe = metrics.get("sharpe", 0)
        if sharpe < cfg.sharpe_critical:
            alerts.append(PerformanceAlert(
                timestamp=current_date, level=AlertLevel.CRITICAL,
                strategy_name=strategy_name, metric="sharpe",
                current_value=sharpe, threshold=cfg.sharpe_critical,
                message=f"策略{strategy_name} Sharpe={sharpe:.3f} 低于临界值{cfg.sharpe_critical}"
            ))
        elif sharpe < cfg.sharpe_warning:
            alerts.append(PerformanceAlert(
                timestamp=current_date, level=AlertLevel.WARNING,
                strategy_name=strategy_name, metric="sharpe",
                current_value=sharpe, threshold=cfg.sharpe_warning,
                message=f"策略{strategy_name} Sharpe={sharpe:.3f} 低于警告值{cfg.sharpe_warning}"
            ))

        # 最大回撤预警
        mdd = metrics.get("max_drawdown_pct", 0)
        if mdd < cfg.max_drawdown_critical:
            alerts.append(PerformanceAlert(
                timestamp=current_date, level=AlertLevel.CRITICAL,
                strategy_name=strategy_name, metric="max_drawdown_pct",
                current_value=mdd, threshold=cfg.max_drawdown_critical,
                message=f"策略{strategy_name} 最大回撤={mdd:.2f}% 超过临界值{cfg.max_drawdown_critical}%"
            ))
        elif mdd < cfg.max_drawdown_warning:
            alerts.append(PerformanceAlert(
                timestamp=current_date, level=AlertLevel.WARNING,
                strategy_name=strategy_name, metric="max_drawdown_pct",
                current_value=mdd, threshold=cfg.max_drawdown_warning,
                message=f"策略{strategy_name} 最大回撤={mdd:.2f}% 超过警告值{cfg.max_drawdown_warning}%"
            ))

        # 胜率预警
        wr = metrics.get("win_rate", 0)
        if wr < cfg.win_rate_critical:
            alerts.append(PerformanceAlert(
                timestamp=current_date, level=AlertLevel.CRITICAL,
                strategy_name=strategy_name, metric="win_rate",
                current_value=wr, threshold=cfg.win_rate_critical,
                message=f"策略{strategy_name} 胜率={wr:.1f}% 低于临界值{cfg.win_rate_critical}%"
            ))
        elif wr < cfg.win_rate_warning:
            alerts.append(PerformanceAlert(
                timestamp=current_date, level=AlertLevel.WARNING,
                strategy_name=strategy_name, metric="win_rate",
                current_value=wr, threshold=cfg.win_rate_warning,
                message=f"策略{strategy_name} 胜率={wr:.1f}% 低于警告值{cfg.win_rate_warning}%"
            ))

        self._alerts.extend(alerts)
        self._last_evaluation[strategy_name] = current_date
        return alerts

    def get_alerts(self, level: Optional[AlertLevel] = None) -> List[PerformanceAlert]:
        """获取预警列表。"""
        if level is None:
            return self._alerts
        return [a for a in self._alerts if a.level == level]

    def generate_report(self, strategy_name: str, metrics: Dict[str, float],
                        rolling_metrics: Optional[pd.DataFrame] = None) -> str:
        """
        生成策略绩效报告（Markdown格式）。
        """
        lines = [
            f"# 策略绩效报告: {strategy_name}",
            f"",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## 核心指标",
            f"",
            f"| 指标 | 值 |",
            f"|------|-----|",
        ]

        key_metrics = [
            ("总收益率(%)", "total_return_pct"),
            ("年化收益率(%)", "annual_return_pct"),
            ("最大回撤(%)", "max_drawdown_pct"),
            ("年化波动率(%)", "annual_volatility_pct"),
            ("Sharpe比率", "sharpe"),
            ("Sortino比率", "sortino"),
            ("Calmar比率", "calmar"),
            ("胜率(%)", "win_rate"),
            ("盈亏比", "profit_factor"),
            ("交易次数", "trade_count"),
        ]

        for label, key in key_metrics:
            val = metrics.get(key, "N/A")
            if isinstance(val, float):
                if key in ("sharpe", "sortino", "calmar"):
                    lines.append(f"| {label} | {val:.4f} |")
                elif key in ("trade_count",):
                    lines.append(f"| {label} | {int(val)} |")
                else:
                    lines.append(f"| {label} | {val:.2f} |")
            else:
                lines.append(f"| {label} | {val} |")

        # 预警信息
        recent_alerts = [a for a in self._alerts if a.strategy_name == strategy_name]
        if recent_alerts:
            lines.append(f"")
            lines.append(f"## 预警信息")
            lines.append(f"")
            for alert in recent_alerts[-5:]:
                lines.append(f"- [{alert.level.value}] {alert.message}")

        # 优化建议
        lines.append(f"")
        lines.append(f"## 优化建议")
        lines.append(f"")

        sharpe = metrics.get("sharpe", 0)
        mdd = metrics.get("max_drawdown_pct", 0)
        wr = metrics.get("win_rate", 0)

        if sharpe < 0:
            lines.append("- Sharpe比率为负，建议检查策略逻辑或暂停使用")
        if mdd < -20:
            lines.append("- 最大回撤过大，建议加强风控或减小仓位")
        if wr < 40:
            lines.append("- 胜率偏低，建议调整入场条件或增加过滤指标")
        if sharpe > 0 and mdd > -10 and wr > 50:
            lines.append("- 策略表现良好，可考虑适度增加仓位")

        return "\n".join(lines)
