"""回测运行器 — 结果数据类型。

定义 PyBrokerResult 和 WalkforwardResult 数据类。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PyBrokerResult:
    """PyBroker 回测结果封装。"""

    metrics: Dict[str, float]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    switch_log: pd.DataFrame
    bootstrap_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class WalkforwardResult:
    """Walkforward 向前滚动分析结果。"""

    windows: List[Dict[str, Any]]
    overall_metrics: Dict[str, float]
    equity_curves: List[pd.DataFrame]

    def plot_equity_curves(self):
        """绘制各窗口净值曲线（需 plotly）。"""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            for i, eq in enumerate(self.equity_curves):
                fig.add_trace(
                    go.Scatter(
                        x=eq["date"],
                        y=eq["equity"],
                        mode="lines",
                        name=f"Window {i + 1}",
                    )
                )
            fig.update_layout(
                title="Walkforward Equity Curves",
                xaxis_title="Date",
                yaxis_title="Equity",
            )
            fig.show()
        except ImportError:
            logger.warning("plotly 未安装，无法绘图。请运行: pip install plotly")
            for i, eq in enumerate(self.equity_curves):
                logger.debug(
                    "Window %d: final equity = %.2f",
                    i + 1,
                    eq["equity"].iloc[-1],
                )
