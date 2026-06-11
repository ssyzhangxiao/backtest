"""图表绘制模块 — 常量与辅助方法。"""

import plotly.graph_objects as go
import pandas as pd
from typing import List, Optional

COLORS = {
    "blue": "#1f77b4", "orange": "#ff7f0e", "green": "#2ca02c",
    "red": "#d62728", "purple": "#9467bd", "brown": "#8c564b",
    "pink": "#e377c2", "gray": "#7f7f7f", "olive": "#bcbd22",
    "cyan": "#17becf",
}

REGIME_COLORS = {
    "trend": "rgba(76, 175, 80, 0.12)",
    "range": "rgba(255, 193, 7, 0.12)",
}


class BasePlotMixin:
    """PlotManager 辅助方法 Mixin。"""

    @staticmethod
    def _ensure_date_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """确保 DataFrame 有日期列。"""
        if date_col not in df.columns:
            if "index" in df.columns:
                df = df.rename(columns={"index": date_col})
            else:
                df = df.reset_index()
                if "index" in df.columns:
                    df = df.rename(columns={"index": date_col})
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        return df

    @staticmethod
    def _get_equity_col(df: pd.DataFrame, equity_col: str = "equity") -> str:
        """自动检测净值列名。"""
        if equity_col in df.columns:
            return equity_col
        for candidate in ("market_value", "portfolio_value", "close"):
            if candidate in df.columns:
                return candidate
        return equity_col

    @classmethod
    def _get_color(cls, idx: int) -> str:
        """按索引获取颜色。"""
        color_list = list(COLORS.values())
        return color_list[idx % len(color_list)]

    @staticmethod
    def _empty_fig(title: str = "", height: int = 400, missing_cols: Optional[List[str]] = None) -> go.Figure:
        """生成空图表（数据缺失时）。"""
        msg = "暂无数据"
        if missing_cols:
            msg = f"缺少必要列: {', '.join(missing_cols)}"
        fig = go.Figure()
        fig.add_annotation(
            text=msg, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray"),
        )
        fig.update_layout(title=title, height=height)
        return fig

    @staticmethod
    def _check_df(df, title: str, missing_cols: Optional[List[str]] = None) -> Optional[go.Figure]:
        """检查 DataFrame 是否有效，无效则返回空图表。"""
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return BasePlotMixin._empty_fig(title=title, missing_cols=missing_cols)
        if missing_cols and isinstance(df, pd.DataFrame):
            absent = [c for c in missing_cols if c not in df.columns]
            if absent:
                return BasePlotMixin._empty_fig(title=title, missing_cols=absent)
        return None
