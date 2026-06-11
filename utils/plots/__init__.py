"""图表绘制模块。

使用 Plotly 绘制回测结果可视化图表，供 Streamlit 前端调用。
覆盖六大模块：数据概览、策略绩效、风险归因、交易执行、参数优化、市场状态。
所有图表返回 plotly.graph_objects.Figure 对象，可直接用于 st.plotly_chart()。
"""

from utils.plots._base import COLORS, REGIME_COLORS, BasePlotMixin
from utils.plots._price_volume import PriceVolumeMixin
from utils.plots._equity import EquityMixin
from utils.plots._risk import RiskMixin
from utils.plots._trading import TradingMixin
from utils.plots._optimization import OptimizationMixin
from utils.plots._regime import RegimeMixin


class PlotManager(
    PriceVolumeMixin,
    EquityMixin,
    RiskMixin,
    TradingMixin,
    OptimizationMixin,
    RegimeMixin,
):
    """统一图表管理器，组合六大功能 Mixin。"""

    pass


__all__ = [
    "PlotManager",
    "COLORS",
    "REGIME_COLORS",
    "BasePlotMixin",
    "PriceVolumeMixin",
    "EquityMixin",
    "RiskMixin",
    "TradingMixin",
    "OptimizationMixin",
    "RegimeMixin",
]
