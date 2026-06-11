"""报告生成模块。

拆分自原 core/report_builder.py（2809行），按职责分为7个子模块：
- metrics.py          计算工具函数（回撤/夏普/波动率等）
- data_collector.py   数据收集（从输出目录扫描CSV + 策略数据加载）
- html_template.py    HTML/CSS/JS 模板常量
- generator.py        主逻辑（generate_report + 模板渲染）
- evaluation.py       动态评价 HTML 生成
- sections.py         图表数据构建
- diversification.py  多品种分散化分析 + 调仓决策分析 + OOS对比表

用法:
    from core.report_builder import generate_report
    generate_report(output_dir="output_backtest_pybroker")
"""

from core.report_builder.generator import generate_report
from core.report_builder.metrics import (
    TRADING_DAYS_PER_YEAR,
    RISK_FREE_RATE,
    annualized_return,
    build_kpi_card,
    calmar_ratio,
    compute_daily_returns,
    compute_drawdown,
    compute_sharpe,
    compute_volatility,
    correlation_matrix,
    extract_monthly_returns,
    get_strategy_label,
    read_csv,
    read_equity_csv,
    rolling_max_drawdown,
    rolling_sharpe,
    safe_float,
)
from core.report_builder.data_collector import (
    collect_from_directory,
    collect_from_validation,
    load_strategies_data,
)
from core.report_builder.diversification import (
    build_diversification_html,
    build_rebalance_html,
    build_strategy_oos_rows,
)
from core.report_builder.sections import build_chart_data
from core.report_builder.evaluation import build_dynamic_evaluation_html

__all__ = [
    # 主入口
    "generate_report",
    # 数据收集
    "collect_from_directory",
    "collect_from_validation",
    "load_strategies_data",
    # 图表数据
    "build_chart_data",
    # 分散化/调仓/OOS
    "build_diversification_html",
    "build_rebalance_html",
    "build_strategy_oos_rows",
    # 动态评价
    "build_dynamic_evaluation_html",
    # 计算工具
    "safe_float",
    "annualized_return",
    "build_kpi_card",
    "calmar_ratio",
    "compute_drawdown",
    "compute_daily_returns",
    "compute_volatility",
    "compute_sharpe",
    "correlation_matrix",
    "rolling_sharpe",
    "rolling_max_drawdown",
    "extract_monthly_returns",
    "get_strategy_label",
    "read_csv",
    "read_equity_csv",
    # 常量
    "TRADING_DAYS_PER_YEAR",
    "RISK_FREE_RATE",
]
