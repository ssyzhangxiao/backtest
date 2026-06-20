"""实盘模拟模块 — L1/L3 离线日级回放入口。

公开接口：
  层级             类/函数                    用途
  ─────────────────────────────────────────────────────────
  L1 (定性)        DailyReplaySimulator        离线日级回放（反推 position）
  L1 (定性)        run_live_simulation          批量跑 9 品种 daily_replay
  L3 (精确)        TradingCalendar              交易日历（PyBrokerDataSource + 节假日）
  L3 (精确)        check_data_completeness      数据完整性检查
  L3 (精确)        generate_dashboard           生成 HTML 看板（净值曲线 + 持仓 + 绩效）
  L3 (精确)        run_daily_sim (scripts/)     每日 OOS 回测入口脚本

规则 17：不重复造轮子 — 绩效委托 MetricsCalculator，日历复用 utils/date_utils.py。
规则 18：Pipeline.daily_sim() 可链式调用 L3 流程。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .calendar import TradingCalendar, get_trading_calendar
from .daily_replay import (
    DEFAULT_SYMBOLS,
    analyze_replay_quality,
    run_daily_replay,
)
from .data_checker import (
    DataCompletenessReport,
    check_data_completeness,
    check_data_completeness_summary,
    check_field_completeness,
)
from .dashboard import (
    compute_performance_summary,
    generate_dashboard,
    plot_equity_curve,
    plot_position_heatmap,
)

__all__ = [
    # L1: 简化回放
    "DailyReplaySimulator",
    "run_live_simulation",
    "run_daily_replay",
    "analyze_replay_quality",
    # L3: 交易日历
    "TradingCalendar",
    "get_trading_calendar",
    # L3: 数据检查
    "check_data_completeness",
    "check_data_completeness_summary",
    "check_field_completeness",
    "DataCompletenessReport",
    # L3: 看板
    "generate_dashboard",
    "compute_performance_summary",
    "plot_equity_curve",
    "plot_position_heatmap",
    # 共用
    "DEFAULT_SYMBOLS",
]


class DailyReplaySimulator:
    """L1 离线日级回放模拟器（按次日开盘价撮合模拟）。"""

    def __init__(
        self,
        output_dir: str = "output_backtest_pybroker",
        initial_capital: float = 1_000_000.0,
    ):
        self.output_dir = output_dir
        self.initial_capital = initial_capital
        self._results: Dict[str, pd.DataFrame] = {}
        self._summary: Optional[pd.DataFrame] = None

    def replay(self, symbol: str, start: str, end: str, ohlc_df=None) -> pd.DataFrame:
        """单品种 daily_replay 模拟。"""
        df = run_daily_replay(
            symbol, start, end,
            output_dir=self.output_dir,
            initial_capital=self.initial_capital,
            ohlc_df=ohlc_df,
        )
        self._results[symbol] = df
        return df

    def replay_all(
        self,
        symbols: Optional[List[str]] = None,
        start: str = "2024-12-01",
        end: str = "2025-01-15",
        ohlc_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """批量跑多品种 daily_replay，返回定性分析 summary。"""
        symbols = symbols or DEFAULT_SYMBOLS
        all_summary = []
        for sym in symbols:
            try:
                sub_ohlc = None
                if ohlc_df is not None and not ohlc_df.empty:
                    sub_ohlc = ohlc_df[ohlc_df['symbol'] == sym][['date', 'open', 'close']].copy()
                df = self.replay(sym, start, end, ohlc_df=sub_ohlc)
                if not df.empty:
                    q = analyze_replay_quality(df)
                    all_summary.append(q)
                else:
                    all_summary.append({"symbol": sym, "error": "empty df"})
            except Exception as e:
                all_summary.append({"symbol": sym, "error": str(e)})
        self._summary = pd.DataFrame(all_summary)
        return self._summary

    @property
    def summary(self) -> pd.DataFrame:
        return self._summary if self._summary is not None else pd.DataFrame()

    def save(self, out_dir: str = "output_backtest_pybroker/l1_daily_replay") -> Path:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        for sym, df in self._results.items():
            df.to_csv(p / f"replay_{sym.replace('.', '_')}.csv", index=False)
        if not self.summary.empty:
            self.summary.to_csv(p / "summary.csv", index=False)
        return p


def run_live_simulation(
    symbols: Optional[List[str]] = None,
    start: str = "2024-12-01",
    end: str = "2025-01-15",
    save: bool = True,
) -> Dict[str, Any]:
    """批量跑 9 品种 daily_replay 模拟（L1 实盘模拟入口）。"""
    sim = DailyReplaySimulator()
    summary = sim.replay_all(symbols, start, end)
    out_path = None
    if save:
        out_path = sim.save()
    return {
        "summary": summary,
        "results": sim._results,
        "output_dir": str(out_path) if save else None,
    }