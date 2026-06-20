"""L1 daily_replay 验证 — 跑 1 周（2024-12-23 ~ 2024-12-31）实盘模拟。

入口：python scripts/run_l1_daily_replay.py
输出：output_backtest_pybroker/l1_daily_replay/
"""
import sys
from pathlib import Path

sys.path.insert(0, "/Users/luojiutian/Documents/backtest")
from loguru import logger

from runner.live import DailyReplaySimulator, run_live_simulation, DEFAULT_SYMBOLS

print("=" * 80)
print("L1 实盘模拟 — 离线日级回放")
print("=" * 80)
print()
print("策略: three_factor (动量+期限+基差动量) — 9 品种等权")
print("窗口: 2024-12-01 ~ 2025-01-15（约 30 个交易日）")
print("模拟方式: T 日收盘信号 → T+1 开盘价撮合 → T+1 收盘价结算")
print("对比基准: e12 实际回测 equity（按回测内部撮合模型）")
print()

# 1. 加载 OHLC 数据
print("[1/3] 加载 9 品种 OHLC 数据...")
from runner.pipeline import Pipeline

# 用 Pipeline 入口（已测试可工作）
pipe = Pipeline("config.yaml").load_data()
data_source = pipe._data
# 拿到完整的 OHLC DataFrame
from core.engine.pybroker_data_source import PyBrokerDataSource
if isinstance(data_source, PyBrokerDataSource):
    ohlc_df = data_source.to_pybroker_df()
else:
    ohlc_df = data_source._df
print(f"  ✓ 加载 {len(ohlc_df)} 行 OHLC 数据")
print(f"  品种: {ohlc_df['symbol'].unique().tolist()}")
print(f"  日期: {ohlc_df['date'].min().date()} ~ {ohlc_df['date'].max().date()}")
print()

# 2. 跑 9 品种 daily_replay
print("[2/3] 跑 9 品种 daily_replay...")
sim = DailyReplaySimulator()
summary = sim.replay_all(DEFAULT_SYMBOLS, "2024-12-01", "2025-01-15", ohlc_df=ohlc_df)
for _, row in summary.iterrows():
    if 'error' in row and row.get('error'):
        print(f"  {row.get('symbol', '?'):12s}: ❌ {row['error']}")
    else:
        print(f"  {row['symbol']:12s}: long={row['long_pct']:5.1f}% | "
              f"short={row['short_pct']:5.1f}% | flat={row['flat_pct']:5.1f}% | "
              f"dir_acc={row['direction_accuracy_pct']:5.1f}% | "
              f"pnl_ratio={row['pnl_ratio']:+6.2f}")
print()

# 3. 汇总
print("[3/3] 汇总 + 保存")
if not summary.empty:
    print(f"\n  === 9 品种 daily_replay 汇总（2024-12-01 ~ 2025-01-15）===")
    print(f"  总真实回测 PnL: {summary['cum_backtest_pnl'].sum():+,.0f}")
    print(f"  总模拟 PnL:     {summary['cum_replay_pnl'].sum():+,.0f}")
    print(f"  平均方向胜率:   {summary['direction_accuracy_pct'].mean():.1f}%")
    print(f"  平均 PnL 比率:  {summary['pnl_ratio'].mean():+.2f}（1.0=完美模拟，<1=低估，>1=高估）")
    print(f"  平均做多占比:   {summary['long_pct'].mean():.1f}%")
    print(f"  平均做空占比:   {summary['short_pct'].mean():.1f}%")
    out_path = sim.save("output_backtest_pybroker/l1_daily_replay")
    print(f"\n  ✓ 详细数据已保存: {out_path}/")
    print(f"    - replay_*.csv (9 个品种逐日明细)")
    print(f"    - summary.csv (汇总)")
    print()
    print("  ⚠️  L1 简化模型限制：")
    print("    - 仅反推 5 天调仓周期内的方向（±1）")
    print("    - 不能精确还原每日仓位大小（受 max_position_pct=0.15 限制）")
    print("    - 用于定性分析（调仓频率、方向胜率），不用于精确 PnL 对比")
    print("    - 精确 PnL 对比需走 L2（TqSdk 模拟盘）或真实 OOS daily backtest")
else:
    print("  ❌ 无有效结果")
