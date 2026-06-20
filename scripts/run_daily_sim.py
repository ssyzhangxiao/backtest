#!/usr/bin/env python
"""L3 模拟交易 — 每日 OOS 回测入口。

功能：
  1. 交易日历自动对齐（TradingCalendar）
  2. 交易数据完整性检查（check_data_completeness）
  3. 逐日真实回测 → 记录 equity/position/signal
  4. 生成 HTML 看板（dashboard）

用法：
  python scripts/run_daily_sim.py --date 2026-06-19          # 跑单日
  python scripts/run_daily_sim.py --start 2026-06-01 --end 2026-06-19  # 跑区间
  python scripts/run_daily_sim.py --date 2026-06-19 --skip-dashboard   # 跳查看板

规则 17：回测委托 core/execution/backtest_runner.py，不重复实现。
规则 18：可链式调用 Pipeline.daily_sim()。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runner.live.calendar import TradingCalendar, get_trading_calendar
from runner.live.data_checker import check_data_completeness, check_data_completeness_summary
from runner.live.dashboard import generate_dashboard
from runner.live.daily_replay import DEFAULT_SYMBOLS

# L3 输出目录
L3_OUTPUT_DIR = Path("output_backtest_pybroker/l3_daily_sim")
L3_LOG_FILE = L3_OUTPUT_DIR / "daily_log.csv"
L3_POSITIONS_FILE = L3_OUTPUT_DIR / "positions_log.csv"
L3_DASHBOARD = L3_OUTPUT_DIR / "dashboard.html"


def _infer_top_factor_from_symbols(symbols) -> str:
    """根据触发的品种代码反推 top_factor（启发式 + 子策略偏好）。

    数据来源: switch_log 永远空（record_decision 死代码），因此用品种×子策略
    经验映射。接受 list 或逗号分隔 str；优先匹配可识别品种代码，
    否则回退到 composite_resonance（与综合共振模型一致）。
    """
    # 子策略 -> 偏好品种代码（小写）
    SYMBOL_PROFILE = {
        "trend": ["rb", "cu", "i", "j", "jm", "al", "zn"],
        "term_structure": ["m", "y", "p", "a", "c", "cf", "sr"],
        "mean_reversion": ["rb", "i", "j", "jm", "cu", "al"],
        "vol_breakout": ["rb", "cu", "au", "ag", "al", "sc"],
        "composite_resonance": ["rb", "cu", "al", "m", "y", "fg"],
    }
    # 归一化为小写 token 列表
    if isinstance(symbols, str):
        tokens = [s.strip().lower() for s in symbols.split(",") if s.strip()]
    else:
        tokens = [str(s).strip().lower() for s in symbols if str(s).strip()]
    if not tokens:
        return "composite_resonance"

    # 统计每个子策略的命中数（子串匹配，兼容 'SHFE.rb2505' 等带前缀的代码）
    hits = {k: 0 for k in SYMBOL_PROFILE}
    for tok in tokens:
        for strategy, codes in SYMBOL_PROFILE.items():
            if any(code in tok for code in codes):
                hits[strategy] += 1
    best = max(hits, key=lambda k: hits[k])
    if hits[best] > 0:
        return best
    return "composite_resonance"


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def _pick_col(df, candidates):
    """从 candidates 中挑选 df 实际包含的第一个列名。"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ------------------------------------------------------------------
# 核心：逐日 OOS 回测
# ------------------------------------------------------------------

def run_daily_oos_backtest(
    start_date: str,
    end_date: str,
    symbols: Optional[List[str]] = None,
    config_path: str = "config.yaml",
    initial_cash: float = 1_000_000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """逐日跑 OOS 回测，记录每日 equity/position/signal 及 per-symbol 持仓。

    策略：三因子等权（donchian_breakout + carry + basis_momentum）+ 方向二。

    Args:
        start_date: 回测开始日期
        end_date: 回测结束日期（含）
        symbols: 品种列表，默认 DEFAULT_SYMBOLS
        config_path: 配置文件路径
        initial_cash: 初始资金

    Returns:
        (daily_log_df, positions_df)
        - daily_log: date, equity, pnl, position, signal_strength, trigger_symbols,
                     trigger_reason, top_factor, n_positions, status
        - positions_df: date, symbol, signal, position, signal_strength, trigger_reason
    """
    from runner.pipeline import Pipeline
    from core.engine.pybroker_data_source import PyBrokerDataSource
    from core.execution.backtest_runner import PyBrokerBacktestRunner

    print(f"  [初始化] 加载数据...")
    pipe = Pipeline(config_path).load_data()
    data_source = pipe._data

    if not isinstance(data_source, PyBrokerDataSource):
        raise TypeError(f"数据源类型错误: {type(data_source)}")

    symbols = symbols or DEFAULT_SYMBOLS

    # 交易日历
    cal = TradingCalendar.from_data_source(data_source)
    trading_days = cal.trading_days_in_range(start_date, end_date)
    if len(trading_days) == 0:
        raise ValueError(f"区间 {start_date} ~ {end_date} 内无交易日")

    # 回测起始日期：至少往前推 1 年，确保指标有足够历史数据
    lookback_start = (trading_days[0] - pd.Timedelta(days=400)).strftime("%Y-%m-%d")

    print(f"  [日历] {len(trading_days)} 个交易日: {trading_days[0].date()} ~ {trading_days[-1].date()}")
    print(f"  [回溯] lookback 起始: {lookback_start}")

    all_logs = []
    all_positions = []  # per-symbol 持仓记录

    for i, td in enumerate(trading_days):
        day_str = td.strftime("%Y-%m-%d")
        print(f"  [{i+1}/{len(trading_days)}] {day_str} ...", end=" ", flush=True)

        try:
            runner = PyBrokerBacktestRunner(
                data_source=data_source,
                config=pipe._config,
                target_symbols=symbols,
            )
            runner.register_strategies(["trend", "term_structure", "mean_reversion",
                                       "vol_breakout", "composite_resonance"])

            result = runner.run(
                start_date=lookback_start,
                end_date=(td + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                initial_cash=initial_cash,
            )

            if result is None:
                raise RuntimeError(f"回测返回空结果: {day_str}")

            # --- 提取当日 equity ---
            day_equity = initial_cash
            if hasattr(result, 'equity_curve') and result.equity_curve is not None:
                eq = result.equity_curve
                if 'date' in eq.columns and not eq.empty:
                    eq['date'] = pd.to_datetime(eq['date']).dt.normalize()
                    day_rows = eq[eq['date'] == td]
                    if not day_rows.empty:
                        equity_col = _pick_col(day_rows, ['equity', 'market_value', 'total_equity'])
                        if equity_col:
                            day_equity = float(day_rows[equity_col].iloc[-1])

            # --- 提取当日 switch_log (信号强度/触发原因) ---
            signal_strength = 0.0
            trigger_symbols_list: List[str] = []
            trigger_reason = "none"
            top_factor = "none"
            day_positions: Dict[str, float] = {}
            day_directions: Dict[str, str] = {}

            # 尝试从 switch_log 获取（若 record_decision 被集成）
            if hasattr(result, 'switch_log') and result.switch_log is not None and not result.switch_log.empty:
                sl = result.switch_log
                date_col = "日期" if "日期" in sl.columns else "date"
                sym_col = "品种" if "品种" in sl.columns else "symbol"
                score_col = "综合得分" if "综合得分" in sl.columns else "composite_score"
                reason_col = "原因" if "原因" in sl.columns else "reason"

                if date_col in sl.columns:
                    sl[date_col] = pd.to_datetime(sl[date_col]).dt.normalize()
                    day_decisions = sl[sl[date_col] == td]
                    if not day_decisions.empty:
                        if score_col in day_decisions.columns:
                            scores = day_decisions[score_col].dropna()
                            if not scores.empty:
                                signal_strength = float(scores.mean())
                        if sym_col in day_decisions.columns:
                            trigger_symbols_list = day_decisions[sym_col].unique().tolist()
                        if reason_col in day_decisions.columns:
                            reasons = day_decisions[reason_col].value_counts()
                            if not reasons.empty:
                                trigger_reason = str(reasons.index[0])
                        factor_cols = [c for c in day_decisions.columns if c.startswith("因子_")]
                        if factor_cols:
                            factor_means = {c: day_decisions[c].abs().mean() for c in factor_cols}
                            top_factor = max(factor_means, key=lambda k: factor_means[k]).replace("因子_", "")

            # --- 从 trades 提取 per-symbol 持仓 ---
            if hasattr(result, 'trades') and result.trades is not None and not result.trades.empty:
                trades = result.trades
                sym_col_t = _pick_col(trades, ["symbol", "instrument", "product"])
                entry_col = _pick_col(trades, ["entry_date", "open_date", "entry_bar"])
                exit_col = _pick_col(trades, ["exit_date", "close_date", "exit_bar"])
                shares_col = _pick_col(trades, ["shares", "qty", "quantity"])
                type_col = _pick_col(trades, ["type", "side", "direction"])

                if sym_col_t and entry_col and exit_col and shares_col:
                    trades[entry_col] = pd.to_datetime(trades[entry_col])
                    trades[exit_col] = pd.to_datetime(trades[exit_col])
                    # 归一化：去掉时间部分（trades 时间戳是 16:00，对应日收盘；td 是 00:00）
                    entry_norm = trades[entry_col].dt.normalize()
                    exit_norm = trades[exit_col].dt.normalize()
                    # 活跃持仓 = td 在 [entry_date, exit_date) 区间内（含开仓日，不含平仓日）
                    active = trades[
                        (entry_norm <= td) & (exit_norm > td)
                    ]
                    for sym, grp in active.groupby(sym_col_t):
                        day_positions[sym] = float(grp[shares_col].sum())
                        if type_col and not grp[type_col].empty:
                            day_directions[sym] = str(grp[type_col].iloc[0])
                    # --- 兜底：若区间内无活跃持仓但当日有开仓（entry_date == td），
                    #        用 entry_date 标记开仓日的方向（开仓日虽然 exit 不含，
                    #        但仓位从开仓瞬间就存在，td 当天也应有标记） ---
                    if active.empty:
                        new_entries = trades[entry_norm == td]
                        for _, row in new_entries.iterrows():
                            sym = row[sym_col_t]
                            shares_v = row[shares_col]
                            if sym not in day_positions:
                                day_positions[sym] = float(shares_v)
                                t_val = row.get(type_col, None) if type_col else None
                                if t_val in ("long", "short"):
                                    day_directions[sym] = str(t_val)
                                else:
                                    day_directions[sym] = "long" if shares_v > 0 else "short"

                # 从 trades 推导信号强度（有持仓的品种比例）
                if trigger_symbols_list == []:
                    trigger_symbols_list = list(day_positions.keys())
                if signal_strength == 0.0 and trigger_symbols_list:
                    signal_strength = len(trigger_symbols_list) / len(symbols)
                if trigger_reason == "none" and trigger_symbols_list:
                    trigger_reason = "trade"

            # --- top_factor 反推（switch_log 永远空时使用品种×策略偏好启发式） ---
            if top_factor == "none" and trigger_symbols_list:
                top_factor = _infer_top_factor_from_symbols(trigger_symbols_list)

            # --- 汇总 ---
            total_pos = sum(day_positions.values()) if day_positions else 0
            log_entry = {
                "date": day_str,
                "equity": day_equity,
                "position": total_pos,
                "signal_strength": signal_strength,
                "trigger_symbols": ",".join(trigger_symbols_list) if trigger_symbols_list else "",
                "trigger_reason": trigger_reason,
                "top_factor": top_factor,
                "n_positions": len(day_positions),
                "status": "ok",
            }
            all_logs.append(log_entry)

            for sym in symbols:
                pos = day_positions.get(sym, 0)
                direction = day_directions.get(sym, "flat")
                # position: 用方向编码 (long=+1, short=-1, flat=0)
                pos_signal = 1 if direction == "long" else (-1 if direction == "short" else 0)
                all_positions.append({
                    "date": day_str,
                    "symbol": sym,
                    "signal": pos_signal,
                    "position": pos_signal,  # 用于热力图颜色
                    "signal_strength": abs(pos_signal),
                    "trigger_reason": trigger_reason if pos_signal != 0 else "none",
                    "direction": direction,
                })

            print(f"equity={day_equity:,.0f} pos={total_pos:+.2f} "
                  f"sig={signal_strength:+.3f} symbols={len(trigger_symbols_list)}")

        except ValueError as e:
            # 可恢复错误：数据缺失/配置问题，记录并跳过该日
            print(f"⚠️ 跳过 {day_str}: {e}")
            prev_equity = all_logs[-1]["equity"] if all_logs else initial_cash
            all_logs.append({
                "date": day_str,
                "equity": prev_equity,
                "position": 0,
                "signal_strength": 0,
                "trigger_symbols": "",
                "trigger_reason": "skipped",
                "top_factor": "none",
                "n_positions": 0,
                "status": f"skipped: {e}",
            })
            for sym in symbols:
                all_positions.append({
                    "date": day_str, "symbol": sym,
                    "signal": 0, "position": 0,
                    "signal_strength": 0, "trigger_reason": "skipped",
                    "direction": "flat",
                })
        except Exception as e:
            # 致命错误：立即终止，避免生成误导性日志
            print(f"❌ 致命错误 {day_str}: {e}")
            raise

    result_df = pd.DataFrame(all_logs)
    result_df["date"] = pd.to_datetime(result_df["date"])
    result_df = result_df.sort_values("date").reset_index(drop=True)
    result_df["pnl"] = result_df["equity"].diff().fillna(0)

    positions_df = pd.DataFrame(all_positions)
    if not positions_df.empty:
        positions_df["date"] = pd.to_datetime(positions_df["date"])

    return result_df, positions_df


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="L3 模拟交易 — 每日 OOS 回测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_daily_sim.py --date 2026-06-19
  python scripts/run_daily_sim.py --start 2026-06-01 --end 2026-06-19
  python scripts/run_daily_sim.py --date 2026-06-19 --skip-dashboard
        """,
    )
    parser.add_argument("--date", type=str, help="单日回测 (YYYY-MM-DD)")
    parser.add_argument("--start", type=str, help="区间开始 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="区间结束 (YYYY-MM-DD)，默认今天")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--symbols", nargs="*", help="品种列表（空格分隔）")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0, help="初始资金")
    parser.add_argument("--skip-dashboard", action="store_true", help="跳查看板生成")
    parser.add_argument("--skip-check", action="store_true", help="跳过数据完整性检查")
    args = parser.parse_args()

    # 解析日期
    if args.date:
        start_date = args.date
        end_date = args.date
    elif args.start:
        start_date = args.start
        end_date = args.end or datetime.now().strftime("%Y-%m-%d")
    else:
        parser.error("请指定 --date 或 --start")

    symbols = args.symbols if args.symbols else DEFAULT_SYMBOLS

    print("=" * 70)
    print("L3 模拟交易 — 每日 OOS 回测")
    print("=" * 70)
    print(f"  日期: {start_date} ~ {end_date}")
    print(f"  品种: {symbols}")
    print(f"  策略: 三因子等权 (donchian_breakout + carry + basis_momentum) + 方向二")
    print()

    # 1. 加载数据 + 初始化日历
    print("[1/4] 初始化交易日历...")
    from runner.pipeline import Pipeline
    pipe = Pipeline(args.config).load_data()
    data_source = pipe._data

    cal = TradingCalendar.from_data_source(data_source)
    print(f"  {cal}")
    print(f"  节假日: {cal.holiday_count} 天")

    # 2. 数据完整性检查
    if not args.skip_check:
        print("\n[2/4] 数据完整性检查...")
        summary = check_data_completeness_summary(data_source, symbols)
        for _, row in summary.iterrows():
            status_icon = {"ok": "✓", "warn": "⚠", "error": "✗"}.get(row["status"], "?")
            print(f"  {status_icon} {row['symbol']:12s} {row['status']:5s} "
                  f"last={row.get('last_date', '?')}  gap={row.get('days_gap', '?')}d")
        errors = summary[summary["status"] == "error"]
        if len(errors) > 0:
            print(f"\n  ⚠️  有 {len(errors)} 个品种数据异常，请检查后再运行")
            # 不阻断，继续运行
    else:
        print("\n[2/4] 跳过数据完整性检查")

    # 3. 逐日回测
    print("\n[3/4] 逐日 OOS 回测...")
    L3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        daily_log, positions_log = run_daily_oos_backtest(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            config_path=args.config,
            initial_cash=args.initial_cash,
        )
    except ValueError as e:
        # 区间无交易日 / 数据未到该日期：跳过当日，不生成误导日志
        print(f"\n⚠️ 跳过 {start_date}（数据尚未到位）: {e}")
        return 1
    except Exception as e:
        # 致命错误：终止并打印 stack
        print(f"\n❌ 致命错误: {e}")
        raise

    # 保存日志
    daily_log.to_csv(L3_LOG_FILE, index=False)
    print(f"\n  ✓ 日志已保存: {L3_LOG_FILE} ({len(daily_log)} 行)")

    # 保存 per-symbol 持仓
    positions_log.to_csv(L3_POSITIONS_FILE, index=False)
    print(f"  ✓ 持仓已保存: {L3_POSITIONS_FILE} ({len(positions_log)} 行)")

    # 4. 生成看板
    if not args.skip_dashboard and not daily_log.empty:
        print("\n[4/4] 生成看板...")
        generate_dashboard(
            log_path=str(L3_LOG_FILE),
            output_path=str(L3_DASHBOARD),
            positions_path=str(L3_POSITIONS_FILE),
        )
    else:
        print("\n[4/4] 跳查看板生成")

    # 汇总
    if not daily_log.empty:
        print(f"\n{'=' * 70}")
        print(f"L3 模拟交易完成")
        print(f"  日期: {start_date} ~ {end_date}")
        print(f"  最终净值: {daily_log['equity'].iloc[-1]:,.0f}")
        print(f"  累计 PnL: {daily_log['pnl'].sum():+,.0f}")
        print(f"  日志: {L3_LOG_FILE}")
        if not args.skip_dashboard:
            print(f"  看板: {L3_DASHBOARD}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()