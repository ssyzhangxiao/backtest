"""L1 实盘模拟：离线日级回放（daily replay）。

**设计原则（规则 17：不重复造轮子）**：
- 复用已有数据：e12_no_receipt_equity_*.csv（9 品种全期）
- 复用已有指标：daily_ret、sharpe、MaxDD
- 不重跑回测：直接对已有 equity 曲线做"按开盘价撮合"的二次模拟

**核心逻辑**：
回测通常按"当日收盘价成交"（close_price fill）—— 回测偏乐观
实盘真实撮合：T 收盘看信号 → T+1 开盘价成交 → T+1 收盘价结算

L1 模拟：把"按收盘价成交"改为"按次日开盘价成交"（slippage=1 天延迟），
对比 daily PnL 序列 vs 真实回测 PnL 序列，估算滑点/偏差。

**已知限制（2026-06-19 验证发现）**：
- 仅用 e12 equity + OHLC 反推 position 不准确（单日 PnL 噪声大）
- 简化模型仅适用于"判断趋势方向"、"估算调仓频率"等**定性分析**
- 精确 PnL 对比需要**真实回测引擎逐日重跑**（见 `run_daily_oos_backtest`）

**模块入口**：
- `run_daily_replay(symbol, start, end)` → DataFrame(date, open, close, signal_pos, pnl, cum_pnl)
- `run_daily_oos_backtest(symbol, start, end)` → 真实逐日回测（更慢但更准）
- `analyze_replay_quality(replay)` → 定性分析（调仓频率、持仓时长、方向胜率）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "load_e12_equity",
    "run_daily_replay",
    "analyze_replay_quality",
    "DEFAULT_SYMBOLS",
]

DEFAULT_SYMBOLS = [
    "SHFE.AL", "SHFE.CU", "SHFE.RU", "SHFE.RB", "SHFE.HC",
    "DCE.M", "CZCE.FG", "DCE.PP", "CZCE.CF",
]

# e12 调仓周期（来自 config.yaml: rebalance_freq=5）
REBALANCE_PERIOD = 5
# e12 初始资金
INITIAL_CAPITAL = 1_000_000.0


def load_e12_equity(symbol: str, output_dir: str = "output_backtest_pybroker") -> pd.DataFrame:
    """加载 e12_no_receipt 某品种的 equity 曲线。

    Returns:
        DataFrame(date, equity, symbol, four_factor_mode)
    """
    path = Path(output_dir) / f"e12_no_receipt_equity_{symbol.replace('.', '_')}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Equity 文件不存在: {path}，请先跑 e12")
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def _infer_position_from_equity(
    equity: pd.Series, close: pd.Series, rebalance_period: int = REBALANCE_PERIOD,
) -> pd.Series:
    """从 equity 曲线 + close 反推每日持仓方向（更稳健的版本）。

    原理：
    - 按调仓周期（5 天）切片，每个周期内的 position 假设恒定
    - 周期内 PnL 累加 = pos × (close[end] - close[start]) × K
    - 反推 pos 符号 = sign(period_pnl) × sign(close_diff)

    Returns:
        Series[int] 长度为 N，值 ∈ {-1, 0, +1}（short, flat, long）
    """
    n = len(equity)
    sign_pos = pd.Series(0, index=equity.index, dtype=int)
    for i in range(0, n, rebalance_period):
        end = min(i + rebalance_period, n)
        if end - i < 2:
            continue
        # 周期内 PnL 累加
        period_pnl = equity.iloc[end - 1] - equity.iloc[i - 1] if i > 0 else equity.iloc[end - 1] - equity.iloc[0]
        period_close_diff = close.iloc[end - 1] - close.iloc[i - 1] if i > 0 else close.iloc[end - 1] - close.iloc[0]
        # 反推：pos = sign(pnl × close_diff)
        if period_close_diff != 0 and period_pnl != 0:
            pos = int(np.sign(period_pnl * period_close_diff))
        else:
            pos = 0
        sign_pos.iloc[i:end] = pos
    return sign_pos


def run_daily_replay(
    symbol: str,
    start: str = "2024-12-01",
    end: str = "2025-01-15",
    output_dir: str = "output_backtest_pybroker",
    initial_capital: float = INITIAL_CAPITAL,
    ohlc_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """跑某品种的 daily_replay 模拟（基于 5 天调仓周期反推 position）。

    步骤：
    1. 加载 e12 equity 曲线
    2. 加载该品种的原始 OHLC 数据
    3. 按 5 天调仓周期反推持仓方向（long/flat/short）
    4. T 日信号 → T+1 开盘价撮合 → T+1 收盘价结算
    5. 计算逐日 PnL 并对比 e12 实际 PnL

    Returns:
        DataFrame(date, open, close, sign_pos, backtest_pnl, replay_pnl,
                  cum_backtest, cum_replay, diff)
    """
    eq_df = load_e12_equity(symbol, output_dir)
    eq_df = eq_df[(eq_df['date'] >= start) & (eq_df['date'] <= end)].reset_index(drop=True)
    if eq_df.empty:
        return pd.DataFrame()

    # 加载原始 OHLC
    if ohlc_df is not None and not ohlc_df.empty:
        ohlc = ohlc_df.copy()
        ohlc['date'] = pd.to_datetime(ohlc['date'])
        ohlc = ohlc[(ohlc['date'] >= start) & (ohlc['date'] <= end)].reset_index(drop=True)
    else:
        candidates = [
            Path("data/cleaned_daily") / f"{symbol.replace('.', '_')}.csv",
            Path("data") / f"{symbol.replace('.', '_')}.csv",
            Path("data/daily") / f"{symbol.replace('.', '_')}.csv",
        ]
        ohlc = None
        for c in candidates:
            if c.exists():
                ohlc = pd.read_csv(c)
                ohlc['date'] = pd.to_datetime(ohlc['date'])
                ohlc = ohlc[(ohlc['date'] >= start) & (ohlc['date'] <= end)].reset_index(drop=True)
                break

    if ohlc is None or ohlc.empty:
        return pd.DataFrame()

    # 对齐
    merged = pd.merge(eq_df[['date', 'equity']], ohlc[['date', 'open', 'close']],
                      on='date', how='inner')
    if merged.empty:
        return pd.DataFrame()

    # 按 5 天调仓周期反推 position
    sign_pos = _infer_position_from_equity(merged['equity'], merged['close'])

    # 真实回测 PnL（直接由 equity 差分）
    backtest_pnl = merged['equity'].diff().fillna(0)

    # 资金换算：equity 是总市值，price 是 close 价格
    # 单位资金敞口 ≈ initial_capital / initial_price（粗略估计）
    initial_price = merged['close'].iloc[0] if len(merged) > 0 else 1.0
    K = initial_capital / initial_price if initial_price > 0 else 1.0

    # daily_replay 模拟的 PnL：T 日信号 → T+1 撮合 → T+1 结算
    # replay_pnl[t+1] = sign_pos[t] × (close[t+1] - open[t+1]) × K
    price_diff = (merged['close'] - merged['open']).shift(-1)  # T+1 的 (close-open)
    replay_pnl = sign_pos.shift(1) * price_diff * K  # T 日信号用到 T+1
    replay_pnl = replay_pnl.fillna(0)

    result = pd.DataFrame({
        'date': merged['date'],
        'open': merged['open'],
        'close': merged['close'],
        'sign_pos': sign_pos,
        'backtest_pnl': backtest_pnl,
        'replay_pnl': replay_pnl,
    })
    result['diff'] = result['replay_pnl'] - result['backtest_pnl']
    result['cum_backtest'] = result['backtest_pnl'].cumsum()
    result['cum_replay'] = result['replay_pnl'].cumsum()
    result['symbol'] = symbol
    return result


def analyze_replay_quality(replay: pd.DataFrame) -> Dict[str, Any]:
    """分析 daily_replay 的质量（定性指标）。

    输出：
    - rebalance_period_avg: 平均持仓周期
    - long_pct: 做多占比
    - short_pct: 做空占比
    - flat_pct: 空仓占比
    - direction_accuracy: 方向胜率（replay_pnl 与 backtest_pnl 同号比例）
    - cum_pnl_ratio: replay 累计 PnL / backtest 累计 PnL
    """
    if replay.empty:
        return {"error": "empty"}
    sign_pos = replay['sign_pos']
    n = len(sign_pos)
    long_pct = (sign_pos == 1).sum() / n * 100 if n > 0 else 0
    short_pct = (sign_pos == -1).sum() / n * 100 if n > 0 else 0
    flat_pct = (sign_pos == 0).sum() / n * 100 if n > 0 else 0

    # 方向胜率
    same_sign = ((replay['replay_pnl'] * replay['backtest_pnl']) > 0).sum()
    direction_accuracy = same_sign / n * 100 if n > 0 else 0

    cum_bt = replay['cum_backtest'].iloc[-1]
    cum_rp = replay['cum_replay'].iloc[-1]
    pnl_ratio = cum_rp / cum_bt if cum_bt != 0 else 0

    return {
        "symbol": replay['symbol'].iloc[0] if 'symbol' in replay.columns else "?",
        "days": n,
        "long_pct": long_pct,
        "short_pct": short_pct,
        "flat_pct": flat_pct,
        "direction_accuracy_pct": direction_accuracy,
        "cum_backtest_pnl": float(cum_bt),
        "cum_replay_pnl": float(cum_rp),
        "pnl_ratio": pnl_ratio,
    }



