"""
实验 A：单品种单策略 default vs best_params 对比

目标：验证 best_params（trend.window=5）相比 default（trend.window=20）
      能否真正提升 SHFE.RB 品种上 trend 策略的 IC 和 signal-to-PnL Sharpe。

方法：
  1. 加载 SHFE.RB 单品种数据（TqSdk）
  2. 调 sub_strategy_aggregator 拿 trend signal 时序（两遍：default / best）
  3. 计算并对比：
     - trend_signal 与下期收益的 Pearson IC / 滚动 IC IR
     - 自建 signal-to-PnL 引擎：signal * 下期收益 = 策略日 PnL
     - PnL 序列的 Sharpe、最大回撤、胜率
"""

import os
import sys

sys.path.insert(0, "/Users/mac/Documents/期货策略回测/回测学习/quant_system")

import numpy as np
import pandas as pd
from loguru import logger

from core.engine.pybroker_data_source import (
    create_hybrid_data_source,
    PyBrokerDataSource,
)

SYMBOL = "SHFE.RB"
STRATEGY = "trend"

logger.remove()
logger.add(sys.stdout, level="INFO")


def load_single_symbol(symbol: str) -> PyBrokerDataSource:
    phone = os.environ.get("TQSDK_PHONE")
    password = os.environ.get("TQSDK_PASSWORD")
    return create_hybrid_data_source(
        phone=phone,
        password=password,
        symbols=[symbol],
        data_dir="data",
        data_length=4000,
    )


def extract_trend_signals(ds: PyBrokerDataSource, custom_params) -> pd.Series:
    from core.engine.sub_strategy_indicators import register_default_indicators

    register_default_indicators()
    from core.ext.factors.alpha_futures.sub_strategy_aggregator import (
        compute_sub_strategy_scores_from_ohlcv,
    )

    df = ds.to_pybroker_df()
    rb = df[df["symbol"] == SYMBOL].copy()
    captured = dict(custom_params or {})
    scored = compute_sub_strategy_scores_from_ohlcv(
        rb,
        strategy_params={"trend": captured} if captured else None,
    )
    scored.index = rb["date"].values
    return scored["trend"]


def compute_ic(signal: pd.Series, price: pd.Series, horizon: int = 1) -> dict:
    ret = price.pct_change(horizon).shift(-horizon)
    valid = signal.notna() & ret.notna() & np.isfinite(signal) & np.isfinite(ret)
    s = signal[valid].values
    r = ret[valid].values
    if len(s) < 30:
        return {
            "ic_overall": np.nan,
            "ic_rolling_mean": np.nan,
            "ic_rolling_std": np.nan,
            "ic_ir": np.nan,
            "n": len(s),
        }
    corr = np.corrcoef(s, r)[0, 1]
    df = pd.DataFrame({"s": pd.Series(s), "r": pd.Series(r)})
    df.index = pd.DatetimeIndex(signal[valid].index)
    df["ic"] = df["s"].rolling(60).corr(df["r"])
    ic_mean = df["ic"].mean()
    ic_std = df["ic"].std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
    return {
        "ic_overall": float(corr),
        "ic_rolling_mean": float(ic_mean),
        "ic_rolling_std": float(ic_std),
        "ic_ir": float(ic_ir),
        "n": len(s),
    }


def signal_to_pnl_metrics(
    signal: pd.Series, price: pd.Series, horizon: int = 1, scale: float = 0.1
) -> dict:
    """把 [-1, 1] signal 映射到日 PnL：pnl_t = signal_t * ret_{t+1} * scale。"""
    ret = price.pct_change(horizon).shift(-horizon)
    pnl = (signal * ret * scale).dropna()
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) < 30:
        return {
            "pnl_sharpe": np.nan,
            "pnl_ann_return": np.nan,
            "pnl_max_dd": np.nan,
            "pnl_win_rate": np.nan,
            "pnl_n_days": len(pnl),
        }
    ann = 252
    sharpe = pnl.mean() / pnl.std() * np.sqrt(ann) if pnl.std() > 0 else 0
    cum = (1 + pnl).cumprod()
    ann_return = cum.iloc[-1] ** (ann / len(pnl)) - 1
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return {
        "pnl_sharpe": float(sharpe),
        "pnl_ann_return": float(ann_return),
        "pnl_max_dd": float(dd.min()),
        "pnl_win_rate": float((pnl > 0).mean()),
        "pnl_n_days": len(pnl),
    }


def main():
    print("=" * 70)
    print(f"A 实验：{SYMBOL} + {STRATEGY}  default(window=20) vs best(window=5)")
    print("=" * 70)

    ds = load_single_symbol(SYMBOL)
    df = ds.to_pybroker_df()
    rb = df[df["symbol"] == SYMBOL].sort_values("date").reset_index(drop=True)
    print(f"RB 数据范围: {rb['date'].min()} ~ {rb['date'].max()}, {len(rb)} 行")

    cases = {
        "A_default_window20": {},
        "A_best_window5": {"trend": {"window": 5}},
    }

    results = {}
    for name, params in cases.items():
        print(f"\n{'=' * 70}\n[Case] {name}: custom_params={params}\n{'=' * 70}")
        signal = extract_trend_signals(ds, params.get(STRATEGY) if params else None)
        price_ser = rb.set_index("date")["close"]
        ic_dict = compute_ic(signal, price_ser, horizon=1)
        pnl_dict = signal_to_pnl_metrics(signal, price_ser, horizon=1)
        results[name] = {
            "ic": ic_dict,
            "pnl": pnl_dict,
            "signal_mean": float(signal.mean()),
            "signal_std": float(signal.std()),
            "signal_nonzero_pct": float((signal.abs() > 0.01).mean() * 100),
        }
        print(f"  IC overall       : {ic_dict['ic_overall']:+.4f}")
        print(f"  IC rolling mean  : {ic_dict['ic_rolling_mean']:+.4f}")
        print(f"  IC rolling std   : {ic_dict['ic_rolling_std']:.4f}")
        print(f"  IC IR            : {ic_dict['ic_ir']:+.4f}")
        print(f"  Signal mean      : {results[name]['signal_mean']:+.4f}")
        print(f"  Signal std       : {results[name]['signal_std']:.4f}")
        print(f"  |signal|>0.01 %  : {results[name]['signal_nonzero_pct']:.1f}%")
        print(f"  --- signal-to-PnL (horizon=1d, scale=10%) ---")
        print(f"  PnL Sharpe       : {pnl_dict['pnl_sharpe']:+.4f}")
        print(f"  PnL Ann Return   : {pnl_dict['pnl_ann_return']:+.2%}")
        print(f"  PnL Max DD       : {pnl_dict['pnl_max_dd']:+.2%}")
        print(f"  PnL Win Rate     : {pnl_dict['pnl_win_rate']:+.2%}")
        print(f"  PnL N Days       : {pnl_dict['pnl_n_days']}")

    print("\n" + "=" * 70)
    print("汇总：")
    print("=" * 70)
    for k, v in results.items():
        print(f"\n{k}:")
        for kk, vv in v.items():
            print(f"  {kk}: {vv}")

    out_dir = "/Users/mac/Documents/期货策略回测/回测学习/quant_system/output_backtest_pybroker"
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for k, v in results.items():
        row = {"case": k}
        for kk, vv in v.items():
            if isinstance(vv, dict):
                for kkk, vvv in vv.items():
                    row[kkk] = vvv
            else:
                row[kk] = vv
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{out_dir}/a_single_strategy_compare.csv", index=False)
    print(f"\n已保存: {out_dir}/a_single_strategy_compare.csv")


if __name__ == "__main__":
    main()
