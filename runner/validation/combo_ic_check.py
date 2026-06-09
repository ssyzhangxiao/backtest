"""组合 IC 验证：对通过 abs(IC)>0.03 的 6 个候选因子做等权组合。"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from core.factors import AlphaFutures24, AlphaFuturesConfig

CANDIDATES = ["TS_03", "TS_01", "TS_02", "H_01", "M_05", "V_02"]


def main() -> None:
    from runner import Pipeline

    pipe = Pipeline("config.yaml").load_data()
    config = pipe._config
    data_source = pipe._data_source
    symbols = config.symbols
    fwd = 5
    panel_rows: List[Dict[str, Any]] = []
    calc = AlphaFutures24(AlphaFuturesConfig())

    for symbol in symbols:
        try:
            ohlcv = data_source.query(
                config.train_start, config.test_end, symbols=[symbol]
            )
            if ohlcv is None or len(ohlcv) < 100:
                continue
            ohlcv = ohlcv.sort_values("date").reset_index(drop=True)
            close = ohlcv["close"].values.astype(float)
            high = ohlcv["high"].values.astype(float)
            low = ohlcv["low"].values.astype(float)
            open_price = ohlcv["open"].values.astype(float)
            oi = (
                ohlcv["open_interest"].values.astype(float)
                if "open_interest" in ohlcv.columns
                else None
            )
            if oi is None:
                continue
            near_price = close
            far_price = (
                ohlcv["far_close"].values.astype(float)
                if "far_close" in ohlcv.columns
                else np.full_like(close, np.nan)
            )
            volume = (
                ohlcv["volume"].values.astype(float)
                if "volume" in ohlcv.columns
                else np.zeros_like(close)
            )
            is_dominant = (
                ohlcv["is_dominant"].values.astype(bool)
                if "is_dominant" in ohlcv.columns
                else None
            )
            delivery_exclude = (
                ohlcv["delivery_exclude"].values.astype(bool)
                if "delivery_exclude" in ohlcv.columns
                else None
            )

            factors = calc.compute_all(
                close=close,
                open_price=open_price,
                high=high,
                low=low,
                open_interest=oi,
                near_price=near_price,
                far_price=far_price,
                volume=volume,
                is_dominant=is_dominant,
                delivery_exclude=delivery_exclude,
            )
            factors = calc.post_process(factors, do_winsorize=True)

            forward_ret = np.full_like(close, np.nan, dtype=float)
            forward_ret[:-fwd] = (close[fwd:] - close[:-fwd]) / close[:-fwd]

            dates_arr = ohlcv["date"].values
            for fname, fvals in factors.items():
                if fname not in CANDIDATES:
                    continue
                for i in range(len(ohlcv)):
                    val = fvals[i]
                    ret = forward_ret[i]
                    if not np.isfinite(val) or not np.isfinite(ret):
                        continue
                    panel_rows.append(
                        {
                            "date": dates_arr[i],
                            "symbol": symbol,
                            "factor": fname,
                            "value": float(val),
                            "ret": float(ret),
                        }
                    )
        except Exception as e:
            logger.warning(f"  {symbol}: 失败 - {e}")

    panel = pd.DataFrame(panel_rows)
    panel["date"] = pd.to_datetime(panel["date"])
    logger.info(f"面板规模: {len(panel)} 行, {panel['date'].nunique()} 交易日")

    # 1) 单因子 cross-sectional IC (与 factor_alpha24 一致)
    single_results = []
    for fname in CANDIDATES:
        sub = panel[panel["factor"] == fname]
        daily_ics = []
        for _d, g in sub.groupby("date"):
            if len(g) < 5:
                continue
            v = g["value"].rank()
            r = g["ret"].rank()
            if v.std() < 1e-10 or r.std() < 1e-10:
                continue
            ic = float(np.corrcoef(v, r)[0, 1])
            if np.isfinite(ic):
                daily_ics.append(ic)
        if not daily_ics:
            continue
        arr = np.array(daily_ics)
        single_results.append(
            {
                "factor": fname,
                "mean_ic": float(np.mean(arr)),
                "ir": float(np.mean(arr) / np.std(arr)),
                "n_days": len(daily_ics),
            }
        )

    # 2) 组合 cross-sectional IC (等权)
    #    对每个 (date, symbol) 聚合 6 个因子为组合得分，再算 rank IC
    combo_panel = (
        panel.groupby(["date", "symbol"])
        .agg(value=("value", "mean"), ret=("ret", "mean"))
        .reset_index()
    )
    combo_ics = []
    for _d, g in combo_panel.groupby("date"):
        if len(g) < 5:
            continue
        v = g["value"].rank()
        r = g["ret"].rank()
        if v.std() < 1e-10 or r.std() < 1e-10:
            continue
        ic = float(np.corrcoef(v, r)[0, 1])
        if np.isfinite(ic):
            combo_ics.append(ic)
    combo_arr = np.array(combo_ics) if combo_ics else np.array([0.0])
    combo_mean = float(np.mean(combo_arr))
    combo_ir = (
        combo_mean / float(np.std(combo_arr))
        if float(np.std(combo_arr)) > 1e-10
        else 0.0
    )

    # 3) 因子互相关 (每天 6 因子 rank 相关, 取均值)
    pivot = panel.pivot_table(
        index=["date", "symbol"], columns="factor", values="value"
    )
    corr_matrix = pivot.corr().abs()
    np.fill_diagonal(corr_matrix.values, np.nan)
    avg_corr = float(corr_matrix.mean().mean())

    # 4) 输出
    print("\n" + "=" * 70)
    print("  阶段 B：组合 IC 验证 — 6 候选因子等权组合")
    print("=" * 70)
    print(f"{'因子':<8}{'|IC|':<10}{'IR':<10}{'天数':<8}{'pass':<8}")
    print("-" * 70)
    for r in single_results:
        passed = "✅" if abs(r["mean_ic"]) >= 0.03 and abs(r["ir"]) >= 0.5 else "  "
        print(
            f"{r['factor']:<8}{abs(r['mean_ic']):<10.4f}{r['ir']:<10.4f}{r['n_days']:<8}{passed}"
        )
    print("-" * 70)
    print(
        f"{'COMBO':<8}{abs(combo_mean):<10.4f}{combo_ir:<10.4f}{len(combo_ics):<8}{'✅' if abs(combo_mean) >= 0.03 and abs(combo_ir) >= 0.5 else '  '}"
    )
    print("=" * 70)
    print(f"\n组合平均互相关: {avg_corr:.3f} (阈值<0.6)")
    print(f"组合 |IC|={abs(combo_mean):.4f}, IR={combo_ir:.4f}")
    print(
        f"规则9 (|IC|>=0.03 AND |IR|>=0.5): {'✅ 通过' if abs(combo_mean) >= 0.03 and abs(combo_ir) >= 0.5 else '❌ 未通过'}"
    )

    # 5) 保存
    out = Path("output_backtest_pybroker/validation/factor_combo_ic.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(single_results)
    rows.append(
        {
            "factor": "COMBO_EQUAL_WEIGHT",
            "mean_ic": combo_mean,
            "ir": combo_ir,
            "n_days": len(combo_ics),
        }
    )
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
