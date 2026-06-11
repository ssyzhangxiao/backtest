"""因子面板构建 — 从数据源计算24因子并构建横截面面板。"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.factors import AlphaFutures24, AlphaFuturesConfig


def build_factor_panel(
    data_source,
    config: BacktestConfig,
    fwd_period: int = 5,
    do_winsorize: bool = True,
) -> pd.DataFrame:
    """计算所有品种的因子值 + 前瞻收益，构建横截面面板。

    Returns:
        DataFrame with columns: date, symbol, factor, value, ret
    """
    calc = AlphaFutures24(AlphaFuturesConfig())
    symbols = config.symbols
    panel_rows: List[Dict[str, Any]] = []

    for symbol in symbols:
        try:
            ohlcv = data_source.query(
                config.train_start, config.test_end, symbols=[symbol]
            )
            if ohlcv is None or len(ohlcv) < 50:
                logger.warning("  %s: 数据不足，跳过", symbol)
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
                logger.warning("  %s: 无持仓量数据，跳过", symbol)
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

            if do_winsorize:
                factors = calc.post_process(factors, do_winsorize=True)

            forward_ret = np.full_like(close, np.nan, dtype=float)
            forward_ret[:-fwd_period] = (
                close[fwd_period:] - close[:-fwd_period]
            ) / close[:-fwd_period]

            dates_arr = ohlcv["date"].values
            for fname, fvals in factors.items():
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
            logger.warning("  %s: 因子计算失败 - %s", symbol, e)

    if not panel_rows:
        return pd.DataFrame()

    panel_df = pd.DataFrame(panel_rows)
    panel_df["date"] = pd.to_datetime(panel_df["date"])
    return panel_df


def compute_cross_sectional_ic(panel_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """对面板数据计算每个因子的横截面 IC/IR。

    Returns:
        {factor_name: {"mean_ic": ..., "std_ic": ..., "ir": ..., "n_days": ...}}
    """
    factor_names = sorted(panel_df["factor"].unique())
    results: Dict[str, Dict[str, float]] = {}

    for fname in factor_names:
        sub = panel_df[panel_df["factor"] == fname]
        daily_ics: List[float] = []
        for _date, g in sub.groupby("date"):
            if len(g) < 5:
                continue
            try:
                v = g["value"].rank()
                r = g["ret"].rank()
                if v.std() < 1e-10 or r.std() < 1e-10:
                    continue
                ic = float(np.corrcoef(v, r)[0, 1])
            except Exception:
                continue
            if np.isfinite(ic):
                daily_ics.append(ic)

        if not daily_ics:
            continue
        ic_arr = np.array(daily_ics)
        mean_ic = float(np.mean(ic_arr))
        std_ic = float(np.std(ic_arr))
        ir = mean_ic / std_ic if std_ic > 1e-10 else 0.0
        results[fname] = {
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "ir": ir,
            "n_days": len(daily_ics),
        }

    return results
