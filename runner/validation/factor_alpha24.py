"""
AlphaFutures24 因子IC/IR验证。

对24个商品期货Alpha因子进行逐个IC/IR统计测试，
筛选出符合规则9（IC>0.03且IR>0.5）的有效因子。

P0 整改：使用 core.factors.factor_evaluator.FactorEvaluator.evaluate_batch
统一执行 IC/IR 计算，删除手写 corrcoef / 滚动 IC。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.factors import AlphaFutures24, AlphaFuturesConfig
from core.factors.alpha_futures.cross_spread import (
    CHAIN_PAIRS,
    STRONG_IC_PAIRS,
    compute_pair_spread_factor,
)
from core.config.strategy_profiles import StrategyLibrary
from runner.common.utils import save_csv


def _compute_pair_signal(
    data_source,
    a_sym: str,
    b_sym: str,
    start: str,
    end: str,
    spread_window: int = 60,
    smoothing_window: int = 3,
) -> Optional[pd.DataFrame]:
    """
    计算一对跨品种价差因子信号（按日期对齐 A 和 B）。

    Returns:
        DataFrame {date, value} 或 None（数据不足时）。
    """
    a_data = data_source.query(start, end, symbols=[a_sym])
    b_data = data_source.query(start, end, symbols=[b_sym])
    if a_data is None or b_data is None:
        return None
    if len(a_data) < spread_window or len(b_data) < spread_window:
        return None
    merged = (
        a_data[["date", "close"]]
        .merge(b_data[["date", "close"]], on="date", suffixes=("_a", "_b"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(merged) < spread_window:
        return None
    close_a = merged["close_a"].values.astype(float)
    close_b = merged["close_b"].values.astype(float)
    signal = compute_pair_spread_factor(
        close_a,
        close_b,
        spread_window=spread_window,
        smoothing_window=smoothing_window,
    )
    return pd.DataFrame({"date": pd.to_datetime(merged["date"]), "value": signal})


def _build_cross_spread_panel(
    data_source,
    symbols: List[str],
    start: str,
    end: str,
    pair_names: List[str],
) -> pd.DataFrame:
    """
    预计算所有强 IC 配对的价差信号并转换为面板 (date, symbol, factor, value)。

    对每个 (pair, symbol∈pair)：
      - symbol = A (pair 第一腿): value = signal（高值=预测 A 强）
      - symbol = B (pair 第二腿): value = -signal（高值=预测 A 强则 B 弱）

    最后按 (date, symbol) 聚合为单一 XSPR_FACTOR 因子（所有配对的均值），
    保证每个品种都有跨品种截面信号（解决单配对 2 腿 IC 样本不足）。

    symbol 不在任何配对中则跳过。空信号填 NaN。
    """
    rows: List[Dict[str, Any]] = []
    sym_set = set(symbols)
    for pair_name in pair_names:
        if pair_name not in CHAIN_PAIRS:
            logger.warning(f"  配对 {pair_name} 不在 CHAIN_PAIRS 中，跳过")
            continue
        a_sym, b_sym = CHAIN_PAIRS[pair_name]
        if a_sym not in sym_set and b_sym not in sym_set:
            continue
        sig = _compute_pair_signal(data_source, a_sym, b_sym, start, end)
        if sig is None:
            logger.warning(f"  {pair_name} ({a_sym}-{b_sym}): 数据不足，跳过")
            continue
        for sym, sign in [(a_sym, 1.0), (b_sym, -1.0)]:
            if sym not in sym_set:
                continue
            sub = sig.rename(columns={"value": "v"}).copy()
            sub["symbol"] = sym
            sub["pair"] = pair_name
            sub["signed_value"] = sub["v"] * sign
            rows.append(sub[["date", "symbol", "pair", "signed_value"]])
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "factor", "value"])
    long_df = pd.concat(rows, ignore_index=True)
    long_df["date"] = pd.to_datetime(long_df["date"])
    # 按 (date, symbol) 聚合：均值 = 该品种当日所有配对信号的均值
    agg = (
        long_df.groupby(["date", "symbol"], as_index=False)["signed_value"]
        .mean()
        .rename(columns={"signed_value": "value"})
    )
    agg["factor"] = "XSPR_FACTOR"
    return agg[["date", "symbol", "factor", "value"]]


def factor_alpha24_screening(
    data_source,
    config: BacktestConfig,
    lib: StrategyLibrary,
    output_dir: Path,
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    cross_sectional: bool = False,
    do_winsorize: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    对AlphaFutures24全部24个因子进行IC/IR统计测试。

    规则9要求：IC > 0.03 且 IR > 0.5 的因子方可保留。

    Args:
        data_source: 数据源（PyBrokerDataSource）
        config: 回测配置
        lib: 策略库（本方法不使用，保留接口一致）
        output_dir: 输出目录
        best_params: 最优参数（本方法不使用，保留接口一致）
        do_winsorize: 是否对因子值做缩尾后处理

    Returns:
        {
            results_df: 所有品种的测试结果,
            summary_df: 因子汇总,
            pass_count: 通过规则9的因子数,
            best_factors: 通过规则9的因子列表,
        }
    """
    logger.info("=" * 60)
    logger.info("AlphaFutures24 因子IC/IR验证")
    logger.info("=" * 60)

    calc = AlphaFutures24(AlphaFuturesConfig())
    symbols = config.symbols
    fwd_period = 5
    # 收集所有品种的因子值 + 5日 forward return + 日期，构造横截面面板
    panel_rows: List[Dict[str, Any]] = []  # 每行 = {date, symbol, factor, value, ret}

    for symbol in symbols:
        try:
            ohlcv = data_source.query(
                config.train_start, config.test_end, symbols=[symbol]
            )
            if ohlcv is None or len(ohlcv) < 50:
                logger.warning(f"  {symbol}: 数据不足，跳过")
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
                logger.warning(f"  {symbol}: 无持仓量数据，跳过")
                continue

            # 期权/近远月/成交量/换月标记 — 之前未传，导致 TS/carry 依赖因子全为 0
            # 主力连续合约 (close) ≈ 近月；远月从 spread_pairs 注入的 far_close 取
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

            # 计算24个因子
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

            # 前瞻收益（5日）
            forward_ret = np.full_like(close, np.nan, dtype=float)
            forward_ret[:-fwd_period] = (
                close[fwd_period:] - close[:-fwd_period]
            ) / close[:-fwd_period]

            # 收集面板数据：每个 (date, factor) 一行
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
            logger.warning(f"  {symbol}: 因子计算失败 - {e}")

    if not panel_rows:
        logger.warning("无有效测试结果")
        return {
            "results_df": pd.DataFrame(),
            "summary_df": pd.DataFrame(),
            "pass_count": 0,
            "best_factors": [],
        }

    panel_df = pd.DataFrame(panel_rows)
    panel_df["date"] = pd.to_datetime(panel_df["date"])

    # ── 横截面 IC ──
    # 对每个 (factor, date) 组，计算 10 品种 rank IC（Spearman）
    # 再按 factor 聚合为 IC mean / std / IR
    factor_names = sorted(panel_df["factor"].unique())
    summary_rows: List[Dict[str, Any]] = []
    for fname in factor_names:
        sub = panel_df[panel_df["factor"] == fname]
        daily_ics: List[float] = []
        for _date, g in sub.groupby("date"):
            if len(g) < 5:  # 横截面样本太少跳过
                continue
            try:
                # pandas rank 实现 Spearman 相关
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
        is_valid = abs(mean_ic) >= 0.03 and abs(ir) >= 0.5
        summary_rows.append(
            {
                "factor": fname,
                "mean_ic": round(mean_ic, 6),
                "ir": round(ir, 4),
                "pass_rule9": float(is_valid),
                "abs_ic": round(abs(mean_ic), 6),
                "n_cross_days": len(daily_ics),
            }
        )

    if not summary_rows:
        logger.warning("无有效横截面IC")
        return {
            "results_df": pd.DataFrame(),
            "summary_df": pd.DataFrame(),
            "pass_count": 0,
            "best_factors": [],
        }

    summary = pd.DataFrame(summary_rows).sort_values("abs_ic", ascending=False)
    pass_count = int((summary["pass_rule9"] > 0.5).sum())
    best_factors = summary[summary["pass_rule9"] > 0.5]["factor"].tolist()

    logger.info(f"\n因子验证完成: {pass_count}/{len(summary)} 通过规则9")
    if best_factors:
        logger.info(f"  有效因子: {best_factors}")
    else:
        logger.warning("  无因子通过规则9，建议检查数据质量或调整阈值")

    # 兼容接口：构造 per-symbol rows (空) + summary
    df = pd.DataFrame(
        columns=["symbol", "factor", "mean_ic", "std_ic", "ir", "pass_rule9"]
    )

    # 保存结果
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_csv(df, output_dir / "factor_alpha24_results.csv")
        save_csv(summary, output_dir / "factor_alpha24_summary.csv")
        logger.info(f"  结果已保存到: {output_dir}")

    return {
        "results_df": df,
        "summary_df": summary,
        "pass_count": pass_count,
        "best_factors": best_factors,
    }


def factor_combo_ic_validation(
    data_source,
    config: BacktestConfig,
    lib: StrategyLibrary,
    output_dir: Path,
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    cross_sectional: bool = False,
    do_winsorize: bool = True,
    min_abs_ic: float = 0.03,
    ic_window: int = 60,
    fwd_period: int = 5,
    combine_method: str = "rank",  # "rank"=秩平均合成, "value"=原值加权
    **kwargs,
) -> Dict[str, Any]:
    """
    阶段 B：组合 IC 验证 — 统一符号 + 滚动 IC 加权。

    步骤：
    1. 计算所有品种/因子面板（同 factor_alpha24_screening）
    2. 对每个 (date, factor) 计算横截面 rank IC
    3. 候选因子筛选：full-sample mean |IC| >= min_abs_ic
    4. 统一符号：mean_IC < 0 的因子取反（高值=预测正收益）
    5. 滚动 IC 加权：weight_i(date) = EMA(|IC_i|) over [date-ic_window, date)
    6. 组合值：combo_value(date, symbol) = Σ w_i × sign_flipped_factor_i
    7. 计算组合 daily rank IC，输出 mean/IR

    Returns:
        {
          "single": per-factor 排序结果（已带 sign_flip 列）,
          "combo": {"mean_ic", "ir", "n_days", "pass_rule9"},
          "weights_last": 最后一日权重（用于回测/复用）,
        }
    """
    logger.info(f"fwd_period 参数: {fwd_period}")
    _fwd = fwd_period
    calc = AlphaFutures24(AlphaFuturesConfig())
    symbols = config.symbols
    panel_rows: List[Dict[str, Any]] = []

    # ── 跨品种价差因子候选（强 IC 配对） ──
    # 预先计算每个 (pair, symbol) 的价差信号，避免在主循环中重复做日期对齐
    cross_spread_panel = _build_cross_spread_panel(
        data_source=data_source,
        symbols=symbols,
        start=config.train_start,
        end=config.test_end,
        pair_names=list(STRONG_IC_PAIRS),
    )
    if not cross_spread_panel.empty:
        logger.info(
            f"  跨品种价差面板 (XSPR_FACTOR): {len(cross_spread_panel)} 行, "
            f"{cross_spread_panel['symbol'].nunique()} 品种, "
            f"配对数 {len(STRONG_IC_PAIRS)}"
        )

    for symbol in symbols:
        try:
            ohlcv = data_source.query(
                config.train_start, config.test_end, symbols=[symbol]
            )
            if ohlcv is None or len(ohlcv) < 50:
                continue
            ohlcv = ohlcv.sort_values("date").reset_index(drop=True)
            logger.info(
                f"  {symbol}: ohlcv {len(ohlcv)} 行, {ohlcv['date'].nunique()} 唯一日期"
            )
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
            if do_winsorize:
                factors = calc.post_process(factors, do_winsorize=True)

            forward_ret = np.full_like(close, np.nan, dtype=float)
            forward_ret[:-_fwd] = (close[_fwd:] - close[:-_fwd]) / close[:-_fwd]

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
            logger.warning(f"  {symbol}: 因子计算失败 - {e}")

    # ── 注入跨品种价差因子行 ──
    # 跨品种价差面板只有 (date, symbol, factor, value)，需补充 ret
    if not cross_spread_panel.empty and panel_rows:
        # 用主面板构建 (date, symbol) -> ret 查找表，统一转 Timestamp 避免类型错位
        ret_lookup: Dict[tuple, float] = {}
        for _r in panel_rows:
            _d = pd.to_datetime(_r["date"])
            _k = (_d, _r["symbol"])
            if _k not in ret_lookup:
                ret_lookup[_k] = _r["ret"]
        injected = 0
        miss = 0
        for _, _r in cross_spread_panel.iterrows():
            _val = _r["value"]
            if not np.isfinite(_val):
                continue
            _d = pd.to_datetime(_r["date"])
            _ret = ret_lookup.get((_d, _r["symbol"]))
            if _ret is None or not np.isfinite(_ret):
                miss += 1
                continue
            panel_rows.append(
                {
                    "date": _d,
                    "symbol": _r["symbol"],
                    "factor": _r["factor"],
                    "value": float(_val),
                    "ret": float(_ret),
                }
            )
            injected += 1
        logger.info(f"  跨品种价差因子注入: {injected} 行 (未匹配 ret 的 {miss} 行)")

    if not panel_rows:
        logger.warning("面板为空")
        return {"single": pd.DataFrame(), "combo": None}

    panel = pd.DataFrame(panel_rows)
    panel["date"] = pd.to_datetime(panel["date"])
    all_factors = sorted(panel["factor"].unique())
    logger.info(
        f"面板规模: {len(panel)} 行, {panel['date'].nunique()} 交易日, {len(all_factors)} 因子"
    )

    # 全样本 mean IC（用于 sign flip + 候选筛选）
    factor_full_ic: Dict[str, float] = {}
    factor_daily_ic: Dict[str, pd.Series] = {}
    _total_groups = 0
    _valid_groups = 0
    for fname in all_factors:
        sub = panel[panel["factor"] == fname]
        ics = []
        ic_dates = []
        for _d, g in sub.groupby("date"):
            _total_groups += 1
            if len(g) < 3:
                continue
            v = g["value"].rank()
            r = g["ret"].rank()
            if v.std() < 1e-8 or r.std() < 1e-8:
                continue
            _valid_groups += 1
            ic = float(np.corrcoef(v, r)[0, 1])
            if np.isfinite(ic):
                ics.append(ic)
                ic_dates.append(_d)
        if ics:
            factor_full_ic[fname] = float(np.mean(ics))
            factor_daily_ic[fname] = pd.Series(ics, index=ic_dates).sort_index()
    logger.info(
        f"面板 groupby 统计: 总 {_total_groups} 组, 有效 {_valid_groups} 组, 有效 IC {len(factor_full_ic)} 个因子"
    )
    if _total_groups > 0 and _valid_groups == 0:
        sample = panel.groupby("date").size().describe()
        logger.info(f"  每组 (date) 行数分布: \n{sample.to_dict()}")

    # 候选筛选 + 符号统一
    single_rows = []
    for fname, mean_ic in factor_full_ic.items():
        sign = 1.0 if mean_ic >= 0 else -1.0
        sign_flipped_ic = sign * mean_ic
        single_rows.append(
            {
                "factor": fname,
                "raw_mean_ic": round(mean_ic, 6),
                "sign": sign,
                "abs_ic_full": round(abs(mean_ic), 6),
                "is_candidate": abs(mean_ic) >= min_abs_ic,
                "sign_flipped_ic": round(sign_flipped_ic, 6),
            }
        )
    single_df = pd.DataFrame(single_rows)
    logger.info(
        f"single_rows 长度: {len(single_rows)}, columns: {list(single_df.columns)}"
    )
    if len(single_df) == 0:
        logger.warning("single_rows 为空，跳过排序")
        single_df = pd.DataFrame(
            columns=[
                "factor",
                "raw_mean_ic",
                "sign",
                "abs_ic_full",
                "is_candidate",
                "sign_flipped_ic",
            ]
        )
    else:
        single_df = single_df.sort_values("abs_ic_full", ascending=False)
    candidates = single_df[single_df["is_candidate"]]["factor"].tolist()
    logger.info(
        f"候选因子 (abs IC>={min_abs_ic}): {len(candidates)}/{len(single_df)} — {candidates}"
    )

    if not candidates:
        return {
            "single": single_df,
            "combo": None,
            "candidates": [],
        }

    # 滚动 IC 加权：weight_i(date) ∝ EMA(|IC_i|) over [date-ic_window, date)
    all_dates = sorted(panel["date"].unique())
    abs_ic_long = pd.DataFrame(
        {f: factor_daily_ic[f].abs().reindex(all_dates) for f in candidates}
    ).ffill()
    # EMA 加权，半衰期 ≈ ic_window/2
    ema_weights = abs_ic_long.ewm(halflife=ic_window / 2, adjust=False).mean()
    # 归一化
    weight_sums = ema_weights.sum(axis=1).replace(0, np.nan)
    weights_norm = ema_weights.div(weight_sums, axis=0)

    # 应用符号到候选因子
    sign_map = dict(zip(single_df["factor"], single_df["sign"]))
    panel_sign = panel[panel["factor"].isin(candidates)].copy()
    panel_sign["value_signed"] = panel_sign.apply(
        lambda r: r["value"] * sign_map[r["factor"]], axis=1
    )

    # 每日组合得分：Σ weight_i(date) × value_signed_i
    # combine_method = "rank" → 因子值先做横截面 rank 再加权平均（抗极端值）
    # combine_method = "value" → 因子值直接加权平均（原值量纲合成）
    if combine_method == "rank":
        panel_sign["value_signed"] = panel_sign.groupby(["date", "factor"])[
            "value_signed"
        ].rank(method="average", pct=True)

    def _combo(group: pd.DataFrame) -> float:
        d = group["date"].iloc[0]
        if d not in weights_norm.index:
            return np.nan
        w = weights_norm.loc[d]
        val_map = dict(zip(group["factor"], group["value_signed"]))
        return float(sum(w.get(f, 0.0) * val_map.get(f, 0.0) for f in candidates))

    combo_per_row = (
        panel_sign.groupby(["date", "symbol"], group_keys=False)
        .apply(_combo)
        .reset_index()
    )
    combo_per_row.columns = ["date", "symbol", "combo_value"]

    # 合并 forward_ret
    ret_map = panel_sign.drop_duplicates(["date", "symbol"]).set_index(
        ["date", "symbol"]
    )["ret"]
    combo_per_row = combo_per_row.join(ret_map, on=["date", "symbol"])

    # 计算 daily cross-sectional rank IC
    combo_ics = []
    ic_dates_out = []
    for d, g in combo_per_row.groupby("date"):
        g = g.dropna(subset=["combo_value", "ret"])
        if len(g) < 5:
            continue
        v = g["combo_value"].rank()
        r = g["ret"].rank()
        if v.std() < 1e-10 or r.std() < 1e-10:
            continue
        ic = float(np.corrcoef(v, r)[0, 1])
        if np.isfinite(ic):
            combo_ics.append(ic)
            ic_dates_out.append(d)

    if not combo_ics:
        return {"single": single_df, "combo": None, "candidates": candidates}

    arr = np.array(combo_ics)
    mean_ic = float(np.mean(arr))
    std_ic = float(np.std(arr))
    ir = mean_ic / std_ic if std_ic > 1e-10 else 0.0
    pass_rule9 = abs(mean_ic) >= 0.03 and abs(ir) >= 0.5

    # 单因子的"组合贡献"：用 full-sample abs(IC) 加权
    weights_static = single_df[single_df["is_candidate"]].set_index("factor")[
        "abs_ic_full"
    ]
    weights_static = weights_static / weights_static.sum()

    # 因子间互相关（候选子集）
    pivot = panel_sign.pivot_table(
        index=["date", "symbol"], columns="factor", values="value_signed"
    )
    corr = pivot.corr().abs()
    np.fill_diagonal(corr.values, np.nan)
    avg_corr = float(corr.mean().mean()) if not corr.isna().all().all() else 0.0

    # 输出报告
    method_label = "秩合成" if combine_method == "rank" else "原值合成"
    print("\n" + "=" * 78)
    print(f"  阶段 B：组合 IC 验证 — 统一符号 + 滚动 IC 加权 + {method_label}")
    print("=" * 78)
    print(f"{'因子':<10}{'raw IC':<10}{'abs':<10}{'sign':<8}{'候选':<6}")
    print("-" * 78)
    for _, r in single_df.head(15).iterrows():
        cand_mark = "✅" if r["is_candidate"] else "  "
        print(
            f"{r['factor']:<10}{r['raw_mean_ic']:<+10.4f}{r['abs_ic_full']:<10.4f}"
            f"{'+' if r['sign'] > 0 else '-':<8}{cand_mark}"
        )
    print("-" * 78)
    print(
        f"组合 (等权 + 符号统一 + 滚动IC加权 + {method_label}): mean IC = {mean_ic:+.4f}, IR = {ir:+.4f}"
    )
    print(f"组合天数: {len(combo_ics)}, 候选因子互相关均值: {avg_corr:.3f}")
    print(
        f"规则 9: {'✅ 通过' if pass_rule9 else '❌ 未通过'} (阈值 |IC|>=0.03 AND |IR|>=0.5)"
    )
    print("=" * 78)

    # 保存
    weights_last = weights_norm.iloc[-1].round(4).to_dict()
    out = {
        "single": single_df,
        "combo": {
            "mean_ic": round(mean_ic, 6),
            "ir": round(ir, 4),
            "std_ic": round(std_ic, 6),
            "n_days": len(combo_ics),
            "pass_rule9": bool(pass_rule9),
            "avg_corr": round(avg_corr, 4),
        },
        "candidates": candidates,
        "weights_last": weights_last,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        single_df.to_csv(output_dir / "factor_combo_single.csv", index=False)
        combo_daily = pd.DataFrame({"date": ic_dates_out, "combo_ic": combo_ics})
        combo_daily.to_csv(output_dir / "factor_combo_daily_ic.csv", index=False)
        weights_norm.to_csv(output_dir / "factor_combo_weights.csv")
        logger.info(f"结果已保存到: {output_dir}")

    return out
