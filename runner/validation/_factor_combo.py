"""组合IC验证 — 统一符号 + 滚动IC加权。

步骤：
1. 计算所有品种/因子面板
2. 对每个 (date, factor) 计算横截面 rank IC
3. 候选因子筛选：full-sample mean |IC| >= min_abs_ic
4. 统一符号：mean_IC < 0 的因子取反
5. 滚动 IC 加权：weight_i(date) = EMA(|IC_i|) over [date-ic_window, date)
6. 组合值：combo_value(date, symbol) = sum(w_i * sign_flipped_factor_i)
7. 计算组合 daily rank IC，输出 mean/IR
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.config.strategy_profiles import StrategyLibrary
from core.ext.factors.alpha_futures.cross_spread import STRONG_IC_PAIRS
from runner.common.utils import save_csv
from runner.validation._factor_cross_spread import build_cross_spread_panel
from runner.validation._factor_panel import build_factor_panel


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
    combine_method: str = "rank",
    **kwargs,
) -> Dict[str, Any]:
    """阶段 B：组合 IC 验证 — 统一符号 + 滚动 IC 加权。

    Args:
        combine_method: "rank"=秩平均合成, "value"=原值加权

    Returns:
        {
          "single": per-factor 排序结果（已带 sign_flip 列）,
          "combo": {"mean_ic", "ir", "n_days", "pass_rule9"},
          "weights_last": 最后一日权重,
        }
    """
    logger.info(f"fwd_period 参数: {fwd_period}")
    _fwd = fwd_period
    symbols = config.symbols

    # ── 1. 构建因子面板 ──
    panel_rows: List[Dict[str, Any]] = []

    # 跨品种价差因子候选（强 IC 配对）
    cross_spread_panel = build_cross_spread_panel(
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

    # 主面板
    panel_df = build_factor_panel(
        data_source=data_source,
        config=config,
        fwd_period=_fwd,
        do_winsorize=do_winsorize,
    )

    if panel_df.empty:
        logger.warning("面板为空")
        return {"single": pd.DataFrame(), "combo": None}

    # 转为行列表以便注入跨品种价差
    panel_rows = panel_df.to_dict("records")

    # ── 2. 注入跨品种价差因子行 ──
    if not cross_spread_panel.empty and panel_rows:
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

    panel = pd.DataFrame(panel_rows)
    panel["date"] = pd.to_datetime(panel["date"])
    all_factors = sorted(panel["factor"].unique())
    logger.info(
        f"面板规模: {len(panel)} 行, {panel['date'].nunique()} 交易日, {len(all_factors)} 因子"
    )

    # ── 3. 全样本 mean IC（用于 sign flip + 候选筛选） ──
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
        f"面板 groupby 统计: 总 {_total_groups} 组, 有效 {_valid_groups} 组, "
        f"有效 IC {len(factor_full_ic)} 个因子"
    )
    if _total_groups > 0 and _valid_groups == 0:
        sample = panel.groupby("date").size().describe()
        logger.info(f"  每组 (date) 行数分布: \n{sample.to_dict()}")

    # ── 4. 候选筛选 + 符号统一 ──
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
    if len(single_df) == 0:
        single_df = pd.DataFrame(
            columns=[
                "factor", "raw_mean_ic", "sign",
                "abs_ic_full", "is_candidate", "sign_flipped_ic",
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

    # ── 5. 滚动 IC 加权 ──
    all_dates = sorted(panel["date"].unique())
    abs_ic_long = pd.DataFrame(
        {f: factor_daily_ic[f].abs().reindex(all_dates) for f in candidates}
    ).ffill()
    ema_weights = abs_ic_long.ewm(halflife=ic_window / 2, adjust=False).mean()
    weight_sums = ema_weights.sum(axis=1).replace(0, np.nan)
    weights_norm = ema_weights.div(weight_sums, axis=0)

    # ── 6. 应用符号到候选因子 ──
    sign_map = dict(zip(single_df["factor"], single_df["sign"]))
    panel_sign = panel[panel["factor"].isin(candidates)].copy()
    panel_sign["value_signed"] = panel_sign.apply(
        lambda r: r["value"] * sign_map[r["factor"]], axis=1
    )

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

    # ── 7. 计算 daily cross-sectional rank IC ──
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
    ic_th = getattr(config.factors_config, "ic_threshold", 0.01)
    ir_th = getattr(config.factors_config, "ir_threshold", 0.1)
    pass_rule9 = abs(mean_ic) >= ic_th and abs(ir) >= ir_th

    # 单因子的"组合贡献"
    weights_static = single_df[single_df["is_candidate"]].set_index("factor")[
        "abs_ic_full"
    ]
    weights_static = weights_static / weights_static.sum()

    # 因子间互相关
    pivot = panel_sign.pivot_table(
        index=["date", "symbol"], columns="factor", values="value_signed"
    )
    corr = pivot.corr().abs()
    np.fill_diagonal(corr.values, np.nan)
    avg_corr = float(corr.mean().mean()) if not corr.isna().all().all() else 0.0

    # 输出报告
    _print_combo_report(
        single_df=single_df,
        mean_ic=mean_ic,
        ir=ir,
        n_days=len(combo_ics),
        avg_corr=avg_corr,
        pass_rule9=pass_rule9,
        combine_method=combine_method,
    )

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


def _print_combo_report(
    single_df: pd.DataFrame,
    mean_ic: float,
    ir: float,
    n_days: int,
    avg_corr: float,
    pass_rule9: bool,
    combine_method: str,
) -> None:
    """打印组合IC验证报告。"""
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
        f"组合 (等权 + 符号统一 + 滚动IC加权 + {method_label}): "
        f"mean IC = {mean_ic:+.4f}, IR = {ir:+.4f}"
    )
    print(f"组合天数: {n_days}, 候选因子互相关均值: {avg_corr:.3f}")
    print(
        f"规则 9: {'✅ 通过' if pass_rule9 else '❌ 未通过'} "
        f"(阈值 |IC|>=0.03 AND |IR|>=0.5)"
    )
    print("=" * 78)
