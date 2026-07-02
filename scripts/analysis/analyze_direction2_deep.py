"""
方向二深度分析 — 4 张补充表。

输入:
  - 全期 + OOS 的 summaries.json
  - per_bar_dynamic_b0.3_p0.4.json (用于 Table 3/4)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── 表 1：Pareto 前沿 ──

def pareto_frontier(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """三维 Pareto 前沿：[sharpe ↑, mdd ↑（即 |mdd| ↓）, return ↑]"""
    candidates = [s for s in summaries if s.get("status") == "ok"]
    frontier: List[Dict[str, Any]] = []
    for cand in candidates:
        dominated = False
        for other in candidates:
            if other is cand:
                continue
            sharpe_o = other.get("sharpe") or -1e9
            sharpe_c = cand.get("sharpe") or -1e9
            return_o = other.get("total_return_pct") or -1e9
            return_c = cand.get("total_return_pct") or -1e9
            mdd_o = other.get("max_drawdown_pct") or 0
            mdd_c = cand.get("max_drawdown_pct") or 0
            if (sharpe_o >= sharpe_c
                and return_o >= return_c
                and mdd_o >= mdd_c
                and (sharpe_o > sharpe_c
                     or return_o > return_c
                     or mdd_o > mdd_c)):
                dominated = True
                break
        if not dominated:
            frontier.append(cand)
    frontier.sort(
        key=lambda s: (
            -(s.get("sharpe") or -1e9),
            s.get("max_drawdown_pct") or 0,
        )
    )
    return frontier


def _label(s: Dict[str, Any]) -> str:
    if s.get("blend_method") == "linear":
        return f"{s.get('tag')} (linear)"
    return f"d_b{s.get('xs_position_base')}_p{s.get('xs_opposite_penalty')}"


def _is_recommended(s: Dict[str, Any]) -> bool:
    return (s.get("blend_method") == "dynamic"
            and abs(s.get("xs_position_base", 0) - 0.3) < 1e-6
            and abs(s.get("xs_opposite_penalty", 0) - 0.4) < 1e-6)


def render_pareto(period_name: str, summaries: List[Dict[str, Any]]) -> None:
    print(f"\n=== 表1：{period_name} Pareto 前沿（动态配置）===")
    front = pareto_frontier(summaries)
    print(f"  总配置: {len(summaries)}  |  前沿: {len(front)}")
    print(f"  {'tag':<28}  {'sharpe':>8}  {'return%':>8}  {'mdd%':>7}  {'calmar':>7}")
    for s in front:
        label = _label(s)
        print(
            f"  {label:<28}  "
            f"{s.get('sharpe') or 0:>8.4f}  "
            f"{s.get('total_return_pct') or 0:>7.2f}%  "
            f"{s.get('max_drawdown_pct') or 0:>6.2f}%  "
            f"{_calmar(s):>7.2f}"
        )
    rec_in = any(_is_recommended(s) for s in front)
    if rec_in:
        print(f"  ✓ d_b0.3_p0.4（推荐）**在 Pareto 前沿上**")
    else:
        print(f"  ✗ d_b0.3_p0.4（推荐）不在前沿上（存在严格支配它的配置）")


def render_calmar_top(period_name: str, summaries: List[Dict[str, Any]], top_n: int = 10) -> None:
    print(f"\n=== 表2：{period_name} Calmar Top-{top_n} ===")
    candidates = [s for s in summaries if s.get("status") == "ok"]
    top = sorted(candidates, key=lambda s: _calmar(s), reverse=True)[:top_n]
    print(f"  {'rank':<4}  {'tag':<28}  {'sharpe':>8}  {'return%':>8}  {'mdd%':>7}  {'calmar':>7}")
    for i, s in enumerate(top, 1):
        label = _label(s)
        marker = " ← 推荐" if _is_recommended(s) else ""
        print(
            f"  {i:<4}  {label:<28}  "
            f"{s.get('sharpe') or 0:>8.4f}  "
            f"{s.get('total_return_pct') or 0:>7.2f}%  "
            f"{s.get('max_drawdown_pct') or 0:>6.2f}%  "
            f"{_calmar(s):>7.2f}{marker}"
        )


def render_pos_scale_stats(per_bar_path: Path, period_label: str = "") -> None:
    if not per_bar_path.exists():
        print(f"\n=== 表3：{period_label} pos_scale 调节统计（{per_bar_path} 不存在） ===")
        return
    print(f"\n=== 表3：{period_label} pos_scale 调节统计（来自 {per_bar_path.name}） ===")
    data = json.loads(per_bar_path.read_text(encoding="utf-8"))
    if not data:
        print("  无数据")
        return

    all_scales: List[float] = []
    by_symbol: Dict[str, List[float]] = {}
    opposite_count = 0
    opposite_total = 0
    for entry in data:
        pos_scales = entry.get("pos_scales", {})
        for sym, scale in pos_scales.items():
            if scale is None:
                continue
            all_scales.append(float(scale))
            by_symbol.setdefault(sym, []).append(float(scale))
            if float(scale) < 0.5:
                opposite_count += 1
            opposite_total += 1

    n = len(all_scales)
    print(f"  录制条目: {len(data)} 个调仓日 × {len(by_symbol)} 品种")
    print(f"  总样本数: {n}")
    if n == 0:
        return

    import statistics as stats

    sorted_scales = sorted(all_scales)
    mean_scale = stats.mean(all_scales)
    median_scale = stats.median(all_scales)
    p10 = sorted_scales[int(n * 0.10)]
    p25 = sorted_scales[int(n * 0.25)]
    p75 = sorted_scales[int(n * 0.75)]
    p90 = sorted_scales[int(n * 0.90)]
    near_ceiling = sum(1 for s in all_scales if s > 0.9) / n
    near_base = sum(1 for s in all_scales if s < 0.4) / n
    triggered_penalty = opposite_count / opposite_total if opposite_total else 0.0

    print("  ── pos_scale 分布（按理论公式 base + (ceiling-base)*|z|）──")
    print(f"    mean   = {mean_scale:.4f}")
    print(f"    median = {median_scale:.4f}")
    print(f"    p10/p25/p75/p90 = {p10:.3f} / {p25:.3f} / {p75:.3f} / {p90:.3f}")
    print(f"    > 0.9（接近 ceiling）: {near_ceiling*100:.1f}%")
    print(f"    < 0.4（接近 base）   : {near_base*100:.1f}%")
    print(f"    < 0.5（减仓信号）    : {triggered_penalty*100:.1f}%  ← 触发异号减仓或弱 XS")

    print("  ── 按品种分组 ──")
    for sym in sorted(by_symbol.keys()):
        scales = by_symbol[sym]
        m = stats.mean(scales)
        med = stats.median(scales)
        print(f"    {sym:<12}  n={len(scales):<4}  mean={m:.3f}  median={med:.3f}")


def render_attribution(
    period: str,
    linear_total: float,
    linear_mdd: float,
    dynamic_total: float,
    dynamic_mdd: float,
) -> None:
    print(f"\n=== 表4：{period} 收益归因分解（linear_w1.0 vs dynamic_b0.3_p0.4）===")
    print(f"  纯 CTA 贡献（linear_w1.0, 等权横截面×1.0）: ret={linear_total:+.2f}% mdd={linear_mdd:.2f}%")
    print(f"  动态混合后（dynamic b=0.3 p=0.4）        : ret={dynamic_total:+.2f}% mdd={dynamic_mdd:.2f}%")
    diff = dynamic_total - linear_total
    pct_diff = diff / linear_total * 100 if linear_total else 0
    mdd_reduction = abs(linear_mdd) - abs(dynamic_mdd)
    print(f"  缩放效应（dynamic - linear）              : ret={diff:+.2f}pp  ({pct_diff:+.1f}%)")
    print(f"  回撤变化（|linear_mdd| - |dynamic_mdd|）  : {mdd_reduction:+.2f}pp  ({'+回撤减少' if mdd_reduction > 0 else '⚠ 回撤增加'})")

    if diff < 0 and mdd_reduction > 0:
        print("  ✓ 缩放为负贡献 + 回撤减少 = 典型'减震器'（用收益换回撤）")
    elif diff < 0 and mdd_reduction <= 0:
        print("  ✗ 缩放为负贡献且回撤未减少 = 双重劣势")
    elif diff >= 0 and mdd_reduction > 0:
        print("  ★ 缩放为正贡献且回撤减少 = 帕累托改善（双优）")
    else:
        print("  ⚠ 缩放为正贡献但回撤增加 = 承担更大风险换收益")


def _calmar(s: Dict[str, Any]) -> float:
    mdd = abs(s.get("max_drawdown_pct") or 0)
    if mdd < 1e-6:
        return 0.0
    return (s.get("total_return_pct") or 0) / mdd


def main() -> int:
    base = Path("output_backtest_pybroker")
    full_path = base / "cta_dynamic_sweep_full" / "summaries.json"
    oos_path = base / "cta_dynamic_sweep_oos" / "summaries.json"

    # Table 1 + 2: 各时期独立显示
    for name, path in [("全期 (2016-2025)", full_path),
                        ("OOS (2021-2025)", oos_path)]:
        if not path.exists():
            print(f"[skip] {path} 不存在")
            continue
        summaries = json.loads(path.read_text(encoding="utf-8"))
        render_pareto(name, summaries)
        render_calmar_top(name, summaries, top_n=10)

    # Table 3: pos_scale 调节统计 (全期 + OOS)
    for period, sub in [("全期", "cta_dynamic_sweep_full"),
                         ("OOS", "cta_dynamic_sweep_oos")]:
        per_bar = base / sub / "per_bar_dynamic_b0.3_p0.4.json"
        render_pos_scale_stats(per_bar, period_label=period)

    # Table 4: 收益归因分解 (全期 + OOS)
    for period, sub in [("全期", "cta_dynamic_sweep_full"),
                         ("OOS", "cta_dynamic_sweep_oos")]:
        s_path = base / sub / "summaries.json"
        if not s_path.exists():
            continue
        s_list = json.loads(s_path.read_text(encoding="utf-8"))
        lin_w1 = next((s for s in s_list if s.get("tag") == "linear_w1.0"), None)
        dyn = next((s for s in s_list if _is_recommended(s)), None)
        if lin_w1 and dyn:
            render_attribution(
                period=period,
                linear_total=lin_w1.get("total_return_pct") or 0,
                linear_mdd=lin_w1.get("max_drawdown_pct") or 0,
                dynamic_total=dyn.get("total_return_pct") or 0,
                dynamic_mdd=dyn.get("max_drawdown_pct") or 0,
            )

    # 跨时期稳定性对比
    print("\n=== 跨时期稳定性：b=0.3 p=0.4 vs linear_w1.0 ===")
    for period, path in [("全期", full_path), ("OOS", oos_path)]:
        if not path.exists():
            continue
        s_list = json.loads(path.read_text(encoding="utf-8"))
        lin = next((s for s in s_list if s.get("tag") == "linear_w1.0"), None)
        dyn = next((s for s in s_list if _is_recommended(s)), None)
        if not (lin and dyn):
            continue
        dyn_all = [s for s in s_list if s.get("blend_method") == "dynamic"
                    and s.get("status") == "ok"]
        dyn_all_sorted = sorted(dyn_all, key=lambda s: _calmar(s), reverse=True)
        rank_dyn = next((i for i, s in enumerate(dyn_all_sorted, 1)
                          if _is_recommended(s)), -1)
        print(f"  [{period}] linear_w1.0 : ret={lin['total_return_pct']:+.2f}% mdd={lin['max_drawdown_pct']:.2f}% calmar={_calmar(lin):.2f}")
        print(f"  [{period}] dyn_b0.3_p0.4: ret={dyn['total_return_pct']:+.2f}% mdd={dyn['max_drawdown_pct']:.2f}% calmar={_calmar(dyn):.2f}  (rank#{rank_dyn}/{len(dyn_all)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
