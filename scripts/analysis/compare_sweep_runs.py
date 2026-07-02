"""
对比 sweep 旧结果 vs 新 baseline（weight=1.0 数据稳定性验证）。

用法：
    python scripts/compare_sweep_runs.py
    python scripts/compare_sweep_runs.py --old <old.json> --new <new.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比两次 sweep 结果")
    parser.add_argument(
        "--old",
        default="output_backtest_pybroker/cta_weight_sweep/sweep_results.json",
        help="旧结果 JSON 路径",
    )
    parser.add_argument(
        "--new",
        default="output_backtest_pybroker/cta_weight_sweep/baseline/sweep_results.json",
        help="新结果 JSON 路径",
    )
    parser.add_argument(
        "--output",
        default="output_backtest_pybroker/cta_weight_sweep/comparison.md",
        help="对比报告输出路径",
    )
    return parser.parse_args()


def _load(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_pct(v: Any) -> str:
    # total_return_pct/max_drawdown_pct 已是百分比单位（如 6.25 表示 6.25%），
    # 仅追加 % 后缀，不应再 ×100
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}%"
    except (TypeError, ValueError):
        return str(v)


def _by_weight(rows: List[Dict[str, Any]]) -> Dict[float, Dict[str, Any]]:
    out: Dict[float, Dict[str, Any]] = {}
    for r in rows:
        w = r.get("weight")
        if w is None:
            continue
        try:
            w_f = float(w)
        except (TypeError, ValueError):
            continue
        out[w_f] = r
    return out


def _diff_pct(old: Any, new: Any) -> str:
    if old is None or new is None:
        return "—"
    try:
        o, n = float(old), float(new)
        if o == 0:
            return "n/a"
        d = (n - o) / abs(o) * 100
        return f"{d:+.2f}%"
    except (TypeError, ValueError):
        return "—"


def render(
    old_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
    output: str,
) -> None:
    old_by = _by_weight(old_rows)
    new_by = _by_weight(new_rows)

    all_weights = sorted(set(old_by.keys()) | set(new_by.keys()))

    lines = [
        "# Sweep 对比报告（稳定性验证）",
        "",
        f"旧结果：{len(old_rows)} 个 weight",
        f"新结果：{len(new_rows)} 个 weight",
        "",
        "## 一、weight=1.0 复现性验证（关键）",
        "",
        "| 指标 | 旧 | 新 | 相对差 | 解读 |",
        "|------|----|----|--------|------|",
    ]

    # 单独抽出 weight=1.0 做稳定性判定
    stable_metrics = ["sharpe", "total_return_pct", "max_drawdown_pct", "trade_count"]
    consistency_verdict: List[str] = []
    if 1.0 in old_by and 1.0 in new_by:
        for key in stable_metrics:
            o = old_by[1.0].get(key)
            n = new_by[1.0].get(key)
            rel = _diff_pct(o, n)
            lines.append(f"| {key} | {_fmt(o)} | {_fmt(n)} | {rel} | |")
        # 判定：sharpe 相对差 < 5% 算稳定
        o_sharpe = old_by[1.0].get("sharpe")
        n_sharpe = new_by[1.0].get("sharpe")
        if o_sharpe and n_sharpe:
            rel_sharpe = abs(float(n_sharpe) - float(o_sharpe)) / max(abs(float(o_sharpe)), 1e-9)
            if rel_sharpe < 0.05:
                consistency_verdict.append(
                    f"✅ weight=1.0 Sharpe 相对差 {rel_sharpe*100:.2f}% < 5% — **数据稳定**，可以信任新结果"
                )
            else:
                consistency_verdict.append(
                    f"⚠️ weight=1.0 Sharpe 相对差 {rel_sharpe*100:.2f}% ≥ 5% — **数据漂移**，需查因（随机种子/缓存/时间窗漂移）"
                )
    else:
        lines.append("| — | 旧未跑 | 新未跑 | — | weight=1.0 至少要有一边数据 |")
        consistency_verdict.append("❓ weight=1.0 缺少对照")

    lines.extend(["", "## 二、weight=0.9 / 0.95 对比", ""])
    lines.append(
        "| weight | 旧 sharpe | 新 sharpe | sharpe差 | 旧 return% | 新 return% | return%差 | 状态 |"
    )
    lines.append(
        "|--------|-----------|-----------|----------|------------|------------|-----------|------|"
    )
    for w in [0.9, 0.95]:
        o = old_by.get(w, {})
        n = new_by.get(w, {})
        o_s = o.get("sharpe")
        n_s = n.get("sharpe")
        o_r = o.get("total_return_pct")
        n_r = n.get("total_return_pct")
        s_diff = _diff_pct(o_s, n_s)
        r_diff = _diff_pct(o_r, n_r)
        lines.append(
            f"| {w} | {_fmt(o_s)} | {_fmt(n_s)} | {s_diff} | "
            f"{_fmt_pct(o_r)} | {_fmt_pct(n_r)} | {r_diff} | "
            f"{'新' if w not in old_by else '对照'} |"
        )

    lines.extend(["", "## 三、判定", ""])
    lines.extend(f"- {v}" for v in consistency_verdict)
    if not consistency_verdict:
        lines.append("- 无 weight=1.0 对照数据，跳过稳定性判定")

    lines.extend(["", "## 四、原始数据", "", "### 旧（最近一次）", "```json"])
    lines.append(json.dumps(old_rows, indent=2, ensure_ascii=False, default=str)[:2000])
    lines.append("```")
    lines.extend(["", "### 新（baseline）", "```json"])
    lines.append(json.dumps(new_rows, indent=2, ensure_ascii=False, default=str)[:2000])
    lines.append("```")

    md = "\n".join(lines)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"报告已写入: {out_path}")
    print()
    print(md)


def main() -> int:
    args = parse_args()
    old_rows = _load(args.old)
    new_rows = _load(args.new)
    if not old_rows and not new_rows:
        print(f"❌ 旧/新结果都不存在：\n  old: {args.old}\n  new: {args.new}")
        return 1
    render(old_rows, new_rows, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
