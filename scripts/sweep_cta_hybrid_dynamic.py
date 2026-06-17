"""
方向二：横截面动态混合 sweep — 扫 xs_position_base / xs_opposite_penalty
找出能让 CTA 收益不降、回撤减少的动态仓位参数。

逻辑：
  - 用 hybrid_blend_method=dynamic 模式
  - 横截面不输出方向，只输出仓位缩放系数
  - 网格扫 (base, penalty) 二维空间
  - 对比基线（hybrid_blend_method=linear, weight=0.9/0.95/1.0）

用法：
    # 默认 3x3 网格（base ∈ {0.4, 0.5, 0.6}, penalty ∈ {0.3, 0.5, 0.7}）
    python scripts/sweep_cta_hybrid_dynamic.py

    # 自定义网格
    python scripts/sweep_cta_hybrid_dynamic.py \
        --bases 0.3 0.4 0.5 0.6 0.7 \
        --penalties 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
        --ceiling 1.0

    # 含线性基线（自动加 3 行 linear@0.9/0.95/1.0）
    python scripts/sweep_cta_hybrid_dynamic.py --include-linear
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="方向二：横截面动态仓位缩放参数扫描")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--bases",
        nargs="+",
        type=float,
        default=[0.4, 0.5, 0.6],
        help="xs_position_base 网格（XS 强度=0 时的 CTA 仓位下限）",
    )
    parser.add_argument(
        "--ceiling",
        type=float,
        default=1.0,
        help="xs_position_ceiling 固定值（默认 1.0 = 满仓）",
    )
    parser.add_argument(
        "--penalties",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.7],
        help="xs_opposite_penalty 网格（CTA 与 XS 异号时减仓系数）",
    )
    parser.add_argument("--signal-mode", default="hybrid")
    parser.add_argument("--data-source", default="csv", choices=["csv", "tqsdk"])
    parser.add_argument(
        "--output-dir", default="output_backtest_pybroker/cta_dynamic_sweep"
    )
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument(
        "--include-linear",
        action="store_true",
        help="同时跑 linear 基线（weight=0.9/0.95/1.0）",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="覆盖 config.yaml 的 full_start_date（不传则用 yaml 默认）",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="覆盖 config.yaml 的 full_end_date（不传则用 yaml 默认）",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="输出目录后缀（如 'oos' / 'full'），便于分两段跑隔离结果",
    )
    return parser.parse_args()


def _run_one(
    args: argparse.Namespace,
    *,
    blend_method: str,
    cta_hybrid_weight: float,
    xs_position_base: float = 0.5,
    xs_position_ceiling: float = 1.0,
    xs_opposite_penalty: float = 0.5,
    tag: str = "",
) -> Dict[str, Any]:
    """单参数组合回测。"""
    from loguru import logger
    from core.config import BacktestConfig
    from core.execution.backtest_runner import PyBrokerBacktestRunner
    from core.engine.pybroker_data_source import PyBrokerDataSource

    t0 = time.time()
    try:
        overrides = {
            "backtest__cta_hybrid_weight": cta_hybrid_weight,
            "backtest__signal_mode": args.signal_mode,
            "backtest__hybrid_blend_method": blend_method,
            "backtest__xs_position_base": xs_position_base,
            "backtest__xs_position_ceiling": xs_position_ceiling,
            "backtest__xs_opposite_penalty": xs_opposite_penalty,
        }
        if args.start_date is not None:
            overrides["backtest__full_start_date"] = args.start_date
        if args.end_date is not None:
            overrides["backtest__full_end_date"] = args.end_date
        config = BacktestConfig.from_yaml(args.config, overrides=overrides)

        import yaml

        with open(args.config, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
        data_cfg = raw_config.get("data", {})
        data_dir = data_cfg.get("csv_data_dir") or "./data"
        adapter_factory = __import__(
            "core.ext.adapters", fromlist=["create_data_source"]
        )
        adapter = adapter_factory.create_data_source("csv", data_dir=data_dir)
        loader = adapter._loader
        csv_paths = []
        for sym in config.symbols:
            for fname in [f"{sym}.csv", f"{sym.replace('.', '_')}.csv"]:
                p = Path(data_dir) / fname
                if p.exists():
                    csv_paths.append(str(p))
                    break
        loader.load_csv_files_by_paths(csv_paths)
        df = loader.full_df
        col_map = {
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "open_interest": "open_interest",
            "symbol": "symbol",
        }
        df_std = df.rename(
            columns={k: v for k, v in col_map.items() if k in df.columns}
        )
        ds = PyBrokerDataSource(df=df_std)

        runner = PyBrokerBacktestRunner(
            data_source=ds, config=config, target_symbols=config.symbols
        )
        strategies = args.strategies or [
            "trend",
            "term_structure",
            "mean_reversion",
            "vol_breakout",
            "composite_resonance",
        ]
        runner.register_strategies(strategies)

        result = runner.run(
            start_date=config.full_start,
            end_date=config.full_end,
        )
        elapsed = time.time() - t0

        m = result.metrics if hasattr(result, "metrics") else {}
        summary: Dict[str, Any] = {
            "tag": tag or f"{blend_method}_w{cta_hybrid_weight}",
            "blend_method": blend_method,
            "cta_hybrid_weight": cta_hybrid_weight,
            "xs_position_base": xs_position_base,
            "xs_position_ceiling": xs_position_ceiling,
            "xs_opposite_penalty": xs_opposite_penalty,
            "start_date": config.full_start,
            "end_date": config.full_end,
            "status": "ok",
            "elapsed_sec": round(elapsed, 1),
        }
        for key in [
            "sharpe",
            "total_return_pct",
            "max_drawdown_pct",
            "calmar",
            "win_rate",
            "profit_factor",
            "trade_count",
            "total_pnl",
        ]:
            if key in m:
                try:
                    summary[key] = float(m[key]) if m[key] is not None else None
                except (TypeError, ValueError):
                    summary[key] = m[key]

        logger.info(
            f"=== {summary['tag']}  return%={summary.get('total_return_pct')}  "
            f"sharpe={summary.get('sharpe')}  mdd={summary.get('max_drawdown_pct')}  "
            f"elapsed={elapsed:.1f}s ==="
        )
        return summary

    except Exception as e:
        import traceback

        elapsed = time.time() - t0
        logger.error(f"=== {tag} 失败: {e} ===\n{traceback.format_exc()}")
        return {
            "tag": tag,
            "blend_method": blend_method,
            "status": "fail",
            "error": str(e),
            "elapsed_sec": round(elapsed, 1),
        }


def _render_md(summaries: List[Dict[str, Any]]) -> str:
    lines = [
        "# 方向二 — 横截面动态仓位缩放 sweep",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "公式：`final = cta * (base + (ceiling-base)*|z|)`；",
        "异号时 `*=` opposite_penalty。",
        "",
        "## 1. 线性基线（blend=linear）",
        "",
        "| tag | sharpe | return% | mdd% | calmar | 状态 |",
        "|-----|--------|---------|------|--------|------|",
    ]
    lin = [s for s in summaries if s.get("blend_method") == "linear"]
    if not lin:
        lines.append("| （未跑） | — | — | — | — | — |")
    else:
        for s in lin:
            lines.append(
                f"| {s['tag']} | {_fmt(s.get('sharpe'), 3)} | "
                f"{_fmt_pct(s.get('total_return_pct'))} | "
                f"{_fmt_pct(s.get('max_drawdown_pct'))} | "
                f"{_fmt(s.get('calmar'), 3)} | {s.get('status')} |"
            )

    lines.extend(
        [
            "",
            "## 2. 动态混合 sweep（blend=dynamic）",
            "",
            "行=base，列=penalty。单元格：`return% / sharpe / mdd%`",
            "",
        ]
    )
    dyn = [s for s in summaries if s.get("blend_method") == "dynamic"]
    if not dyn:
        lines.append("（无运行）")
    else:
        bases = sorted({s["xs_position_base"] for s in dyn})
        penalties = sorted({s["xs_opposite_penalty"] for s in dyn})
        header = (
            "| base \\\\ penalty | " + " | ".join(f"p={p:g}" for p in penalties) + " |"
        )
        sep = "|---" * (len(penalties) + 1) + "|"
        lines.extend([header, sep])
        for b in bases:
            row = [f"base={b:g}"]
            for p in penalties:
                cell = next(
                    (
                        s
                        for s in dyn
                        if abs(s["xs_position_base"] - b) < 1e-9
                        and abs(s["xs_opposite_penalty"] - p) < 1e-9
                    ),
                    None,
                )
                if cell is None or cell.get("status") != "ok":
                    row.append("—")
                else:
                    row.append(
                        f"{_fmt_pct(cell.get('total_return_pct'))} / "
                        f"{_fmt(cell.get('sharpe'), 2)} / "
                        f"{_fmt_pct(cell.get('max_drawdown_pct'))}"
                    )
            lines.append("| " + " | ".join(row) + " |")

    # 关键对比
    lines.extend(["", "## 3. 关键对比", ""])
    ok_dyn = [s for s in dyn if s.get("status") == "ok"]
    ok_lin = [s for s in lin if s.get("status") == "ok"]
    if ok_lin and ok_dyn:
        best_lin = max(ok_lin, key=lambda s: s.get("total_return_pct") or -1e9)
        best_dyn = max(ok_dyn, key=lambda s: s.get("total_return_pct") or -1e9)
        lines.extend(
            [
                f"- 最佳线性基线: **{best_lin['tag']}** "
                f"(return%={_fmt_pct(best_lin.get('total_return_pct'))}, "
                f"sharpe={_fmt(best_lin.get('sharpe'), 3)}, "
                f"mdd%={_fmt_pct(best_lin.get('max_drawdown_pct'))})",
                f"- 最佳动态配置: **base={best_dyn['xs_position_base']}, "
                f"penalty={best_dyn['xs_opposite_penalty']}** "
                f"(return%={_fmt_pct(best_dyn.get('total_return_pct'))}, "
                f"sharpe={_fmt(best_dyn.get('sharpe'), 3)}, "
                f"mdd%={_fmt_pct(best_dyn.get('max_drawdown_pct'))})",
            ]
        )
        ret_diff = (best_dyn.get("total_return_pct") or 0) - (
            best_lin.get("total_return_pct") or 0
        )
        mdd_diff = (best_dyn.get("max_drawdown_pct") or 0) - (
            best_lin.get("max_drawdown_pct") or 0
        )
        lines.extend(
            [
                "",
                f"- 收益差 (dynamic − linear): **{ret_diff:+.2f} 个百分点**",
                f"- 回撤差 (dynamic − linear): **{mdd_diff:+.2f} 个百分点** "
                f"（负=减少回撤，正=放大回撤）",
            ]
        )
        if mdd_diff < 0 and ret_diff >= -2.0:
            lines.append(
                "\n**结论：动态模式减少了回撤，且收益损失 ≤ 2%，可考虑采纳。**"
            )
        elif ret_diff > 0:
            lines.append("\n**结论：动态模式同时改善了收益和回撤，强烈推荐采纳。**")
        else:
            lines.append("\n**结论：动态模式无明显优势，回退到线性基线。**")
    return "\n".join(lines)


def _fmt(v, prec: int = 2) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{prec}f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_pct(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}%"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if args.output_suffix:
        out_dir = out_dir.parent / f"{out_dir.name}_{args.output_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []

    # 1) 线性基线（可选）
    if args.include_linear:
        for w in [0.9, 0.95, 1.0]:
            summaries.append(
                _run_one(
                    args,
                    blend_method="linear",
                    cta_hybrid_weight=w,
                    tag=f"linear_w{w}",
                )
            )

    # 2) 动态混合网格
    for base in args.bases:
        for pen in args.penalties:
            summaries.append(
                _run_one(
                    args,
                    blend_method="dynamic",
                    cta_hybrid_weight=0.9,  # 动态模式中 weight 不影响结果
                    xs_position_base=base,
                    xs_position_ceiling=args.ceiling,
                    xs_opposite_penalty=pen,
                    tag=f"dynamic_b{base:g}_p{pen:g}",
                )
            )

    # 输出
    json_path = out_dir / "summaries.json"
    json_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_path = out_dir / "summary.md"
    md_path.write_text(_render_md(summaries), encoding="utf-8")

    print(f"\n✓ JSON: {json_path}")
    print(f"✓ Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
