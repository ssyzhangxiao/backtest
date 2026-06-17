"""
5 点 weight 网格 sweep — 直接调 PyBrokerBacktestRunner 触发 hybrid 路径。

原因：E1/E2/E4/E5 实验在 per-symbol 循环里只跑 sub-strategy 等权打分，
不读 config.cta_hybrid_weight。要触发 weight 影响必须直接调主回测 runner。

用法：
    python scripts/sweep_cta_hybrid_weight.py
    python scripts/sweep_cta_hybrid_weight.py --weights 0.0 0.3 0.5 0.7 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CTA 混合权重网格搜索（直接调主 runner）"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.0, 0.3, 0.5, 0.7, 1.0],
        help="weight 网格",
    )
    parser.add_argument(
        "--signal-mode",
        default="hybrid",
        choices=["hybrid", "cross_sectional", "cta"],
        help="信号模式（默认 hybrid）",
    )
    parser.add_argument("--data-source", default="csv", choices=["csv", "tqsdk"])
    parser.add_argument(
        "--output-dir", default="output_backtest_pybroker/cta_weight_sweep"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="CTA 策略子集（默认全部）",
    )
    # ── 横截面质量参数（用于让横截面"沉默"或"放大"） ──
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=None,
        help="横截面入场阈值（提高→更少交易）；不传则保持 yaml 默认",
    )
    parser.add_argument(
        "--rebalance-days",
        type=int,
        default=None,
        help="调仓周期（天数）；不传则保持 yaml 默认",
    )
    parser.add_argument(
        "--min-position-pct",
        type=float,
        default=None,
        help="最小开仓比例；不传则保持 yaml 默认",
    )
    parser.add_argument(
        "--only-factors",
        nargs="+",
        default=None,
        help="仅启用这些子策略（如 term_structure composite_resonance）；其他归零",
    )
    return parser.parse_args()


def _run_one_weight(
    weight: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """跑单 weight：构造临时 overrides 配置 → 调 PyBrokerBacktestRunner.run()。"""
    from loguru import logger
    from core.config import BacktestConfig
    from core.execution.backtest_runner import PyBrokerBacktestRunner
    from core.engine.pybroker_data_source import PyBrokerDataSource

    t0 = time.time()
    try:
        # 每次都重新加载（支持 overrides）
        overrides = {
            "backtest__cta_hybrid_weight": weight,
            "backtest__signal_mode": args.signal_mode,
        }
        # 横截面质量参数
        if args.entry_threshold is not None:
            overrides["backtest__entry_threshold"] = args.entry_threshold
        if args.rebalance_days is not None:
            overrides["backtest__rebalance_freq"] = args.rebalance_days
        if args.min_position_pct is not None:
            overrides["backtest__min_position_pct"] = args.min_position_pct
        if args.only_factors is not None:
            # 仅保留指定子策略，其他归零
            all_strats = [
                "trend",
                "term_structure",
                "mean_reversion",
                "vol_breakout",
                "composite_resonance",
            ]
            overrides["factor_weights"] = {
                s: (1.0 / len(args.only_factors) if s in args.only_factors else 0.0)
                for s in all_strats
            }
        config = BacktestConfig.from_yaml(args.config, overrides=overrides)

        # 构造 PyBrokerBacktestRunner（runner 内部会从 data_source 读 OHLCV）
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
        # 标准化列名
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

        # 注册策略（5子策略 + register_strategies 触发）
        runner = PyBrokerBacktestRunner(
            data_source=ds, config=config, target_symbols=config.symbols
        )
        # CTA 模式下用 6 个 CTA 策略
        if args.signal_mode == "cta":
            strategies = args.strategies or [
                "carry",
                "vol_mean_reversion",
                "donchian_breakout",
                "momentum_ma",
                "tsi_garch",
                "pair_trading",
            ]
        else:
            strategies = args.strategies or [
                "trend",
                "term_structure",
                "mean_reversion",
                "vol_breakout",
                "composite_resonance",
            ]
        runner.register_strategies(strategies)

        # 跑
        result = runner.run(
            start_date=config.full_start,
            end_date=config.full_end,
        )
        elapsed = time.time() - t0

        # 提取指标（注意：PyBroker 用 total_return_pct 表示百分比，total_return 表示小数）
        m = result.metrics if hasattr(result, "metrics") else {}
        summary = {
            "weight": weight,
            "signal_mode": args.signal_mode,
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
        # 落盘完整 metrics
        summary["_raw_metrics"] = {k: str(v) for k, v in m.items()}

        logger.info(
            f"=== weight={weight}  status=ok  "
            f"return={summary.get('total_return_pct', summary.get('total_return'))}  "
            f"sharpe={summary.get('sharpe')}  "
            f"elapsed={elapsed:.1f}s ==="
        )
        return summary

    except Exception as e:
        import traceback

        elapsed = time.time() - t0
        logger.error(f"=== weight={weight} 失败: {e} ===\n{traceback.format_exc()}")
        return {
            "weight": weight,
            "status": "fail",
            "error": str(e),
            "elapsed_sec": round(elapsed, 1),
        }


def _render_markdown_table(summaries: List[Dict[str, Any]]) -> str:
    lines = [
        "# CTA Hybrid Weight Sweep 结果",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 主表",
        "",
        "| weight | total_return | total_return_% | sharpe | max_drawdown | 耗时(s) | 状态 |",
        "|--------|--------------|----------------|--------|--------------|---------|------|",
    ]
    for s in summaries:
        w = s.get("weight", "—")
        tr_abs = _fmt_float(s.get("total_return"))
        tr_pct = _fmt_pct(s.get("total_return_pct"))
        sh = _fmt_float(s.get("sharpe"), 3)
        mdd = _fmt_pct(s.get("max_drawdown"))
        el = s.get("elapsed_sec", "—")
        st = s.get("status", "?")
        lines.append(f"| {w} | {tr_abs} | {tr_pct} | {sh} | {mdd} | {el} | {st} |")

    lines.extend(["", "## 解读", ""])
    ok = [s for s in summaries if s.get("status") == "ok"]
    if not ok:
        lines.append("无成功运行的 weight。")
        return "\n".join(lines)

    # 按 total_return_pct > total_return > sharpe 找最佳
    def score(s):
        return (
            s.get("total_return_pct")
            or (s.get("total_return") or 0) * 100
            or s.get("sharpe")
            or -1e9
        )

    best = max(ok, key=score)
    worst = min(ok, key=score)
    lines.append(
        f"- 最高收益: **weight={best['weight']}** "
        f"(return%={_fmt_pct(best.get('total_return_pct'))}, "
        f"sharpe={_fmt_float(best.get('sharpe'), 3)})"
    )
    lines.append(
        f"- 最低收益: weight={worst['weight']} "
        f"(return%={_fmt_pct(worst.get('total_return_pct'))}, "
        f"sharpe={_fmt_float(worst.get('sharpe'), 3)})"
    )
    spread = abs(score(best) - score(worst))
    lines.append(
        f"- 极差: {spread:.3f}（越小说明 weight 影响越小 → 体系对该参数不敏感）"
    )
    return "\n".join(lines)


def _fmt_pct(v) -> str:
    # total_return_pct/max_drawdown_pct 已是百分比单位（如 16.17 表示 16.17%），
    # 仅追加 % 后缀，避免 .2% 格式化器再 ×100
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}%"
    except (TypeError, ValueError):
        return str(v)


def _fmt_float(v, digits=3) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"CTA Hybrid Weight Sweep — {len(args.weights)} 个值")
    print(f"signal_mode={args.signal_mode} | 数据源={args.data_source}")
    print(f"{'=' * 60}\n")

    summaries: List[Dict[str, Any]] = []
    print(f"开始跑 {len(args.weights)} 个 weight...")
    for w in args.weights:
        s = _run_one_weight(w, args)
        summaries.append(s)
        print(
            f"  weight={w:>4}  status={s.get('status')}  "
            f"return_pct={_fmt_pct(s.get('total_return_pct'))}  "
            f"sharpe={_fmt_float(s.get('sharpe'), 3)}  "
            f"elapsed={s.get('elapsed_sec')}s"
        )

    print(f"\n写入报告...")
    json_path = out_dir / "sweep_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False, default=str)
    print(f"  JSON: {json_path}")

    md = _render_markdown_table(summaries)
    md_path = out_dir / "sweep_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  MD  : {md_path}\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
