"""
方向三：配对交易横截面信号实验

对比 3 种配置：
  - baseline: 方向二 dynamic (b=0.3, p=0.4) — 现有生产基线
  - v1:       dynamic + pair_trading 替换 XS 多因子
  - v2:       linear@0.9 + pair_trading 辅助信号

验收标准（OOS）：
  - Sharpe ≥ 0.04 (基线 0.046)
  - Calmar ≥ 2.5  (基线 2.82)
  - MDD ≤ 3.0%

决策：v1/v2 任一达标 → 切换；均不达标 → 保留方向二，方向三归档为研究分支。

用法：
    # 默认 OOS（2021-2024）
    python scripts/exp_pair_trading.py

    # 全期（2016-2024）
    python scripts/exp_pair_trading.py --start-date 2016-01-01 --end-date 2024-12-31

    # 自定义品种
    python scripts/exp_pair_trading.py --symbols SHFE.AL SHFE.CU DCE.M CZCE.FG
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
    parser = argparse.ArgumentParser(description="方向三：配对交易横截面实验")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-source", default="csv", choices=["csv", "tqsdk"])
    parser.add_argument(
        "--output-dir", default="output_backtest_pybroker/exp_pair_trading",
    )
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument(
        "--start-date", default="2021-01-01",
        help="默认 OOS：2021-01-01（基线）",
    )
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "trend",
            "term_structure",
            "mean_reversion",
            "vol_breakout",
            "composite_resonance",
        ],
    )
    parser.add_argument(
        "--acceptance-sharpe", type=float, default=0.04,
    )
    parser.add_argument(
        "--acceptance-calmar", type=float, default=2.5,
    )
    parser.add_argument(
        "--acceptance-mdd", type=float, default=3.0,
    )
    return parser.parse_args()


def _run_experiment(
    args: argparse.Namespace,
    *,
    blend_method: str,
    cta_hybrid_weight: float,
    xs_position_base: float,
    xs_opposite_penalty: float,
    pair_trading_enabled: bool,
    tag: str,
) -> Dict[str, Any]:
    """单个实验配置回测。"""
    from loguru import logger
    from core.config import BacktestConfig
    from core.execution.backtest_runner import PyBrokerBacktestRunner
    from core.engine.pybroker_data_source import PyBrokerDataSource

    t0 = time.time()
    try:
        overrides = {
            "backtest__cta_hybrid_weight": cta_hybrid_weight,
            "backtest__signal_mode": "hybrid",
            "backtest__hybrid_blend_method": blend_method,
            "backtest__xs_position_base": xs_position_base,
            "backtest__xs_opposite_penalty": xs_opposite_penalty,
            "backtest__pair_trading_enabled": pair_trading_enabled,
            "backtest__full_start_date": args.start_date,
            "backtest__full_end_date": args.end_date,
        }
        if args.symbols:
            overrides["symbols"] = args.symbols
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
            "date": "date", "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume", "open_interest": "open_interest",
            "symbol": "symbol",
        }
        df_std = df.rename(
            columns={k: v for k, v in col_map.items() if k in df.columns}
        )
        ds = PyBrokerDataSource(df=df_std)

        runner = PyBrokerBacktestRunner(
            data_source=ds,
            config=config,
            target_symbols=config.symbols,
            record_per_bar=False,
        )
        runner.register_strategies(args.strategies)
        result = runner.run(
            start_date=config.full_start,
            end_date=config.full_end,
        )
        elapsed = time.time() - t0

        m = result.metrics if hasattr(result, "metrics") else {}
        summary: Dict[str, Any] = {
            "tag": tag,
            "blend_method": blend_method,
            "cta_hybrid_weight": cta_hybrid_weight,
            "xs_position_base": xs_position_base,
            "xs_opposite_penalty": xs_opposite_penalty,
            "pair_trading_enabled": pair_trading_enabled,
            "start_date": config.full_start,
            "end_date": config.full_end,
            "n_symbols": len(config.symbols),
            "status": "ok",
            "elapsed_sec": round(elapsed, 1),
        }
        for key in [
            "sharpe", "total_return_pct", "max_drawdown_pct",
            "calmar", "win_rate", "profit_factor", "trade_count", "total_pnl",
        ]:
            if key in m:
                try:
                    summary[key] = float(m[key]) if m[key] is not None else None
                except (TypeError, ValueError):
                    summary[key] = m[key]

        logger.info(
            f"=== {tag}  return%={summary.get('total_return_pct')}  "
            f"sharpe={summary.get('sharpe')}  mdd={summary.get('max_drawdown_pct')}  "
            f"calmar={summary.get('calmar')}  elapsed={elapsed:.1f}s ==="
        )
        return summary

    except Exception as e:
        import traceback

        elapsed = time.time() - t0
        logger.error(f"=== {tag} 失败: {e} ===\n{traceback.format_exc()}")
        return {
            "tag": tag,
            "blend_method": blend_method,
            "pair_trading_enabled": pair_trading_enabled,
            "status": "fail",
            "error": str(e),
            "elapsed_sec": round(elapsed, 1),
        }


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_pct(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.2f}%"
    except (TypeError, ValueError):
        return str(x)


def _check_acceptance(
    s: Dict[str, Any],
    sharpe_th: float,
    calmar_th: float,
    mdd_th: float,
) -> Dict[str, bool]:
    return {
        "sharpe_ok": (s.get("sharpe") or 0) >= sharpe_th,
        "calmar_ok": (s.get("calmar") or 0) >= calmar_th,
        "mdd_ok": abs(s.get("max_drawdown_pct") or 0) <= mdd_th,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir / f"{args.start_date}_{args.end_date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        # 基线：方向二 dynamic b=0.3 p=0.4
        {
            "tag": "baseline_dyn_b0.3_p0.4",
            "blend_method": "dynamic",
            "cta_hybrid_weight": 0.5,
            "xs_position_base": 0.3,
            "xs_opposite_penalty": 0.4,
            "pair_trading_enabled": False,
        },
        # v1: 配对信号替换 XS 多因子
        {
            "tag": "v1_pair_replace_dyn",
            "blend_method": "dynamic",
            "cta_hybrid_weight": 0.5,
            "xs_position_base": 0.3,
            "xs_opposite_penalty": 0.4,
            "pair_trading_enabled": True,
        },
        # v2: 配对信号作为辅助（linear@0.9）
        {
            "tag": "v2_pair_aux_lin0.9",
            "blend_method": "linear",
            "cta_hybrid_weight": 0.9,
            "xs_position_base": 0.3,
            "xs_opposite_penalty": 0.4,
            "pair_trading_enabled": True,
        },
    ]

    print(f"\n=== 方向三：配对交易横截面实验 ({args.start_date} ~ {args.end_date}) ===\n")
    print(f"输出目录: {out_dir}\n")
    print(f"验收标准: Sharpe ≥ {args.acceptance_sharpe}, "
          f"Calmar ≥ {args.acceptance_calmar}, |MDD| ≤ {args.acceptance_mdd}%\n")

    summaries: List[Dict[str, Any]] = []
    for exp in experiments:
        s = _run_experiment(
            args,
            blend_method=exp["blend_method"],
            cta_hybrid_weight=exp["cta_hybrid_weight"],
            xs_position_base=exp["xs_position_base"],
            xs_opposite_penalty=exp["xs_opposite_penalty"],
            pair_trading_enabled=exp["pair_trading_enabled"],
            tag=exp["tag"],
        )
        summaries.append(s)

    # 输出 JSON
    json_path = out_dir / "summaries.json"
    json_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n→ 详细结果: {json_path}\n")

    # 输出对比表
    print("=" * 100)
    print("对比表")
    print("=" * 100)
    header = f"{'tag':<32} {'status':<6} {'ret%':>8} {'sharpe':>7} {'mdd%':>7} {'calmar':>7} {'SR✓':>4} {'CL✓':>4} {'DD✓':>4}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        if s.get("status") != "ok":
            print(f"{s['tag']:<32} {'FAIL':<6} {s.get('error', '')[:40]}")
            continue
        acc = _check_acceptance(
            s, args.acceptance_sharpe, args.acceptance_calmar, args.acceptance_mdd,
        )
        print(
            f"{s['tag']:<32} {s['status']:<6} "
            f"{_fmt_pct(s.get('total_return_pct')):>8} "
            f"{_fmt(s.get('sharpe'), 3):>7} "
            f"{_fmt_pct(s.get('max_drawdown_pct')):>7} "
            f"{_fmt(s.get('calmar'), 2):>7} "
            f"{'✓' if acc['sharpe_ok'] else '✗':>4} "
            f"{'✓' if acc['calmar_ok'] else '✗':>4} "
            f"{'✓' if acc['mdd_ok'] else '✗':>4}"
        )

    # 决策
    print("\n" + "=" * 100)
    print("决策")
    print("=" * 100)
    baseline = summaries[0]
    if baseline.get("status") != "ok":
        print("⚠ 基线失败，无法对比")
        return
    base_acc = _check_acceptance(
        baseline, args.acceptance_sharpe, args.acceptance_calmar, args.acceptance_mdd,
    )
    print(f"基线 acceptance: SR✓={base_acc['sharpe_ok']} CL✓={base_acc['calmar_ok']} DD✓={base_acc['mdd_ok']}")

    for s in summaries[1:]:
        if s.get("status") != "ok":
            print(f"  {s['tag']}: 失败 ({s.get('error', '')[:60]})")
            continue
        acc = _check_acceptance(
            s, args.acceptance_sharpe, args.acceptance_calmar, args.acceptance_mdd,
        )
        if all(acc.values()):
            print(
                f"  ✓ {s['tag']} 达标 → 切换 (ret={_fmt_pct(s.get('total_return_pct'))} "
                f"sharpe={_fmt(s.get('sharpe'), 3)} mdd={_fmt_pct(s.get('max_drawdown_pct'))} "
                f"calmar={_fmt(s.get('calmar'), 2)})"
            )
        else:
            fail = [k for k, v in acc.items() if not v]
            print(
                f"  ✗ {s['tag']} 不达标 ({', '.join(fail)})"
            )

    all_fail = all(
        s.get("status") != "ok" or not all(
            _check_acceptance(
                s, args.acceptance_sharpe, args.acceptance_calmar, args.acceptance_mdd,
            ).values()
        )
        for s in summaries[1:]
    )
    if all_fail:
        print("\n→ 所有 v1/v2 不达标 → 保留方向二 (b=0.3, p=0.4) 作为生产基线，方向三归档为研究分支")


if __name__ == "__main__":
    main()
