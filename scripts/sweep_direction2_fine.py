"""
方向二参数精细扫描 — 确认 b=0.3, p=0.4 局部最优
3×3 = 9 配置：base ∈ {0.25, 0.30, 0.35}, penalty ∈ {0.35, 0.40, 0.45}

用法：
    # 默认 OOS（2021-2024）
    python scripts/sweep_direction2_fine.py

    # 全期
    python scripts/sweep_direction2_fine.py --start-date 2016-01-01 --end-date 2024-12-31
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
    parser = argparse.ArgumentParser(description="方向二参数精细扫描（确认 b=0.3 p=0.4 局部最优）")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--bases", nargs="+", type=float, default=[0.25, 0.30, 0.35],
    )
    parser.add_argument(
        "--penalties", nargs="+", type=float, default=[0.35, 0.40, 0.45],
    )
    parser.add_argument(
        "--start-date", default="2021-01-01",
    )
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument(
        "--output-dir", default="output_backtest_pybroker/direction2_fine_sweep",
    )
    parser.add_argument("--data-source", default="csv", choices=["csv", "tqsdk"])
    parser.add_argument("--strategies", nargs="+", default=None)
    return parser.parse_args()


def _run_one(
    args: argparse.Namespace,
    *,
    xs_position_base: float,
    xs_opposite_penalty: float,
    tag: str,
) -> Dict[str, Any]:
    from loguru import logger
    from core.config import BacktestConfig
    from core.execution.backtest_runner import PyBrokerBacktestRunner
    from core.engine.pybroker_data_source import PyBrokerDataSource

    t0 = time.time()
    try:
        overrides = {
            "backtest__cta_hybrid_weight": 0.5,
            "backtest__signal_mode": "hybrid",
            "backtest__hybrid_blend_method": "dynamic",
            "backtest__xs_position_base": xs_position_base,
            "backtest__xs_position_ceiling": 1.0,
            "backtest__xs_opposite_penalty": xs_opposite_penalty,
            "backtest__pair_trading_enabled": False,
            "backtest__full_start_date": args.start_date,
            "backtest__full_end_date": args.end_date,
        }
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
            data_source=ds, config=config,
            target_symbols=config.symbols, record_per_bar=False,
        )
        strategies = args.strategies or [
            "trend", "term_structure", "mean_reversion",
            "vol_breakout", "composite_resonance",
        ]
        runner.register_strategies(strategies)
        result = runner.run(
            start_date=config.full_start, end_date=config.full_end,
        )
        elapsed = time.time() - t0

        m = result.metrics if hasattr(result, "metrics") else {}
        s: Dict[str, Any] = {
            "tag": tag,
            "xs_position_base": xs_position_base,
            "xs_opposite_penalty": xs_opposite_penalty,
            "start_date": config.full_start, "end_date": config.full_end,
            "status": "ok", "elapsed_sec": round(elapsed, 1),
        }
        for k in [
            "sharpe", "total_return_pct", "max_drawdown_pct",
            "calmar", "win_rate", "profit_factor", "trade_count", "total_pnl",
        ]:
            if k in m:
                try:
                    s[k] = float(m[k]) if m[k] is not None else None
                except (TypeError, ValueError):
                    s[k] = m[k]
        logger.info(
            f"=== {tag}  ret%={s.get('total_return_pct')}  "
            f"sharpe={s.get('sharpe')}  mdd={s.get('max_drawdown_pct')}  "
            f"calmar={s.get('calmar')}  elapsed={elapsed:.1f}s ==="
        )
        return s
    except Exception as e:
        import traceback
        elapsed = time.time() - t0
        logger.error(f"=== {tag} 失败: {e} ===\n{traceback.format_exc()}")
        return {
            "tag": tag, "xs_position_base": xs_position_base,
            "xs_opposite_penalty": xs_opposite_penalty,
            "status": "fail", "error": str(e), "elapsed_sec": round(elapsed, 1),
        }


def _fmt(x, nd=3):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_pct(x):
    if x is None:
        return "—"
    try:
        return f"{float(x):.2f}%"
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / f"{args.start_date}_{args.end_date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== 方向二参数精细扫描 ({args.start_date} ~ {args.end_date}) ===")
    print(f"base × penalty = {len(args.bases)} × {len(args.penalties)} = "
          f"{len(args.bases) * len(args.penalties)} 配置")
    print(f"输出目录: {out_dir}\n")

    summaries: List[Dict[str, Any]] = []
    for b in args.bases:
        for p in args.penalties:
            tag = f"d_b{b}_p{p}"
            s = _run_one(args, xs_position_base=b, xs_opposite_penalty=p, tag=tag)
            summaries.append(s)

    json_path = out_dir / "summaries.json"
    json_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # 表格：行=base, 列=penalty, 单元格=calmar / sharpe / mdd%
    print("\n" + "=" * 100)
    print("Calmar 矩阵 (行=base, 列=penalty)")
    print("=" * 100)
    header = f"{'base':<8} | " + " | ".join(f"p={p:g}" for p in args.penalties)
    print(header)
    print("-" * len(header))
    for b in args.bases:
        row = [f"b={b:g}"]
        for p in args.penalties:
            cell = next(
                (s for s in summaries
                 if abs(s["xs_position_base"] - b) < 1e-9
                 and abs(s["xs_opposite_penalty"] - p) < 1e-9),
                None,
            )
            if cell is None or cell.get("status") != "ok":
                row.append("—")
            else:
                row.append(
                    f"{_fmt(cell.get('calmar'), 2)} / {_fmt(cell.get('sharpe'), 3)} / {_fmt_pct(cell.get('max_drawdown_pct'))}"
                )
        print(" | ".join(row))

    # 找 Calmar 最大
    ok = [s for s in summaries if s.get("status") == "ok"]
    if ok:
        best_calmar = max(ok, key=lambda s: s.get("calmar") or -1e9)
        best_sharpe = max(ok, key=lambda s: s.get("sharpe") or -1e9)
        print(f"\n→ 最高 Calmar: {best_calmar['tag']} = {best_calmar.get('calmar')}")
        print(f"→ 最高 Sharpe: {best_sharpe['tag']} = {best_sharpe.get('sharpe')}")

        # 找 b=0.3, p=0.4
        target = next(
            (s for s in ok
             if abs(s["xs_position_base"] - 0.30) < 1e-6
             and abs(s["xs_opposite_penalty"] - 0.40) < 1e-6),
            None,
        )
        if target:
            print(f"→ b=0.3,p=0.4  Calmar={target.get('calmar')}  "
                  f"Sharpe={target.get('sharpe')}  MDD={target.get('max_drawdown_pct')}")
            if (target.get("calmar") or -1e9) >= (best_calmar.get("calmar") or -1e9) - 0.1:
                print("  ✓ b=0.3,p=0.4 处于 Calmar 帕累托前沿（与最优相差 < 0.1）")
            else:
                print("  ⚠ b=0.3,p=0.4 不是 Calmar 局部最优，建议切到更优配置")
    print(f"\n→ 详细结果: {json_path}")


if __name__ == "__main__":
    main()
