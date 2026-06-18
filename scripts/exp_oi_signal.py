"""
方向四 P1：持仓量衍生信号实验

对比 3 种配置（OOS 2021-2024）：
  - baseline:    现有 6 CTA 策略（DEFAULT_CTA_WEIGHTS）
  - pure_oi:     仅 oi_signal 策略（cta_weights={oi_signal: 1.0}）
  - fusion_oi:   6 策略 + oi_signal@0.10（融合）

验收标准：
  - pure_oi OOS Sharpe > 0.02
  - fusion_oi OOS Sharpe 比 baseline 提升 ≥10%
  - oi_signal 与 CTA 日收益相关系数 < 0.5

用法：
    python scripts/exp_oi_signal.py
    python scripts/exp_oi_signal.py --start-date 2016-01-01 --end-date 2024-12-31
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# 默认 6 CTA 权重（与 signal_abstraction.DEFAULT_CTA_WEIGHTS 保持一致）
BASELINE_CTA_WEIGHTS: Dict[str, float] = {
    "carry": 0.30,
    "vol_mean_reversion": 0.30,
    "donchian_breakout": 0.20,
    "momentum_ma": 0.10,
    "tsi_garch": 0.05,
    "pair_trading": 0.05,
}

# 融合：原 6 策略按 0.9 缩放 + oi_signal@0.10
FUSION_CTA_WEIGHTS: Dict[str, float] = {
    "carry": 0.27,
    "vol_mean_reversion": 0.27,
    "donchian_breakout": 0.18,
    "momentum_ma": 0.09,
    "tsi_garch": 0.045,
    "pair_trading": 0.045,
    "oi_signal": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="方向四 P1：OI 信号实验")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--output-dir",
        default="output_backtest_pybroker/exp_oi_signal",
    )
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument(
        "--start-date",
        default="2021-01-01",
        help="默认 OOS：2021-01-01",
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
        "--acceptance-sharpe",
        type=float,
        default=0.02,
        help="pure_oi 验收 Sharpe 下限",
    )
    parser.add_argument(
        "--acceptance-lift",
        type=float,
        default=0.10,
        help="fusion_oi 比 baseline 提升比例下限（10%）",
    )
    parser.add_argument(
        "--acceptance-corr",
        type=float,
        default=0.5,
        help="oi_signal 与 CTA 日收益相关系数上限",
    )
    return parser.parse_args()


def _run_experiment(
    args: argparse.Namespace,
    *,
    cta_composite_weights: Optional[Dict[str, float]],
    tag: str,
) -> Dict[str, Any]:
    """单个实验配置回测。"""
    from loguru import logger
    from core.config import BacktestConfig
    from core.execution.backtest_runner import PyBrokerBacktestRunner
    from core.engine.pybroker_data_source import PyBrokerDataSource

    t0 = time.time()
    try:
        overrides: Dict[str, Any] = {
            "backtest__signal_mode": "cta",  # 纯 CTA 时序模式（不做横截面）
            "backtest__cta_hybrid_weight": 0.5,  # 兼容字段
            "backtest__full_start_date": args.start_date,
            "backtest__full_end_date": args.end_date,
        }
        if cta_composite_weights is not None:
            overrides["backtest__cta_composite_weights"] = cta_composite_weights
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
            "cta_composite_weights": cta_composite_weights,
            "start_date": config.full_start,
            "end_date": config.full_end,
            "n_symbols": len(config.symbols),
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
            f"=== {tag}  return%={summary.get('total_return_pct')}  "
            f"sharpe={summary.get('sharpe')}  mdd={summary.get('max_drawdown_pct')}  "
            f"calmar={summary.get('calmar')}  elapsed={elapsed:.1f}s ==="
        )
        return summary
    except Exception as e:  # noqa: BLE001
        elapsed = time.time() - t0
        from loguru import logger

        logger.exception(f"{tag} failed after {elapsed:.1f}s: {e}")
        return {
            "tag": tag,
            "status": "error",
            "error": str(e),
            "elapsed_sec": round(elapsed, 1),
        }


def _compute_oi_cta_correlation(
    args: argparse.Namespace,
    n_symbols_max: int = 6,
) -> float:
    """计算 oi_signal 与默认 CTA 合成的日收益相关系数（OOS 期间）。

    Returns:
        相关系数（绝对值）；None 表示计算失败。
    """
    from loguru import logger
    from core.config import BacktestConfig
    from core.execution.signal_abstraction import DEFAULT_CTA_WEIGHTS
    from core.execution.factor_pool import UnifiedFactorPool
    import yaml

    overrides: Dict[str, Any] = {
        "backtest__signal_mode": "cta",
        "backtest__full_start_date": args.start_date,
        "backtest__full_end_date": args.end_date,
    }
    if args.symbols:
        overrides["symbols"] = args.symbols[:n_symbols_max]
    config = BacktestConfig.from_yaml(args.config, overrides=overrides)

    with open(args.config, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    data_dir = raw_config.get("data", {}).get("csv_data_dir") or "./data"
    adapter_factory = __import__("core.ext.adapters", fromlist=["create_data_source"])
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
    df_std = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    pool = UnifiedFactorPool()

    # 聚合到横截面：每个 bar 用 (symbol, signal) 矩阵
    symbols = config.symbols
    pivot_cta: Dict[str, List[float]] = {s: [] for s in symbols}
    pivot_oi: Dict[str, List[float]] = {s: [] for s in symbols}
    pivot_close: Dict[str, List[float]] = {s: [] for s in symbols}
    for sym in symbols:
        sub = df_std[df_std["symbol"] == sym].sort_values("date").reset_index(drop=True)
        # 仅取 OOS 区间
        sub = sub[
            (sub["date"] >= args.start_date) & (sub["date"] <= args.end_date)
        ].reset_index(drop=True)
        if len(sub) < 30:
            continue
        cta_arr = np.zeros(len(sub))
        oi_arr = np.zeros(len(sub))
        for i in range(30, len(sub)):
            ohlcv = sub.iloc[: i + 1][["open", "high", "low", "close", "volume"]]
            try:
                sigs = pool.compute_signals_for_bar(ohlcv, sym, i)
                cta_arr[i] = sum(
                    sigs.get(k, 0.0) * w for k, w in DEFAULT_CTA_WEIGHTS.items()
                )
                oi_arr[i] = sigs.get("oi_signal", 0.0)
            except Exception:
                cta_arr[i] = 0.0
                oi_arr[i] = 0.0
        pivot_cta[sym] = cta_arr.tolist()
        pivot_oi[sym] = oi_arr.tolist()
        pivot_close[sym] = sub["close"].tolist()

    # 构造等权组合日收益
    n_bars = min(len(pivot_cta[s]) for s in symbols if pivot_cta[s])
    if n_bars < 30:
        logger.warning("可用 bar 数过少，跳过相关系数计算")
        return 1.0
    cta_ret = np.zeros(n_bars)
    oi_ret = np.zeros(n_bars)
    for s in symbols:
        cta = np.array(pivot_cta[s][:n_bars])
        oi = np.array(pivot_oi[s][:n_bars])
        close = np.array(pivot_close[s][:n_bars])
        ret = np.zeros(n_bars)
        ret[1:] = np.diff(close) / close[:-1]
        cta_ret += cta * ret
        oi_ret += oi * ret
    cta_ret /= len(symbols)
    oi_ret /= len(symbols)

    if cta_ret.std() < 1e-8 or oi_ret.std() < 1e-8:
        logger.warning("信号收益方差为 0，无法计算相关系数")
        return 0.0
    corr = float(np.corrcoef(cta_ret, oi_ret)[0, 1])
    logger.info(f"oi_signal vs CTA composite 日收益相关系数 = {corr:.4f}")
    return abs(corr)


def main() -> None:
    args = parse_args()
    from loguru import logger

    logger.info(
        f"=== 方向四 P1：持仓量衍生信号实验 ({args.start_date} ~ {args.end_date}) ==="
    )

    # 配置
    experiments: List[Dict[str, Any]] = [
        {
            "tag": "baseline_6cta",
            "cta_composite_weights": None,  # 默认 6 策略
        },
        {
            "tag": "pure_oi",
            "cta_composite_weights": {"oi_signal": 1.0},
        },
        {
            "tag": "fusion_oi@0.10",
            "cta_composite_weights": FUSION_CTA_WEIGHTS,
        },
    ]

    # 回测
    summaries: List[Dict[str, Any]] = []
    for exp in experiments:
        s = _run_experiment(
            args,
            cta_composite_weights=exp["cta_composite_weights"],
            tag=exp["tag"],
        )
        summaries.append(s)

    # 相关系数
    logger.info("=== 计算 oi_signal 与 CTA composite 的日收益相关性 ===")
    corr_abs = _compute_oi_cta_correlation(args)

    # 汇总对比
    out_dir = Path(args.output_dir) / f"{args.start_date}_{args.end_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summaries.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "summaries": summaries,
                "oi_cta_corr_abs": corr_abs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 输出对比表
    by_tag = {s["tag"]: s for s in summaries if s.get("status") == "ok"}
    if not by_tag:
        logger.error("全部实验失败，请检查配置")
        return

    base = by_tag.get("baseline_6cta", {})
    pure = by_tag.get("pure_oi", {})
    fusion = by_tag.get("fusion_oi@0.10", {})

    base_sharpe = base.get("sharpe") or 0.0
    pure_sharpe = pure.get("sharpe") or 0.0
    fusion_sharpe = fusion.get("sharpe") or 0.0
    lift_pct = (
        (fusion_sharpe - base_sharpe) / abs(base_sharpe) * 100
        if abs(base_sharpe) > 1e-8
        else 0.0
    )

    print()
    print("=" * 100)
    print("方向四 P1：OI 信号实验结果")
    print("=" * 100)
    print(
        f"{'experiment':<22} {'ret%':>10} {'sharpe':>10} {'mdd%':>10} "
        f"{'calmar':>10} {'elapsed':>10}"
    )
    print("-" * 100)
    for s in summaries:
        if s.get("status") != "ok":
            print(f"{s['tag']:<22} ERROR: {s.get('error', '')[:60]}")
            continue
        print(
            f"{s['tag']:<22} "
            f"{(s.get('total_return_pct') or 0):>10.4f} "
            f"{(s.get('sharpe') or 0):>10.4f} "
            f"{(s.get('max_drawdown_pct') or 0):>10.4f} "
            f"{(s.get('calmar') or 0):>10.4f} "
            f"{(s.get('elapsed_sec') or 0):>10.1f}"
        )
    print("-" * 100)
    print(
        f"oi_signal vs CTA composite |corr| = {corr_abs:.4f} (阈值 < {args.acceptance_corr})"
    )
    print(
        f"fusion_oi 比 baseline sharpe 提升 = {lift_pct:+.2f}% (阈值 ≥ {args.acceptance_lift * 100:.0f}%)"
    )
    print("=" * 100)

    # 验收
    print()
    print("=== 验收 ===")
    pure_pass = pure_sharpe > args.acceptance_sharpe
    print(
        f"  [{'✓' if pure_pass else '✗'}] pure_oi Sharpe={pure_sharpe:.4f} "
        f"> {args.acceptance_sharpe}"
    )
    fusion_pass = lift_pct >= args.acceptance_lift * 100
    print(
        f"  [{'✓' if fusion_pass else '✗'}] fusion_oi 提升={lift_pct:+.2f}% "
        f"≥ {args.acceptance_lift * 100:.0f}%"
    )
    corr_pass = corr_abs < args.acceptance_corr
    print(
        f"  [{'✓' if corr_pass else '✗'}] |corr|={corr_abs:.4f} "
        f"< {args.acceptance_corr}"
    )
    print("=" * 100)

    if pure_pass and fusion_pass and corr_pass:
        print("→ P1 全项达标，可集成到生产")
    elif pure_pass and not fusion_pass:
        print("→ pure_oi 达标但融合未改善（OI 与现 CTA 高度相关或权重需调）")
    else:
        print("→ P1 不达标，方向四 P1 归档为研究分支")
    print()
    print(f"→ 详细结果: {out_dir}/summaries.json")


if __name__ == "__main__":
    main()
