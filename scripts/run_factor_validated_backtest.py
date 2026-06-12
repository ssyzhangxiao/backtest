"""
端到端回测：因子验证 → 横截面多因子综合打分（CSV 数据源）。

流程：
  1. 用 CSV 加载 6 品种数据（绕过 TqSdk 凭证）
  2. 5 段式因子验证（ADF/IC/PRF/EventStudy，可选跳过慢的 Review 段）
  3. 全量执行横截面多因子综合打分回测（E1~E11 实验）
  4. 生成报告

用法：
    python scripts/run_factor_validated_backtest.py                    # 完整流程
    python scripts/run_factor_validated_backtest.py --skip-review     # 跳过 review 段
    python scripts/run_factor_validated_backtest.py --validate-only   # 仅验证，不回测
    python scripts/run_factor_validated_backtest.py --backtest-only   # 仅回测，不验证

委托：runner.pipeline.Pipeline（规则 18）+ core.engine.switch_engine（横截面打分）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="因子验证 + 横截面多因子综合打分")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-source", default="csv", choices=["csv", "tqsdk"])
    parser.add_argument(
        "--skip-review", action="store_true", help="跳过慢的 Review 冗余段"
    )
    parser.add_argument("--validate-only", action="store_true", help="仅验证，不回测")
    parser.add_argument("--backtest-only", action="store_true", help="仅回测，不验证")
    parser.add_argument("--experiment", default="all", help="实验名（默认 all）")
    parser.add_argument("--no-report", action="store_true", help="跳过报告生成")
    return parser.parse_args()


def _build_csv_data_source(config, raw_config: dict):
    """构造 CSV PyBrokerDataSource（复用 run_factor5_validation 逻辑）。"""
    from loguru import logger
    from core.ext.adapters import create_data_source
    from core.engine.pybroker_data_source import PyBrokerDataSource

    data_cfg = raw_config.get("data", {}) if isinstance(raw_config, dict) else {}
    data_dir = data_cfg.get("csv_data_dir") or "./data"
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"CSV 数据目录不存在：{data_dir}")

    adapter = create_data_source("csv", data_dir=data_dir)
    loader = adapter._loader

    csv_paths = []
    for sym in config.symbols:
        candidates = [f"{sym}.csv", f"{sym.replace('.', '_')}.csv"]
        for fname in candidates:
            p = Path(data_dir) / fname
            if p.exists():
                csv_paths.append(str(p))
                break
    if not csv_paths:
        raise FileNotFoundError(f"未在 {data_dir} 找到 {config.symbols} 的 csv 文件")

    loader.load_csv_files_by_paths(csv_paths)
    df = loader.full_df
    if df is None or df.empty:
        raise RuntimeError("CSV 数据加载后为空")

    sym_counts = {
        s: int((df["symbol"] == s).sum())
        for s in config.symbols
        if "symbol" in df.columns
    }
    logger.info(f"  CSV 加载完成：symbol 行数={sym_counts}")

    return PyBrokerDataSource(df)


def _load_raw_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    args = parse_args()
    from loguru import logger
    from runner.pipeline import Pipeline
    from runner.backtest.experiments import run_experiment

    logger.info("=" * 70)
    logger.info("端到端回测：因子验证 → 横截面多因子综合打分")
    logger.info("=" * 70)
    logger.info(f"  config       = {args.config}")
    logger.info(f"  data_source  = {args.data_source}")
    logger.info(f"  validate     = {'YES' if not args.backtest_only else 'SKIP'}")
    logger.info(f"  backtest     = {'YES' if not args.validate_only else 'SKIP'}")
    logger.info(f"  skip_review  = {args.skip_review}")
    logger.info(f"  experiment   = {args.experiment}")

    raw_config = _load_raw_config(args.config)
    pipe = Pipeline(args.config)

    # ── 1. 数据加载 ──
    if args.data_source == "csv":
        ds = _build_csv_data_source(pipe._config, raw_config)
        actual_start, actual_end = ds.date_range
        pipe._data = ds
        pipe._raw_config = raw_config
        # 使用2020-2025窗口（因子信号在近期更有效）
        bt_start = "2020-01-01"
        bt_end = "2025-01-01"
        pipe = pipe.with_config(
            full_start=bt_start,
            full_end=bt_end,
            train_start=bt_start,
            train_end=bt_end,
            test_start=bt_start,
            test_end=bt_end,
        )
        logger.info(
            f"  CSV 数据日期: {actual_start} ~ {actual_end}, "
            f"回测窗口: {bt_start} ~ {bt_end}, "
            f"品种={len(pipe._config.symbols)}"
        )
    else:
        pipe.load_data()

    # ── 2. 因子验证（可选） ──
    if not args.backtest_only:
        pipe.validate(method="standard_report")
        val = pipe._results.get("validation", {})
        std = val.get("standard_report", {})
        logger.info(
            f"  5 段式验证: n_factors={std.get('n_factors')}, "
            f"pass_rate={std.get('fully_validated_rate')}"
        )

    # ── 3. 横截面多因子综合打分回测 ──
    if not args.validate_only:
        logger.info("=" * 70)
        logger.info("开始横截面多因子综合打分回测")
        logger.info("=" * 70)

        # run_experiment("all") 自动遍历 E1~E11，对每个品种×策略回测
        # cross_sectional=True 让 PyBrokerBacktestRunner 通过 FactorScoringEngine
        # 对 5 子策略信号做实时横截面标准化 + 排名叠加。
        results = run_experiment(
            name=args.experiment,
            config=pipe._config,  # BacktestConfig 实例
            data_source=ds if args.data_source == "csv" else pipe._data,
            raw_config=raw_config,
            cross_sectional=True,  # 横截面多因子综合打分
        )
        pipe._results["backtest"] = results

        sym_count = len(pipe._config.symbols)
        logger.info(
            f"  横截面打分完成：{args.experiment} 实验, "
            f"{sym_count} 品种, "
            f"results keys={list(results.keys())[:6] if isinstance(results, dict) else type(results).__name__}"
        )

    # ── 4. 报告 ──
    if not args.no_report:
        try:
            pipe.report(fmt="html")
            output_dir = Path(pipe._config.output_dir)
            logger.info(f"  报告已生成: {output_dir}")
        except Exception as e:
            logger.warning(f"  报告生成失败: {e}")

    logger.info("=" * 70)
    logger.success("端到端回测完成")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
