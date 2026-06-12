"""
5 段式因子验证端到端回测脚本（2026-06-12）。

用法：
    python scripts/run_factor5_validation.py [--config config.yaml] [--no-report]
                                            [--data-source csv|tqsdk]

执行：
  1) load_data()                  — 加载 6 品种数据（默认走 CSV 避免 TqSdk 凭证缺失）
  2) validate("standard_report")  — 跑 5 段式因子验证（ADF/IC/PRF/EventStudy/Redundancy）
  3) report("html")               — 生成主报告 + 5 段式 HTML 片段 + 3 个 PNG

委托：runner.pipeline.Pipeline（规则 18 + 规则 17）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5 段式因子验证回测")
    parser.add_argument(
        "--config", default="config.yaml", help="配置文件路径（默认 config.yaml）"
    )
    parser.add_argument(
        "--no-report", action="store_true", help="跳过 HTML 报告生成（仅产出 csv）"
    )
    parser.add_argument(
        "--validate-method",
        default="standard_report",
        choices=[
            "standard_report",  # 5 段式汇总（推荐）
            "factor_adf",
            "factor_prf",
            "event_study",
            "factor_review",  # 含 Spearman 冗余
            "factor_alpha24",  # IC 段
        ],
        help="验证方法（默认 standard_report）",
    )
    parser.add_argument(
        "--data-source",
        default="csv",
        choices=["csv", "tqsdk"],
        help="数据源（默认 csv，避免 TqSdk 凭证依赖）",
    )
    return parser.parse_args()


def _build_pybroker_data_source(config, raw_config: dict):
    """
    构造 PyBroker 兼容数据源（绕过 TqSdk 凭证检查，使用本地 CSV）。

    内部委托 ext/adapters 工厂加载 DataLoader，再从 loader 取 PyBroker 格式
    DataFrame 喂给 PyBrokerDataSource（提供 query() 接口给验证器使用）。

    委托链：core.ext.adapters.create_data_source("csv", ...) → DataLoader
            → get_pybroker_df() → PyBrokerDataSource（规则 17 + 21）。
    """
    from loguru import logger
    from core.ext.adapters import create_data_source
    from core.engine.pybroker_data_source import PyBrokerDataSource

    data_cfg = raw_config.get("data", {}) if isinstance(raw_config, dict) else {}
    data_dir = data_cfg.get("csv_data_dir") or "./data"
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"CSV 数据目录不存在：{data_dir}")

    adapter = create_data_source("csv", data_dir=data_dir)
    loader = adapter._loader  # noqa: SLF001 — 委托 ext/adapters 工厂
    # 加载 config.symbols 指定的 csv
    # 2026-06-12：优先匹配原始完整数据 SHFE.AL.csv（5156 行品种级），
    # 避免误匹配调试文件 SHFE_AL.csv（625 行合约级）。
    csv_paths = []
    for sym in config.symbols:
        candidates = [f"{sym}.csv", f"{sym.replace('.', '_')}.csv"]
        for fname in candidates:
            p = Path(data_dir) / fname
            if p.exists():
                csv_paths.append(str(p))
                break
    if not csv_paths:
        raise FileNotFoundError(
            f"未在 {data_dir} 找到 {config.symbols} 任何 csv（尝试命名 SHFE.AL.csv / SHFE_AL.csv）"
        )
    # 加载前先按文件路径记住 → config symbol 的映射（用于 symbol 大写化）
    csv_to_config_sym: dict = {}
    for sym, p in zip(config.symbols, csv_paths):
        csv_to_config_sym[Path(p).stem] = sym  # SHFE_AL.csv → SHFE.AL

    loader.load_csv_files_by_paths(csv_paths)
    # 2026-06-12：CSV 加载后 full_df 里 symbol 列已是品种级（SHFE.AL/CZCE.FG 等），
    # 验证器 ds.query(..., symbols=['SHFE.AL']) 直接精确匹配。无需再做映射/聚合。
    df = loader.full_df
    if df is None or df.empty:
        raise RuntimeError("CSV 数据加载后为空，请检查 data_dir 下的 csv 文件格式")

    # 调试：每个品种实际可用行数
    sym_counts: dict = {}
    for s in config.symbols:
        n = (df["symbol"] == s).sum() if "symbol" in df.columns else 0
        sym_counts[s] = int(n)
    _dbg = f"  CSV 聚合后 symbol 行数={sym_counts}"
    try:
        logger.info(_dbg)
    except Exception:
        print(_dbg)

    return PyBrokerDataSource(df)


def _load_raw_config(path: str) -> dict:
    """从 yaml 加载完整 raw_config（包含 data 段）。"""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    args = parse_args()

    # 延迟导入：避免脚本被 import 时副作用
    from loguru import logger
    from runner.pipeline import Pipeline

    logger.info("=" * 70)
    logger.info("5 段式因子验证回测")
    logger.info("=" * 70)
    logger.info(f"  config       = {args.config}")
    logger.info(f"  method       = {args.validate_method}")
    logger.info(f"  data_source  = {args.data_source}")
    logger.info(f"  no_report    = {args.no_report}")

    # 1) 加载数据（CSV 路径：不依赖 TqSdk 凭证）
    raw_config = _load_raw_config(args.config)
    pipe = Pipeline(args.config)
    if args.data_source == "csv":
        ds = _build_pybroker_data_source(pipe._config, raw_config)
        # 2026-06-12：按 CSV 实际日期范围缩窄 train/test，避免短品种被
        # full_start=2020/full_end=2025 过滤后剩 < 100 行。
        actual_start, actual_end = ds.date_range
        pipe._data = ds
        pipe._raw_config = raw_config
        # 用 with_config 热更新（规则 18 链式）
        pipe = pipe.with_config(
            full_start=actual_start,
            full_end=actual_end,
            train_start=actual_start,
            train_end=actual_start,
            test_start=actual_start,
            test_end=actual_end,
        )
        # 调试：检查实际 query 出的行数
        from loguru import logger as _lg

        for sym in pipe._config.symbols:
            q = ds.query(pipe._config.train_start, pipe._config.test_end, symbols=[sym])
            _lg.info(
                f"  [DEBUG] ds.query train={pipe._config.train_start} test={pipe._config.test_end} "
                f"sym={sym} → {len(q)} 行"
            )
        logger.info(
            f"  CSV 数据集已加载，品种数={len(pipe._config.symbols)}，"
            f"目录={raw_config.get('data', {}).get('csv_data_dir', './data')}，"
            f"实际日期={actual_start} ~ {actual_end}"
        )
    else:
        pipe.load_data()
        logger.info(f"  TqSdk 数据集已加载，品种数={len(pipe._config.symbols)}")

    # 2) 5 段式因子验证
    pipe.validate(method=args.validate_method)
    val = pipe._results.get("validation", {})
    if args.validate_method == "standard_report":
        std = val.get("standard_report", {})
        logger.info(
            f"  5 段式验证完成：n_factors={std.get('n_factors')}, "
            f"n_fully_validated={std.get('n_fully_validated')}, "
            f"fully_validated_rate={std.get('fully_validated_rate')}"
        )

    # 3) HTML 报告（含 5 段式片段 + 3 PNG）
    if not args.no_report:
        pipe.report(fmt="html")
        output_dir = Path(pipe._config.output_dir)
        logger.info(f"  报告已生成：{output_dir / 'backtest_report_full.html'}")
        fragment = output_dir / "factor_5_section_report.html"
        if fragment.exists():
            logger.info(f"  5 段式片段：{fragment}")
        for png in [
            "factor_prf.png",
            "event_study.png",
            "factor_redundancy_heatmap.png",
        ]:
            p = output_dir / png
            if p.exists():
                logger.info(f"  PNG 已生成：{p}")

    logger.info("=" * 70)
    logger.info("完成")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
