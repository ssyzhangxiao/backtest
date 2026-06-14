"""
⚠️ 已废弃（2026-06-13）— 信号计算已统一到 UnifiedFactorPool。
保留作为参考，新功能请使用 core/execution/factor_pool.py。

单品种 CTA 回测入口脚本。

平行模式（Step 1）：为 2-3 个品种独立运行 CTA 策略，
与横截面结果做对比，验证 CTA 策略的有效性。

不修改现有系统一行代码，完全独立运行。

用法:
    # 简单趋势策略（默认，不依赖 arch 库）
    python scripts/run_cta_backtest.py --symbols SHFE.AU DCE.I --strategy simple_trend

    # 状态感知趋势策略（需 arch 库）
    python scripts/run_cta_backtest.py --symbols SHFE.AU INE.SC --strategy state_aware_trend

    # 全品种运行
    python scripts/run_cta_backtest.py --all

    # 指定配置
    python scripts/run_cta_backtest.py --strategy simple_trend \
        --fast-ma 10 --slow-ma 30 --entry-threshold 0.03
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# 自动加载 .env 文件（含 TQSDK_PHONE / TQSDK_PASSWORD）
load_dotenv()

# 确保项目根目录在 path 中
_RPOJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RPOJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_cta_backtest")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单品种 CTA 回测（平行模式，不改现有系统一行代码）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_cta_backtest.py --symbols SHFE.AU DCE.I --local
  python scripts/run_cta_backtest.py --symbols SHFE.AU DCE.I INE.SC --local
  python scripts/run_cta_backtest.py --symbols INE.SC --fast-ma 5 --slow-ma 20 --local
        """,
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="品种代码列表（如 SHFE.AU DCE.I），默认使用 config.yaml 中的顶层 symbols",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="使用数据源中所有可用品种",
    )
    parser.add_argument(
        "--strategy",
        default="momentum_ma",
        choices=[
            "momentum_ma", "tsi_garch", "donchian_breakout",
            "carry_zscore", "vol_mean_reversion",
            # 旧名别名（兼容）
            "simple_trend", "state_aware_trend", "carry",
        ],
        help="CTA 策略名（默认 momentum_ma，不依赖 arch）。"
             "旧名 simple_trend/state_aware_trend/carry 仍可用。",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="使用本地缓存数据（跳过 TqSdk），需 data_cache/ 中有缓存文件",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="配置文件路径（默认 config.yaml）",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="回测开始日期（如 2023-01-01），默认使用 config.yaml",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="回测结束日期（如 2024-12-31），默认使用 config.yaml",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=None,
        help="初始资金（默认使用 config.yaml）",
    )

    # 简单趋势策略参数
    parser.add_argument("--fast-ma", type=int, default=10, help="快线窗口（默认 10）")
    parser.add_argument("--slow-ma", type=int, default=30, help="慢线窗口（默认 30）")
    parser.add_argument(
        "--no-atr", action="store_true", help="不使用 ATR 缩放信号"
    )

    # 唐奇安通道参数
    parser.add_argument(
        "--entry-lookback", type=int, default=20, help="通道周期（默认 20）"
    )
    parser.add_argument(
        "--atr-stop", type=float, default=3.0, help="止损 ATR 倍数（默认 3.0）"
    )

    # 通用参数
    parser.add_argument(
        "--entry-threshold", type=float, default=0.05, help="入场阈值（默认 0.05）"
    )
    parser.add_argument(
        "--max-position", type=float, default=0.3, help="单品种最大仓位（默认 0.3）"
    )

    return parser.parse_args()


def load_data(
    config_path: str,
    target_symbols: List[str] = None,
    use_local: bool = False,
) -> Any:
    """加载回测数据。

    Args:
        config_path: 配置文件路径
        target_symbols: 目标品种列表
        use_local: 是否使用本地缓存数据（跳过 TqSdk）
    """
    import pickle
    from pathlib import Path

    from core.config import BacktestConfig
    from core.engine.pybroker_data_source import PyBrokerDataSource, create_hybrid_data_source

    config = BacktestConfig.from_yaml(config_path)
    symbols = target_symbols or config.symbols

    logger.info(
        "配置加载完成: %s | 目标品种=%s | 日期=%s~%s",
        config_path,
        symbols[:5],
        config.full_start,
        config.test_start,
    )

    if use_local:
        # 从 data_cache/ 直接加载缓存 pkl，构建连续主力合约
        cache_dir = Path(__file__).resolve().parent.parent / "data_cache"
        rows_all = []
        for sym in symbols:
            sanitized = sym.replace(".", "_").replace("-", "_")
            candidates = sorted(
                cache_dir.glob(f"{sanitized}_*.pkl"),
                key=lambda p: p.stat().st_size,
                reverse=True,
            )
            if not candidates:
                logger.warning("本地无缓存: %s", sym)
                continue
            with open(candidates[0], "rb") as f:
                df = pickle.load(f)
            # 按日期分组，用 open_interest 最大的合约作为主力
            df["product_code"] = sym
            dominant_rows = df.loc[
                df.groupby(df["date"].dt.date)["open_interest"].idxmax()
            ].copy()
            dominant_rows["symbol"] = sym
            rows_all.append(dominant_rows)
            logger.info(
                "%s: 加载缓存 %s, 主力合约 %d 行",
                sym, candidates[0].name, len(dominant_rows),
            )

        if not rows_all:
            raise RuntimeError(f"本地缓存无可用数据 (品种={symbols})")
        combined = pd.concat(rows_all, ignore_index=True)
        combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
        ds = PyBrokerDataSource(combined)
        logger.info("本地缓存加载完成: %d 行, %d 品种", len(ds), len(ds.symbols))
    else:
        ds = create_hybrid_data_source(symbols=symbols)

    return config, ds


def run_single_symbol_cta(
    symbol: str,
    config: Any,
    data_source: Any,
    strategy_name: str,
    strategy_params: Dict,
    entry_threshold: float,
    max_position_pct: float,
) -> Optional[Any]:
    """对单个品种运行 CTA 回测。

    Args:
        symbol: 品种代码
        config: BacktestConfig
        data_source: 数据源
        strategy_name: CTA 策略名
        strategy_params: 策略参数字典
        entry_threshold: 入场阈值
        max_position_pct: 最大仓位

    Returns:
        PyBroker 回测结果，失败返回 None
    """
    try:
        import pybroker
        from pybroker import StrategyConfig
    except ImportError:
        logger.error("PyBroker 未安装")
        return None

    from core.strategies.cta.registry import get_cta_strategy
    from core.engine.cta_executor_builder import CTAExecutorBuilder

    # 获取品种数据（只保留目标品种行）
    try:
        symbol_data = data_source.for_symbol(symbol)
    except ValueError as e:
        logger.warning("跳过 %s: %s", symbol, e)
        return None

    df = symbol_data.to_pybroker_df()
    if len(df) < 100:
        logger.warning("跳过 %s: 数据不足 (%d 行)", symbol, len(df))
        return None

    # 构建 CTA 策略和执行器
    cta = get_cta_strategy(strategy_name, strategy_params)
    builder = CTAExecutorBuilder(
        cta_strategy=cta,
        entry_threshold=entry_threshold,
        max_position_pct=max_position_pct,
    )
    executor_fn = builder.build()

    # 创建 PyBroker 策略（DataFrame 作为第一个参数）
    pb_config = StrategyConfig(initial_cash=config.initial_cash)
    strategy = pybroker.Strategy(df, config.full_start, config.test_start, config=pb_config)
    strategy.add_execution(executor_fn, symbols=[symbol])

    # 运行回测
    logger.info("运行 %s CTA 回测: %s", strategy_name, symbol)
    result = strategy.backtest(warmup=30)

    return result


def extract_metrics(result: Any, symbol: str, strategy_name: str) -> Dict:
    """从 PyBroker 结果中提取关键指标。"""
    if result is None:
        return {}

    metrics = {
        "symbol": symbol,
        "strategy": strategy_name,
        "total_return_pct": 0.0,
        "annual_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "total_trades": 0,
        "win_rate_pct": 0.0,
    }

    if hasattr(result, "metrics_df") and result.metrics_df is not None and "name" in result.metrics_df.columns:
        # metrics_df 是 name/value 格式
        m = result.metrics_df.set_index("name")["value"]

        for key, map_key in [
            ("total_return_pct", "total_return_pct"),
            ("max_drawdown_pct", "max_drawdown_pct"),
            ("sharpe_ratio", "sharpe"),
            ("annual_return_pct", None),
        ]:
            val = m.get(map_key or key)
            if val is not None:
                try:
                    metrics[key] = round(float(val), 2)
                except (ValueError, TypeError):
                    pass

        trade_count = m.get("trade_count")
        if trade_count is not None:
            try:
                metrics["total_trades"] = int(trade_count)
            except (ValueError, TypeError):
                pass

    if result.trades is not None and not result.trades.empty:
        if "pnl" in result.trades.columns:
            pnl = result.trades["pnl"]
            win = (pnl > 0).sum()
            metrics["total_trades"] = len(pnl)
            metrics["win_rate_pct"] = round(float(win) / len(pnl) * 100, 1)

    return metrics


def print_results_table(all_metrics: List[Dict]) -> None:
    """打印结果对比表。"""
    if not all_metrics:
        print("\n⚠️  无有效回测结果")
        return

    df = pd.DataFrame(all_metrics)
    print("\n" + "=" * 80)
    print("  CTA 单品种回测结果")
    print("=" * 80)

    # 选择关键列
    display_cols = [
        "symbol", "total_return_pct", "annual_return_pct",
        "sharpe_ratio", "max_drawdown_pct", "total_trades", "win_rate_pct",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    display_df = df[display_cols].copy()

    # 格式化百分比列
    for col in ["total_return_pct", "annual_return_pct", "max_drawdown_pct", "win_rate_pct"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%")

    print(display_df.to_string(index=False))
    print("=" * 80)

    # 汇总统计
    if "sharpe_ratio" in df.columns:
        avg_sharpe = df["sharpe_ratio"].mean()
        print(f"\n平均 Sharpe: {avg_sharpe:.2f}")
    if "total_return_pct" in df.columns:
        avg_return = df["total_return_pct"].mean()
        print(f"平均总收益: {avg_return:.1f}%")
        win_symbols = (df["total_return_pct"] > 0).sum()
        print(f"盈利品种: {win_symbols}/{len(df)}")


def main() -> None:
    """主入口。"""
    args = _parse_args()

    # 1. 先确定品种列表（从 args，数据源加载前确定）
    if args.symbols:
        symbols = args.symbols
    elif args.all:
        # 需要加载数据后才能获取全品种列表
        symbols = None  # 在 load_data 后补
    else:
        # 默认取 config.yaml 中的前 3 个
        config_pre = __import__("core.config", fromlist=["BacktestConfig"]).BacktestConfig.from_yaml(args.config)
        symbols = config_pre.symbols[:3]

    # 2. 加载数据（传入目标品种）
    config, data_source = load_data(args.config, target_symbols=symbols, use_local=args.local)

    # 2.2 覆盖日期范围（如果通过 CLI 指定）
    if args.start_date:
        config.full_start = args.start_date
    if args.end_date:
        config.test_start = args.end_date

    # 2.5 如果 --all，用数据源的全部品种
    if args.all:
        symbols = data_source.symbols

    if not symbols:
        logger.error("无有效品种，退出")
        sys.exit(1)

    logger.info("CTA 回测品种: %s", symbols)

    # 3. 策略参数（新旧名称均支持）
    strategy_params = {}
    if args.strategy in ("simple_trend", "momentum_ma"):
        strategy_params = {
            "fast_ma": args.fast_ma,
            "slow_ma": args.slow_ma,
            "use_atr_scaling": not args.no_atr,
        }
    elif args.strategy in ("state_aware_trend", "tsi_garch"):
        strategy_params = {
            "target_annual_vol": 0.15,
        }
    elif args.strategy == "donchian_breakout":
        strategy_params = {
            "entry_lookback": args.entry_lookback,
        }
    elif args.strategy in ("carry", "carry_zscore"):
        strategy_params = {
            "lookback": 60,
            "entry_z": 0.5,
            "direction": "both",
        }
    elif args.strategy == "vol_mean_reversion":
        strategy_params = {
            "vol_window": 20,
            "lookback": 252,
            "entry_z": 1.5,
        }

    # 4. 逐个品种运行
    all_metrics: List[Dict] = []
    for symbol in symbols:
        result = run_single_symbol_cta(
            symbol=symbol,
            config=config,
            data_source=data_source,
            strategy_name=args.strategy,
            strategy_params=strategy_params,
            entry_threshold=args.entry_threshold,
            max_position_pct=args.max_position,
        )
        if result:
            metrics = extract_metrics(result, symbol, args.strategy)
            all_metrics.append(metrics)
            logger.info(
                "%s: 收益=%.1f%% Sharpe=%.2f 回撤=%.1f%% 交易=%d",
                symbol,
                metrics.get("total_return_pct", 0),
                metrics.get("sharpe_ratio", 0),
                metrics.get("max_drawdown_pct", 0),
                metrics.get("total_trades", 0),
            )

    # 5. 打印对比表
    print_results_table(all_metrics)

    logger.info("CTA 回测完成: %d/%d 品种成功", len(all_metrics), len(symbols))


if __name__ == "__main__":
    main()
