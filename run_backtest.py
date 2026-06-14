#!/usr/bin/env python3
"""
回测薄壳入口。

委托 runner.Pipeline 编排器执行完整回测流程。
支持多策略横截面打分模式和多策略集成模式。

用法:
  python run_backtest.py                     # 全部实验
  python run_backtest.py --experiment e1     # 单实验
  python run_backtest.py --optimize          # 先优化再回测
  python run_backtest.py --cross-sectional   # 多策略横截面打分模式
  python run_backtest.py --factor-review     # 先因子复核再回测
  python run_backtest.py --factor-screen     # 先筛选AlphaFutures因子再回测
  python run_backtest.py --report csv        # CSV格式报告
  python run_backtest.py --full              # 完整流程: 因子筛选+优化+回测+全部验证
"""

import argparse
import sys
from datetime import datetime

from loguru import logger


def _set_log_level(verbose: bool) -> None:
    """根据 --verbose 设置 loguru 输出级别，便于调试。"""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")


def main() -> None:
    """主执行入口：解析参数 → Pipeline 链式调用。"""
    parser = argparse.ArgumentParser(description="多策略量化回测系统（Pipeline版）")
    parser.add_argument(
        "--config", default="config.yaml",
        help="配置文件路径（默认: config.yaml）",
    )
    parser.add_argument(
        "--experiment", default="all",
        help="实验名称: e1~e11 或 all（默认: all）",
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="先运行参数优化，再执行回测",
    )
    parser.add_argument(
        "--cross-sectional", action="store_true",
        help="启用多策略横截面打分模式（5子策略: 趋势/期限结构/均值回归/波动率突破/复合共振）",
    )
    parser.add_argument(
        "--cta", action="store_true",
        help="启用 CTA 6 策略批量回测（carry/vol_mr/donchian/momentum/tsi_garch/pair_trading）",
    )
    parser.add_argument(
        "--strategy", default=None,
        help="单策略名称（trend/term_structure/mean_reversion/vol_breakout/composite_resonance/cross_sectional），"
             "默认根据--cross-sectional自动选择",
    )
    parser.add_argument(
        "--factor-review", action="store_true",
        help="先运行因子6项复核（存活率/缺失值/异常值抵抗/参数敏感性/正交性/时序稳定性），再执行回测",
    )
    parser.add_argument(
        "--factor-screen", action="store_true",
        help="先运行AlphaFutures因子筛选（IC/IR测试），再执行回测",
    )
    parser.add_argument(
        "--symbol", default=None,
        help="指定品种代码（如 SHFE.RB），仅优化/回测/因子筛选该品种",
    )
    parser.add_argument(
        "--validate", default=None,
        help="验证方法: train_test, monte_carlo, bootstrap, factor_ic, factor_alpha24, factor_review, all",
    )
    parser.add_argument(
        "--report", default="html",
        help="报告格式: html, csv（默认: html）",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="完整流程测试: 因子筛选 + 参数优化 + 回测 + 全部验证（不包含 --factor-review，避免重复）",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="启用 DEBUG 级别日志，便于排错（默认 INFO）",
    )

    args = parser.parse_args()

    # 应用日志级别
    _set_log_level(args.verbose)

    # 如果是 --full，自动设置其他参数（仅 factor_screen + optimize + validate，
    # 不包含 factor_review，避免 pipe.review_factors() 与 screen_factors() 重复调用）
    if args.full:
        args.factor_screen = True
        args.optimize = True
        args.validate = "all"

    print("=" * 80)
    print("  多策略量化回测系统 — Pipeline 版")
    if args.cross_sectional:
        print("  模式: 多策略横截面打分")
    if args.cta:
        print("  模式: CTA 6 策略批量回测")
    if args.full:
        print("  模式: 完整流程测试")
    print(f"  开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        from runner import Pipeline

        pipe = Pipeline(args.config).load_data()

        # P0-任务5整改：验证完整调用链是否就位
        chain_status = pipe.verify_chain()
        logger.info("=" * 60)
        logger.info("完整调用链验证 (P0-任务5):")
        for component, ready in chain_status.items():
            status = "✓" if ready else "✗"
            logger.info("  %s %s", status, component)
        logger.info("=" * 60)

        # 因子6项复核
        if args.factor_review:
            logger.info("执行因子6项复核...")
            pipe.review_factors()

        # AlphaFutures24 因子筛选
        if args.factor_screen:
            logger.info("执行AlphaFutures24因子筛选...")
            symbols = [args.symbol] if args.symbol else None
            pipe.screen_factors(symbols=symbols)

        # 参数优化
        if args.optimize:
            logger.info("执行参数优化...")
            strategy = args.strategy or ("cross_sectional" if args.cross_sectional else None)
            pipe.optimize(strategy=strategy, symbol=args.symbol)

        # CTA 批量回测
        if args.cta:
            pipe.run_cta()

        # 横截面实验回测
        if args.cross_sectional or not args.cta:
            pipe.run_backtest(
                args.experiment,
                cross_sectional=args.cross_sectional,
                strategy=args.strategy,
            )

        # 验证（CTA 模式跳过验证）
        if args.validate and not args.cta:
            best_params = None
            opt_results = pipe.results.get("optimization", {})
            if opt_results and "best_params" in opt_results:
                best_params = opt_results["best_params"]
            pipe.validate(args.validate, best_params=best_params, cross_sectional=args.cross_sectional)

        # 报告
        pipe.report(args.report)

        logger.success("=" * 80)
        logger.success("回测完成")
        logger.success("=" * 80)

    except Exception as e:
        logger.error(f"回测流程失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
