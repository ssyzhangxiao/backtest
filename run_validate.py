#!/usr/bin/env python3
"""
验证薄壳入口。

委托 runner.Pipeline 编排器执行验证流程。
支持多策略横截面打分验证和因子复核验证。

用法:
  python run_validate.py                               # 全部验证
  python run_validate.py --method monte_carlo          # 仅蒙特卡洛
  python run_validate.py --method train_test           # 仅训练/测试分割
  python run_validate.py --method factor_ic            # 仅因子IC稳定性
  python run_validate.py --method factor_alpha24       # 仅AlphaFutures24因子IC/IR验证
  python run_validate.py --method factor_review        # 仅因子6项复核
  python run_validate.py --method cross_sectional      # 多策略横截面打分验证
  python run_validate.py --cross-sectional             # 启用多策略横截面打分模式
"""

import argparse
import sys
from datetime import datetime

from loguru import logger


def _set_log_level(verbose: bool) -> None:
    """根据 --verbose 设置 loguru 输出级别（与 run_backtest.py 一致）。"""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")


def main() -> None:
    """主执行入口：解析参数 → Pipeline 验证调用。"""
    parser = argparse.ArgumentParser(description="策略验证（Pipeline版）")
    parser.add_argument(
        "--config", default="config.yaml",
        help="配置文件路径（默认: config.yaml）",
    )
    parser.add_argument(
        "--method", default="all",
        help="验证方法: train_test, monte_carlo, bootstrap, factor_ic, "
             "factor_alpha24, factor_review, cross_sectional, all（默认: all）",
    )
    parser.add_argument(
        "--cross-sectional", action="store_true",
        help="启用多策略横截面打分模式验证",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="验证完成后生成报告",
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="先执行参数优化，获取 best_params 再验证",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="启用 DEBUG 级别日志，便于排错（默认 INFO）",
    )

    args = parser.parse_args()

    # 应用日志级别
    _set_log_level(args.verbose)

    print("=" * 80)
    print("  策略验证 — Pipeline 版")
    if args.cross_sectional:
        print("  模式: 多策略横截面打分")
    print(f"  开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        from runner import Pipeline

        pipe = Pipeline(args.config).load_data()
        
        best_params = None
        if args.optimize:
            logger.info("执行参数优化以获取 best_params...")
            strategy = "cross_sectional" if args.cross_sectional else None
            pipe.optimize(strategy=strategy)
            opt_results = pipe.results.get("optimization", {})
            if opt_results and "best_params" in opt_results:
                best_params = opt_results["best_params"]
                logger.info(f"获取到 best_params: {list(best_params.keys())}")
        
        pipe.validate(
            args.method,
            best_params=best_params,
            cross_sectional=args.cross_sectional,
        )

        if args.report:
            pipe.report("validation")

        logger.success("=" * 80)
        logger.success("验证完成")
        logger.success("=" * 80)

    except Exception as e:
        logger.error(f"验证流程失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
