#!/usr/bin/env python3
"""
回测薄壳入口。

委托 runner.Pipeline 编排器执行完整回测流程。
原 run_full_backtest.py 保留兼容，新入口推荐使用 Pipeline。

用法:
  python run_backtest.py                     # 全部实验
  python run_backtest.py --experiment e1     # 单实验
  python run_backtest.py --optimize          # 先优化再回测
  python run_backtest.py --report csv        # CSV格式报告
"""

import argparse
import sys
from datetime import datetime

from loguru import logger


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
        "--symbol", default=None,
        help="指定品种代码（如 SHFE.RB），仅优化/回测该品种",
    )
    parser.add_argument(
        "--validate", default=None,
        help="验证方法: train_test, monte_carlo, bootstrap, factor_ic, all",
    )
    parser.add_argument(
        "--report", default="html",
        help="报告格式: html, csv（默认: html）",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  多策略量化回测系统 — Pipeline 版")
    print(f"  开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        from runner import Pipeline

        pipe = Pipeline(args.config).load_data()

        # 参数优化
        if args.optimize:
            logger.info("执行参数优化...")
            pipe.optimize(symbol=args.symbol)

        # 回测实验
        pipe.run_backtest(args.experiment)

        # 验证
        if args.validate:
            best_params = None
            opt_results = pipe.results.get("optimization", {})
            if opt_results and "best_params" in opt_results:
                best_params = opt_results["best_params"]
            pipe.validate(args.validate, best_params=best_params)

        # 报告
        pipe.report(args.report)

        logger.success("=" * 80)
        logger.success("回测完成")
        logger.success("=" * 80)

    except Exception as e:
        logger.error(f"回测流程失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
