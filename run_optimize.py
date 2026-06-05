#!/usr/bin/env python3
"""
参数优化薄壳入口。

委托 runner.Pipeline 编排器执行参数优化流程。
原 run_parameter_optimization.py 保留兼容。

用法:
  python run_optimize.py                     # 全部策略优化
  python run_optimize.py --strategy ts_mom   # 单策略优化
  python run_optimize.py --skip-grid         # 跳过网格搜索
  python run_optimize.py --factor-screen     # 先筛选AlphaFutures24因子再优化
"""

import argparse
import sys
from datetime import datetime

from loguru import logger


def main() -> None:
    """主执行入口：解析参数 → Pipeline 优化调用。"""
    parser = argparse.ArgumentParser(description="参数优化（Pipeline版）")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="配置文件路径（默认: config.yaml）",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="指定策略名称，默认全部策略",
    )
    parser.add_argument(
        "--skip-grid",
        action="store_true",
        help="跳过网格搜索，仅执行窗口搜索和OOS选择",
    )
    parser.add_argument(
        "--factor-screen",
        action="store_true",
        help="先运行AlphaFutures24因子筛选（IC/IR测试），再执行优化",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  参数优化 — Pipeline 版")
    print(f"  开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        from runner import Pipeline

        pipe = Pipeline(args.config).load_data()

        # AlphaFutures24 因子筛选
        if args.factor_screen:
            logger.info("执行AlphaFutures24因子筛选...")
            pipe.screen_factors()

        # 构建优化任务列表
        tasks = ["window", "oos"] if args.skip_grid else ["grid", "window", "oos"]
        pipe.optimize(strategy=args.strategy, tasks=tasks)

        # 输出优化结果
        opt_results = pipe.results.get("optimization", {})
        best_params = opt_results.get("best_params", {})
        if best_params:
            logger.info("\n最优参数:")
            for sname, params in best_params.items():
                logger.info(f"  {sname}: {params}")
        else:
            logger.warning("无有效优化结果")

        logger.success("=" * 80)
        logger.success("参数优化完成")
        logger.success("=" * 80)

    except Exception as e:
        logger.error(f"优化流程失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
