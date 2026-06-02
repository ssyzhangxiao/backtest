#!/usr/bin/env python3
"""
验证薄壳入口。

委托 runner.Pipeline 编排器执行验证流程。
原 run_validation.py 保留兼容。

用法:
  python run_validate.py                           # 全部验证
  python run_validate.py --method monte_carlo      # 仅蒙特卡洛
  python run_validate.py --method train_test       # 仅训练/测试分割
  python run_validate.py --method factor_ic        # 仅因子IC稳定性
"""

import argparse
import sys
from datetime import datetime

from loguru import logger


def main() -> None:
    """主执行入口：解析参数 → Pipeline 验证调用。"""
    parser = argparse.ArgumentParser(description="策略验证（Pipeline版）")
    parser.add_argument(
        "--config", default="config.yaml",
        help="配置文件路径（默认: config.yaml）",
    )
    parser.add_argument(
        "--method", default="all",
        help="验证方法: train_test, monte_carlo, bootstrap, factor_ic, all（默认: all）",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="验证完成后生成报告",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  策略验证 — Pipeline 版")
    print(f"  开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        from runner import Pipeline

        pipe = Pipeline(args.config).load_data()
        pipe.validate(args.method)

        if args.report:
            pipe.report("validation")

        logger.success("=" * 80)
        logger.success("验证完成")
        logger.success("=" * 80)

    except Exception as e:
        logger.error(f"验证流程失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
