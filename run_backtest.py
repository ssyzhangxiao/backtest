"""
完整回测运行脚本。

使用真实期货数据运行组合交易策略系统的完整回测流程。
"""
import logging
import sys

from core.engine.runner import BacktestRunner, BacktestConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # 配置回测参数
    config = BacktestConfig(
        initial_cash=1_000_000,
        in_sample_end="2024-06-30",
        strategy_weights={
            "dual_ma": 0.25,
            "rsi": 0.20,
            "vol_breakout": 0.25,
            "term_structure": 0.20,
            "spread": 0.10,
        },
        stop_loss_pct=0.05,
        max_position_pct=0.2,
    )

    # 选择几个代表性品种进行回测
    data_dir = "./data"

    # 使用螺纹钢(RB)作为主要测试品种
    # DataLoader会自动加载目录下所有CSV
    runner = BacktestRunner(data_dir, config)

    # 加载数据
    logger.info("加载数据...")
    df = runner.load_data("SHFE.RB.csv")

    if df.empty:
        logger.error("数据加载失败")
        sys.exit(1)

    logger.info(f"数据范围: {df['date'].min()} ~ {df['date'].max()}")
    logger.info(f"合约数量: {df['symbol'].nunique()}")

    # 运行回测
    logger.info("开始回测...")
    result = runner.run(
        strategies=["dual_ma", "rsi", "vol_breakout", "term_structure"],
        start_date="2023-01-01",
        end_date="2024-12-31",
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("组合交易策略系统 - 回测结果")
    print("=" * 60)

    print("\n--- 组合绩效 ---")
    for k, v in result.portfolio_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n--- 各策略绩效 ---")
    for name, sr in result.strategy_results.items():
        print(f"\n  [{name}]")
        print(f"    总收益率: {sr.metrics.get('total_return_pct', 0):.2f}%")
        print(f"    年化收益率: {sr.metrics.get('annual_return_pct', 0):.2f}%")
        print(f"    最大回撤: {sr.metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"    Sharpe: {sr.metrics.get('sharpe', 0):.4f}")
        print(f"    交易次数: {sr.metrics.get('trade_count', 0)}")

    # 生成报告
    report_path = runner.generate_report(result, output_dir="./output_backtest")
    print(f"\n报告已保存到: {report_path}")


if __name__ == '__main__':
    main()
