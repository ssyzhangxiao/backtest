"""
PyBroker 主引擎回测示例。

演示完整流程：
  1. 加载数据（TqSdk 在线优先 → 本地 CSV fallback）
  2. 创建 PyBroker 数据源
  3. 注册策略并运行回测
  4. Walkforward 调参
  5. Bootstrap 置信区间
  6. 交叉验证（自研引擎 vs PyBroker）

前置条件：
  - pip install pybroker>=1.0.0 numba>=0.56.0
  - TqSdk 模式: 需提供快期账号（环境变量 TQSDK_PHONE / TQSDK_PASSWORD）
  - 本地模式: data/ 目录下存在 CSV 期货数据文件
"""

from core.engine.pybroker_data_source import (
    PyBrokerDataSource,
    create_hybrid_data_source,
)
from core.engine.regime_indicator import RegimeIndicator
from core.engine.strategy_executor import StrategyExecutorFactory
from core.engine.backtest_runner import (
    PyBrokerBacktestRunner,
    WalkforwardResult,
)
from core.config import BacktestConfig
from core.data_loader import DataLoader
from core.engine.runner import BacktestRunner
from core.market_regime import MarketRegimeDetector
from core.strategy_registry import StrategyLibrary
from core.engine.switch_engine import FactorScoringEngine


def main():
    # ── 1. 加载数据 ──
    print("=" * 60)
    print("PyBroker 主引擎回测")
    print("=" * 60)

    # 方式 A：混合数据源（TqSdk 优先，本地 CSV fallback）
    #   - 设置环境变量 TQSDK_PHONE + TQSDK_PASSWORD 可自动走 TqSdk
    #   - 无 TqSdk 凭证时自动回退到本地 ./data/*.csv
    try:
        data_source = create_hybrid_data_source(
            symbols=[  # TqSdk 模式下加载的品种
                "SHFE.RB",
                "SHFE.AG",
                "SHFE.AU",
                "DCE.M",
                "DCE.I",
                "CZCE.TA",
                "CZCE.MA",
            ],
            data_dir="./data",
        )
        print(f"数据加载完成: {len(data_source)} 行, {len(data_source.symbols)} 合约")
        print(f"日期范围: {data_source.date_range[0]} ~ {data_source.date_range[1]}")
    except RuntimeError as e:
        print(f"[ERROR] 数据加载失败: {e}")
        return

    # 方式 B：手动控制（仅用 DataLoader）
    # loader = DataLoader("./data")
    # loader.load_csv_files("*.csv")
    # loader.build_continuous_series()
    # df = loader.get_pybroker_df()
    # data_source = PyBrokerDataSource(df)

    # ── 2. 配置 ──
    config = BacktestConfig(
        initial_cash=1_000_000,
        commission_rate=0.0003,
        slippage_rate=0.0002,
        max_position_pct=0.2,
        stop_loss_pct=0.05,
        pybroker_bootstrap_samples=10000,
        pybroker_buy_delay=1,
        pybroker_sell_delay=1,
        wf_train_ratio=0.6,
        wf_step_ratio=0.1,
        cross_validate=True,
    )

    # ── 3. 创建 PyBroker 主引擎 ──
    runner = PyBrokerBacktestRunner(data_source, config)
    runner.register_strategies(["ts_momentum", "roll_yield", "alpha019", "alpha032"])

    # ── 4. 运行回测 ──
    print("\n--- 运行回测 ---")
    result = runner.run("2023-01-01", "2024-12-31")

    print("\n回测结果:")
    for key, val in result.metrics.items():
        print(f"  {key:20s}: {val}")

    if not result.switch_log.empty:
        print(f"\n策略切换次数: {len(result.switch_log)}")
        print(result.switch_log.tail(5).to_string(index=False))

    # ── 5. Walkforward 调参 ──
    print("\n--- Walkforward 向前滚动分析 ---")
    try:
        wf_result = runner.walkforward("2020-01-01", "2024-12-31")
        print(f"窗口数: {len(wf_result.windows)}")
        print(f"整体指标: {wf_result.overall_metrics}")
        # wf_result.plot_equity_curves()  # 需 plotly
    except Exception as e:
        print(f"[WARN] Walkforward 失败: {e}")

    # ── 6. Bootstrap 评估 ──
    print("\n--- Bootstrap 置信区间 ---")
    try:
        bs = runner.bootstrap_metrics(n_samples=5000)
        for metric, stats in bs.items():
            print(
                f"  {metric}: mean={stats['mean']:.3f}, "
                f"CI=[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
            )
    except Exception as e:
        print(f"[WARN] Bootstrap 失败: {e}")

    # ── 7. 交叉验证（自研引擎 vs PyBroker） ──
    if config.cross_validate:
        print("\n--- 交叉验证 ---")
        try:
            legacy_runner = BacktestRunner("./data", config)
            legacy_runner.load_data("*.csv")
            legacy_result = legacy_runner.run(
                strategies=["ts_momentum"],
                start_date="2023-01-01",
                end_date="2024-12-31",
            )

            # 取出第一个策略的净值
            first_strategy = list(legacy_result.strategy_results.values())[0]
            legacy_eq = first_strategy.equity_curve

            diff = legacy_runner.cross_validate_with_pybroker(
                pybroker_equity=result.equity_curve,
                own_equity=legacy_eq.set_index("date")["equity"],
            )
            print(f"  净值相关系数: {diff['correlation']}")
            print(f"  最大绝对差异: {diff['max_abs_diff']}")
            print(f"  平均绝对差异: {diff['mean_abs_diff']}")
        except Exception as e:
            print(f"[WARN] 交叉验证失败: {e}")

    print("\n" + "=" * 60)
    print("回测完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
