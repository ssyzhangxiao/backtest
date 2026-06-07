"""
训练/测试分割验证模块。

实现 P1-B 验证：训练期 vs 验证期的策略表现对比。
委托 runner/backtest/ 执行回测，不直接调用原 run_* 脚本。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.engine.backtest_runner import PyBrokerBacktestRunner
from core.engine.pybroker_data_source import PyBrokerDataSource
from core.config.strategy_profiles import StrategyLibrary
from runner.backtest.experiments import run_e6_walkforward, run_e7_out_of_sample
from runner.common.utils import safe_float, save_csv


def task2_train_test_split(
    data_source: PyBrokerDataSource,
    config: BacktestConfig,
    lib: StrategyLibrary,
    output_dir: Path,
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    cross_sectional: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    训练/验证期划分验证（P1-B）。

    调用 runner/backtest/experiments 的标准接口：
      - run_e6_walkforward: 单策略 WalkForward 滚动验证
      - run_e7_out_of_sample: 单策略汇总样本内外对比
    同时补充环境分布统计和按年切片验证。

    Args:
        data_source: 数据源
        config: 回测配置（BacktestConfig）
        lib: 策略库
        output_dir: 输出目录
        best_params: 优化后的最优参数

    Returns:
        验证结果字典
    """
    train_start = config.train_start
    train_end = config.train_end
    test_start = config.test_start
    test_end = config.test_end

    logger.info("=" * 60)
    logger.info("任务2: 训练/验证期划分验证（P1-B）")
    logger.info(f"  训练期: {train_start} ~ {train_end}")
    logger.info(f"  验证期: {test_start} ~ {test_end}")
    logger.info("=" * 60)

    strategy_names = config.strategy_names
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建与 experiments 兼容的配置字典
    val_config = _build_validation_config(config)

    # WalkForward 滚动验证
    logger.info("\n  WalkForward 滚动验证:")
    wf_df = run_e6_walkforward(data_source, val_config, output_dir)

    # 样本外验证
    logger.info("\n  样本外验证:")
    oos_df = run_e7_out_of_sample(data_source, val_config, output_dir)

    # 环境分布统计
    logger.info("\n  环境分布统计:")
    env_stats = _compute_environment_stats(data_source, config)
    if not env_stats.empty:
        env_stats.to_csv(output_dir / "task2_env_stats.csv", index=False)
        logger.info(f"\n{env_stats.to_string(index=False)}")

    # 训练期回测（固定参数 vs 环境感知参数）
    logger.info("\n  训练期回测（固定参数 vs 环境感知参数）:")
    train_fixed = _run_period_backtest(
        strategy_names, data_source, config, train_start, train_end, "fixed",
        best_params=best_params,
    )
    train_regime = _run_period_backtest(
        strategy_names, data_source, config, train_start, train_end, "regime",
        best_params=best_params,
    )

    # 验证期按年切片
    logger.info("\n  验证期按年切片:")
    yearly_results = _run_yearly_validation(
        strategy_names, data_source, config, train_end, test_end, best_params,
    )
    df_yearly = pd.DataFrame(yearly_results)
    df_yearly.to_csv(output_dir / "task2_yearly_validation.csv", index=False)

    # 参数对比表
    param_table = _get_param_comparison_table(output_dir)

    # 汇总
    summary_rows = []
    for sname in strategy_names:
        fk = train_fixed.get(sname, {})
        rk = train_regime.get(sname, {})
        summary_rows.append({
            "strategy": sname,
            "train_fixed_sharpe": round(safe_float(fk.get("sharpe", 0)), 4),
            "train_regime_sharpe": round(safe_float(rk.get("sharpe", 0)), 4),
            "train_fixed_dd": round(safe_float(fk.get("max_drawdown_pct", 0)), 2),
            "train_regime_dd": round(safe_float(rk.get("max_drawdown_pct", 0)), 2),
        })
    df_summary = pd.DataFrame(summary_rows)

    return {
        "walkforward": wf_df,
        "out_of_sample": oos_df,
        "env_stats": env_stats,
        "yearly": df_yearly,
        "param_table": param_table,
        "summary": df_summary,
    }


def _build_validation_config(config: BacktestConfig) -> Dict[str, Any]:
    """
    构建与 runner/backtest/experiments 兼容的配置字典。

    experiments 模块暂仍接收字典，此处做适配层。

    Args:
        config: 回测配置（BacktestConfig）

    Returns:
        回测配置字典
    """
    return {
        "backtest": {
            "initial_cash": config.initial_cash,
            "commission_rate": config.commission_rate,
            "slippage_rate": config.slippage_rate,
            "full_start_date": config.full_start,
            "full_end_date": config.full_end,
            "in_sample_end_date": config.in_sample_end,
            "out_sample_start_date": config.out_sample_start,
        },
        "symbols": config.symbols,
        "strategies": [{"name": s} for s in config.strategy_names],
        "risk_management": {
            "stop_loss_pct": 0.05,
            "position_limit_pct": 0.4,
            "total_position_limit": 0.8,
        },
        "factor_weights": {},
        "monte_carlo": {
            "n_simulations": 1000,
            "random_seed": 42,
        },
        "output": {"output_dir": config.output_dir},
    }


def _compute_environment_stats(
    ds: PyBrokerDataSource,
    config: BacktestConfig,
) -> pd.DataFrame:
    """
    计算各品种的环境分布统计（5类环境）。

    环境感知参数管理系统已废弃，因子打分调仓模式不再需要环境分布统计。
    保留空函数以维持接口兼容性。
    """
    logger.info("环境分布统计已废弃（因子打分模式），跳过")
    return pd.DataFrame()


def _run_period_backtest(
    strategy_names: List[str],
    ds: PyBrokerDataSource,
    config: BacktestConfig,
    start: str,
    end: str,
    mode: str = "fixed",
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    执行某时段的回测，返回各策略KPI。

    mode="fixed": 固定参数（或优化参数）回测
    mode="regime": 环境感知回测（已废弃）

    Args:
        strategy_names: 策略名称列表
        ds: 数据源
        config: 回测配置（BacktestConfig）
        start: 开始日期
        end: 结束日期
        mode: 回测模式
        best_params: 优化参数

    Returns:
        {策略名: {指标名: 值}} 字典
    """
    results = {}

    if mode == "regime":
        return _run_regime_backtest(strategy_names, ds, config, start, end)

    # 固定参数回测
    bt_config = BacktestConfig(
        initial_cash=config.initial_cash,
        commission_rate=config.commission_rate,
        slippage_rate=config.slippage_rate,
    )
    for sname in strategy_names:
        try:
            runner = PyBrokerBacktestRunner(ds, bt_config)
            runner.register_strategies([sname])

            custom_params = None
            if best_params and sname in best_params:
                custom_params = {sname: best_params[sname]}

            result = runner.run(start, end, custom_params=custom_params)
            results[sname] = dict(result.metrics)
        except Exception as e:
            logger.warning(f"  固定参数回测失败 ({sname}, {start}~{end}): {e}")
            results[sname] = {}

    return results


def _run_regime_backtest(
    strategy_names: List[str],
    ds: PyBrokerDataSource,
    config: BacktestConfig,
    start: str,
    end: str,
) -> Dict[str, Dict[str, Any]]:
    """
    环境感知回测（已废弃）。

    环境感知参数管理系统和 V3RegimeAwareRunner 已废弃，
    系统已切换为因子打分调仓模式。保留空函数以维持接口兼容性。
    """
    logger.info("环境感知回测已废弃（因子打分模式），跳过")
    return {}


def _run_yearly_validation(
    strategy_names: List[str],
    ds: PyBrokerDataSource,
    config: BacktestConfig,
    train_end: str,
    test_end: str,
    best_params: Optional[Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    验证期按年切片回测。

    Args:
        strategy_names: 策略名称列表
        ds: 数据源
        config: 回测配置（BacktestConfig）
        train_end: 训练期结束日期
        test_end: 测试期结束日期
        best_params: 优化参数

    Returns:
        按年验证结果列表
    """
    yearly_results = []
    start_year = int(train_end[:4])
    end_year = int(test_end[:4])

    for year in range(start_year + 1, end_year + 1):
        start = f"{year}-01-01"
        end = f"{year}-12-31"

        fixed_kpi = _run_period_backtest(
            strategy_names, ds, config, start, end, "fixed",
            best_params=best_params,
        )
        regime_kpi = _run_period_backtest(
            strategy_names, ds, config, start, end, "regime",
            best_params=best_params,
        )

        for sname in strategy_names:
            fk = fixed_kpi.get(sname, {})
            rk = regime_kpi.get(sname, {})
            yearly_results.append({
                "year": year,
                "strategy": sname,
                "fixed_sharpe": round(safe_float(fk.get("sharpe", 0)), 4),
                "fixed_return": round(safe_float(fk.get("total_return_pct", 0)), 2),
                "fixed_drawdown": round(safe_float(fk.get("max_drawdown_pct", 0)), 2),
                "regime_sharpe": round(safe_float(rk.get("sharpe", 0)), 4),
                "regime_return": round(safe_float(rk.get("total_return_pct", 0)), 2),
                "regime_drawdown": round(safe_float(rk.get("max_drawdown_pct", 0)), 2),
            })

        logger.info(f"    {year}: 完成")

    return yearly_results


def _get_param_comparison_table(output_dir: Path) -> pd.DataFrame:
    """
    获取参数对比表（已废弃）。

    环境感知参数管理系统已废弃，系统已切换为因子打分调仓模式。
    保留空函数以维持接口兼容性。
    """
    logger.info("参数对比表已废弃（因子打分模式），跳过")
    return pd.DataFrame()
