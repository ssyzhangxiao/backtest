"""
蒙特卡洛验证模块。

委托 core/validation/monte_carlo.py 的 MonteCarloSimulator 执行模拟，
同时保留逐策略详细分析逻辑（破产概率、月胜率等扩展指标）。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.engine.backtest_runner import PyBrokerBacktestRunner
from core.engine.pybroker_data_source import PyBrokerDataSource
from core.strategy_registry import StrategyLibrary
from runner.backtest.experiments import run_e9_monte_carlo
from runner.common.utils import is_valid_number

_N_MONTE_CARLO = 1000
_RANDOM_SEED = 42
_DEFAULT_BANKRUPTCY_THRESHOLD = 0.8


def task3_monte_carlo(
    ds: PyBrokerDataSource,
    config: BacktestConfig,
    lib: StrategyLibrary,
    output_dir: Path,
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    蒙特卡洛 1000 次鲁棒性测试（P2-D 验收项）。

    调用 runner/backtest/experiments.run_e9_monte_carlo 的标准接口，
    同时对每个策略单独执行蒙特卡洛并输出详细分布。

    Args:
        ds: 数据源
        config: 回测配置（BacktestConfig）
        lib: 策略库
        output_dir: 输出目录
        best_params: 优化后的最优参数

    Returns:
        蒙特卡洛验证结果字典
    """
    full_start = config.full_start
    full_end = config.full_end
    bankruptcy_threshold = config.bankruptcy_threshold

    logger.info("=" * 60)
    logger.info("任务3: 蒙特卡洛 1000 次鲁棒性测试")
    logger.info(f"  模拟次数: {_N_MONTE_CARLO}")
    logger.info(f"  破产阈值: {bankruptcy_threshold:.1%}")
    logger.info("=" * 60)

    strategy_names = list(config.strategy_names)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 调用标准蒙特卡洛实验
    val_config = _build_mc_config(config, strategy_names)
    mc_base_df = run_e9_monte_carlo(ds, val_config, output_dir)

    # 逐策略蒙特卡洛详细分析
    all_mc_results = _run_per_strategy_mc(
        strategy_names,
        ds,
        config,
        full_start,
        full_end,
        best_params,
        bankruptcy_threshold,
    )

    # 汇总表
    summary_rows = []
    for sname, mc in all_mc_results.items():
        summary_rows.append(
            {
                "strategy": sname,
                "final_mean": round(mc["final_mean"], 4),
                "final_median": round(mc["final_median"], 4),
                "final_5pct": round(mc["final_5pct"], 4),
                "final_95pct": round(mc["final_95pct"], 4),
                "bankruptcy_prob": round(mc["bankruptcy_prob"], 4),
                "avg_max_dd": round(mc["avg_max_dd"], 4),
                "avg_monthly_win_rate": round(mc["avg_monthly_win_rate"], 4),
                "calmar_mean": round(mc["calmar_mean"], 4),
            }
        )

    df_mc = pd.DataFrame(summary_rows)
    df_mc.to_csv(output_dir / "task3_monte_carlo_summary.csv", index=False)

    # 保存详细模拟数据
    for sname, mc in all_mc_results.items():
        detail = pd.DataFrame(
            {
                "sim_id": range(_N_MONTE_CARLO),
                "final_value": mc["final_values"],
                "max_drawdown": mc["max_drawdowns"],
            }
        )
        detail.to_csv(output_dir / f"task3_mc_detail_{sname}.csv", index=False)

    return {"summary": df_mc, "details": all_mc_results, "base": mc_base_df}


def _build_mc_config(
    config: BacktestConfig,
    strategy_names: List[str],
) -> Dict[str, Any]:
    """
    构建蒙特卡洛实验配置。

    Args:
        config: 回测配置（BacktestConfig）
        strategy_names: 策略名称列表

    Returns:
        实验配置字典
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
        "strategies": [{"name": s} for s in strategy_names],
        "risk_management": {
            "stop_loss_pct": 0.05,
            "position_limit_pct": 0.4,
            "total_position_limit": 0.8,
        },
        "factor_weights": {},
        "monte_carlo": {
            "n_simulations": _N_MONTE_CARLO,
            "random_seed": _RANDOM_SEED,
        },
        "output": {"output_dir": config.output_dir},
    }


def _run_per_strategy_mc(
    strategy_names: List[str],
    ds: PyBrokerDataSource,
    config: BacktestConfig,
    full_start: str,
    full_end: str,
    best_params: Optional[Dict[str, Dict[str, Any]]],
    bankruptcy_threshold: float,
) -> Dict[str, Dict[str, Any]]:
    """
    逐策略执行蒙特卡洛详细分析。

    委托 core/validation/monte_carlo.MonteCarloSimulator 执行核心模拟，
    同时计算扩展指标（破产概率、月胜率、Calmar比率）。

    Args:
        strategy_names: 策略名称列表
        ds: 数据源
        config: 回测配置（BacktestConfig）
        full_start: 全期开始日期
        full_end: 全期结束日期
        best_params: 优化参数
        bankruptcy_threshold: 破产阈值

    Returns:
        {策略名: 模拟结果字典}
    """
    all_mc_results = {}

    for sname in strategy_names:
        logger.info(f"\n  策略: {sname}")
        try:
            bt_config = BacktestConfig(
                initial_cash=config.initial_cash,
                commission_rate=config.commission_rate,
                slippage_rate=config.slippage_rate,
            )
            runner = PyBrokerBacktestRunner(ds, bt_config)
            runner.register_strategies([sname])

            custom_params = None
            if best_params and sname in best_params:
                custom_params = {sname: best_params[sname]}

            result = runner.run(full_start, full_end, custom_params=custom_params)
            eq = result.equity_curve

            if eq is None or eq.empty:
                logger.warning(f"  {sname}: 无净值数据，跳过")
                continue

            eq_sorted = eq.sort_values("date")
            returns = eq_sorted["equity"].pct_change().dropna()
            returns = returns[returns.apply(is_valid_number)]

            if len(returns) == 0:
                logger.warning(f"  {sname}: 无有效收益率，跳过")
                continue

            mc_result = _run_monte_carlo_sim(
                returns,
                n_simulations=_N_MONTE_CARLO,
                seed=_RANDOM_SEED,
                bankruptcy_threshold=bankruptcy_threshold,
            )
            all_mc_results[sname] = mc_result

            logger.info(f"    终值均值: {mc_result['final_mean']:.4f}")
            logger.info(f"    终值中位数: {mc_result['final_median']:.4f}")
            logger.info(
                f"    破产概率(终值<{bankruptcy_threshold:.1f}): "
                f"{mc_result['bankruptcy_prob']:.2%}"
            )
            logger.info(f"    最大回撤均值: {mc_result['avg_max_dd']:.2%}")
            logger.info(f"    月胜率均值: {mc_result['avg_monthly_win_rate']:.1%}")

        except Exception as e:
            logger.error(f"  {sname} 蒙特卡洛失败: {e}")

    return all_mc_results


def _run_monte_carlo_sim(
    returns: pd.Series,
    n_simulations: int = 1000,
    seed: int = 42,
    bankruptcy_threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    执行蒙特卡洛模拟。

    委托 core/validation/monte_carlo.MonteCarloSimulator 执行核心向量化模拟，
    同时计算扩展指标（破产概率、月胜率、Calmar比率）。

    Args:
        returns: 日收益率序列
        n_simulations: 模拟次数
        seed: 随机种子
        bankruptcy_threshold: 破产阈值

    Returns:
        模拟结果字典
    """
    rng = np.random.default_rng(seed)
    ret_array = returns.values
    n_days = len(ret_array)

    # 向量化模拟：一次性生成所有路径
    sim_equities = np.zeros((n_simulations, n_days + 1))
    sim_equities[:, 0] = 1.0

    for i in range(n_simulations):
        sampled = rng.choice(ret_array, size=n_days, replace=True)
        sim_equities[i, 1:] = np.cumprod(1.0 + sampled)

    final_values = sim_equities[:, -1]

    # 向量化最大回撤
    peak_equities = np.maximum.accumulate(sim_equities, axis=1)
    peak_safe = np.where(peak_equities > 0, peak_equities, 1.0)
    drawdowns = sim_equities / peak_safe - 1.0
    max_drawdowns = np.min(drawdowns, axis=1)

    # 月胜率计算
    monthly_win_rates = _compute_monthly_win_rates(
        sim_equities,
        returns,
        n_simulations,
        n_days,
    )
    monthly_win_rate = float(np.mean(monthly_win_rates)) if monthly_win_rates else 0.0

    # Calmar比率
    annual_returns = final_values ** (252 / n_days) - 1
    calmar_ratios = annual_returns / np.abs(max_drawdowns + 1e-10)

    # 破产概率
    bankruptcy_prob = float(np.mean(final_values < bankruptcy_threshold))

    return {
        "final_values": final_values,
        "max_drawdowns": max_drawdowns,
        "final_mean": float(np.mean(final_values)),
        "final_median": float(np.median(final_values)),
        "final_5pct": float(np.percentile(final_values, 5)),
        "final_95pct": float(np.percentile(final_values, 95)),
        "bankruptcy_prob": bankruptcy_prob,
        "bankruptcy_threshold": bankruptcy_threshold,
        "avg_max_dd": float(np.mean(max_drawdowns)),
        "avg_monthly_win_rate": monthly_win_rate,
        "calmar_mean": float(np.mean(calmar_ratios)),
    }


def _compute_monthly_win_rates(
    sim_equities: np.ndarray,
    returns: pd.Series,
    n_simulations: int,
    n_days: int,
) -> List[float]:
    """
    计算各模拟路径的月胜率。

    优先使用 DatetimeIndex 重采样，失败时按21交易日分月。

    Args:
        sim_equities: (n_simulations, n_days+1) 净值矩阵
        returns: 原始收益率序列
        n_simulations: 模拟次数
        n_days: 交易日数

    Returns:
        月胜率列表
    """
    monthly_win_rates = []

    for i in range(n_simulations):
        eq = pd.Series(sim_equities[i])
        if hasattr(returns, "index") and isinstance(returns.index, pd.DatetimeIndex):
            try:
                eq.index = returns.index
                monthly_eq = eq.resample("ME").last().dropna()
                monthly_ret = monthly_eq.pct_change().dropna()
            except (ValueError, TypeError):
                monthly_ret = pd.Series(dtype=float)
        else:
            monthly_ret = pd.Series(dtype=float)

        if len(monthly_ret) > 0:
            monthly_win_rates.append(float((monthly_ret > 0).mean()))
        else:
            # 回退：按21交易日分月
            n_months = max(1, n_days // 21)
            wins = 0
            for m in range(n_months):
                start_idx = m * 21
                end_idx = min((m + 1) * 21, n_days)
                if end_idx < n_days + 1 and eq.iloc[end_idx] > eq.iloc[start_idx]:
                    wins += 1
            monthly_win_rates.append(wins / n_months)

    return monthly_win_rates
