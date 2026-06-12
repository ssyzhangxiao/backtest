"""
实验 E8 / E9：统计重采样稳健性验证（Bootstrap + 蒙特卡洛）。

E8：对回测收益序列进行 Bootstrap 重采样，估计 Sharpe 等指标置信区间。
    委托 runner.bootstrap_metrics / MetricsCalculator.bootstrap_confidence_interval 实现。

E9：基于历史收益率，通过有放回重采样模拟未来净值路径分布。
    委托 core.validation.monte_carlo.MonteCarloSimulator 计算，
    直接接受 BacktestConfig（避免 _build_mc_config + get_pybroker_runner 链路丢字段）。

2026-06-11 修复（独立 bug）：E9 改接受 BacktestConfig 而非 Dict。
  此前通过 _build_mc_config 转 dict 后再 get_pybroker_runner → build_backtest_config
  会丢失 factor_weights（默认空），导致 ScoringConfig 把 5 子策略权重置 0，
  最终 0 trade → MC 1000 次 path 全 1.0。
  修复后直接 PyBrokerBacktestRunner(data_source, config, target_symbols=config.symbols)，
  BacktestConfig 全字段（含 factor_weights / stop_loss_pct / max_position_pct / rebalance_days）透传。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.config import BacktestConfig  # 2026-06-11 修复：直接接受 BacktestConfig
from core.engine.pybroker_data_source import PyBrokerDataSource
from core.execution.backtest_runner import PyBrokerBacktestRunner
from core.validation.monte_carlo import MonteCarloSimulator
from utils.metrics import MetricsCalculator
from runner.backtest.runner import get_pybroker_runner, safe_run_backtest
from runner.common.utils import (
    format_metrics,
    handle_backtest_errors,
    is_valid_number,
    save_csv,
)
from runner.strategy.selector import get_strategy_names


# ============================================
# 模块常量
# ============================================

# 2026-06-11 修复：蒙特卡洛 MC 阶段的固定参数（无法从 BacktestConfig 直接读取）
_E9_N_SIMULATIONS = 1000
_E9_RANDOM_SEED = 42
_E9_TRADING_DAYS_PER_YEAR = 252


# ============================================
# E8：Bootstrap 结果数据类
# ============================================


@dataclass
class BootstrapResult:
    """Bootstrap 结果数据类"""

    sharpe_samples: List[float]
    confidence_intervals: pd.DataFrame
    n_samples: int
    random_seed: int


# ============================================
# E8：Bootstrap 置信区间
# ============================================


@handle_backtest_errors(return_value=BootstrapResult([], pd.DataFrame(), 0, 42))
def run_e8_bootstrap(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> BootstrapResult:
    """
    E8：Bootstrap 置信区间。

    对回测收益序列进行 Bootstrap 重采样，估计 Sharpe 等指标的置信区间。
    定义 BootstrapResult 数据类统一返回格式，添加随机种子。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        BootstrapResult 对象
    """
    logger.info("E8：Bootstrap 置信区间")
    bt_cfg = config["backtest"]
    bs_cfg: Dict[str, Any] = config.get("bootstrap", {})
    n_samples = int(bs_cfg.get("n_samples", 5000))
    random_seed = int(bs_cfg.get("random_seed", 42))
    strategy_names = get_strategy_names(config)
    default_strategy = strategy_names[:1] if strategy_names else ["trend"]

    runner = get_pybroker_runner(data_source, config, strategies=default_strategy)
    result = safe_run_backtest(
        runner, bt_cfg["full_start_date"], bt_cfg["full_end_date"], "E8_base"
    )

    if result is None or result.equity_curve is None or result.equity_curve.empty:
        logger.warning("E8：无净值数据，跳过Bootstrap")
        return BootstrapResult([], pd.DataFrame(), n_samples, random_seed)

    bootstrap_result: Any = None
    try:
        bootstrap_result = runner.bootstrap_metrics(n_samples=n_samples)
        logger.info(f"  系统Bootstrap完成: {n_samples} 样本")
    except Exception as e:
        logger.warning(f"  系统Bootstrap失败: {e}, 回退到MetricsCalculator")
        try:
            equity = result.equity_curve["equity"].values
            bootstrap_result = MetricsCalculator.bootstrap_confidence_interval(
                equity, n_samples=n_samples
            )
            logger.info(f"  MetricsCalculator Bootstrap完成: {n_samples} 样本")
        except Exception as e2:
            logger.error(f"  MetricsCalculator Bootstrap也失败: {e2}")
            return BootstrapResult([], pd.DataFrame(), n_samples, random_seed)

    if bootstrap_result is None:
        return BootstrapResult([], pd.DataFrame(), n_samples, random_seed)

    # 结构化结果
    sharpe_samples: List[float] = []
    df_ci = pd.DataFrame()

    if isinstance(bootstrap_result, dict):
        first_val = next(iter(bootstrap_result.values()), None)
        if isinstance(first_val, dict) and "mean" in first_val:
            rows: List[Dict[str, Any]] = []
            for metric_name, vals in bootstrap_result.items():
                if isinstance(vals, dict):
                    rows.append({"metric": metric_name, **vals})
            df_ci = pd.DataFrame(rows)
        else:
            for val in bootstrap_result.values():
                if isinstance(val, list) and len(val) > 0:
                    sharpe_samples = val
                    break
            if sharpe_samples:
                df_samples = pd.DataFrame({"sharpe": sharpe_samples})
                save_csv(df_samples, output_dir / "e8_bootstrap_samples.csv")

    if not df_ci.empty:
        save_csv(df_ci, output_dir / "e8_bootstrap_confidence_intervals.csv")

    return BootstrapResult(sharpe_samples, df_ci, n_samples, random_seed)


# ============================================
# E9：蒙特卡洛模拟
# ============================================


@handle_backtest_errors(return_value=None)
def run_e9_monte_carlo(
    data_source: PyBrokerDataSource,
    config: BacktestConfig,
    output_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    E9：蒙特卡洛模拟（P0 整改：迁移到公共 MonteCarloSimulator）。

    基于历史收益率序列，通过有放回重采样模拟未来净值路径分布。
    使用 core.validation.monte_carlo.MonteCarloSimulator 计算，
    删除原手写的 _compute_monte_carlo_stats 私有实现。

    新增（return_paths=True）：
      - 保存完整模拟路径 (n_simulations, n_days+1) 到 e9_monte_carlo_paths.npz
      - 用于路径分布图 / 历史回放
    复用（MonteCarloResult.is_robust）：
      - Sharpe 的 5% 分位数 > 0 判定为稳健
      - 在 e9_monte_carlo_stats.csv 写入 is_robust 列

    2026-06-11 修复（独立 bug）：改接受 BacktestConfig 而非 Dict。
      此前通过 _build_mc_config 转 dict 后再 get_pybroker_runner → build_backtest_config
      会丢失 factor_weights（默认空），导致 ScoringConfig 把 5 子策略权重置 0，
      最终 0 trade → MC 1000 次 path 全 1.0。
      修复后直接 PyBrokerBacktestRunner(data_source, config, target_symbols=config.symbols)，
      BacktestConfig 全字段（含 factor_weights / stop_loss_pct / max_position_pct / rebalance_days）透传。

    Args:
        data_source: 数据源
        config: BacktestConfig 实例（单一权威源，规则2）
        output_dir: 输出目录

    Returns:
        模拟结果 DataFrame，失败返回 None
    """
    logger.info("E9：蒙特卡洛模拟")
    # 2026-06-11 修复：直接读 BacktestConfig 字段，不再依赖 _build_mc_config 转 dict
    full_start_date = config.full_start
    full_end_date = config.full_end
    n_simulations = _E9_N_SIMULATIONS
    random_seed = _E9_RANDOM_SEED
    bankruptcy_threshold = float(config.bankruptcy_threshold)
    trading_days_per_year = _E9_TRADING_DAYS_PER_YEAR
    # 过滤横截面组合模式（与 phase2 / get_strategy_names 对齐）
    target_strategy_names = [
        n
        for n in (config.strategy_names or get_strategy_names({}))
        if n != "cross_sectional"
    ]

    try:
        # 2026-06-11 修复：直接用 BacktestConfig 构造 runner（与 phase2 风格一致），
        # 避免 _build_mc_config + get_pybroker_runner 链路丢字段。
        runner = PyBrokerBacktestRunner(
            data_source, config, target_symbols=list(config.symbols)
        )
        if target_strategy_names:
            runner.register_strategies(target_strategy_names)
        result = safe_run_backtest(runner, full_start_date, full_end_date, "E9_base")
        if result is None:
            return None

        eq = result.equity_curve
        if eq is None or eq.empty:
            logger.warning("E9：无净值数据，跳过蒙特卡洛模拟")
            return None

        eq_sorted = eq.sort_values("date")
        returns = eq_sorted["equity"].pct_change().dropna()
        returns = returns[returns.apply(is_valid_number)]

        if len(returns) == 0:
            logger.warning("E9：无有效收益率数据")
            return None

        # P0 整改：调用公共模拟器，return_paths=True 获取完整路径
        simulator = MonteCarloSimulator(
            n_simulations=n_simulations,
            random_seed=random_seed,
            trading_days_per_year=trading_days_per_year,
        )
        mc_result = simulator.simulate(returns.values, return_paths=True)
        if mc_result.n_simulations == 0:
            logger.warning("E9：模拟失败（数据不足）")
            return None

        # 终值 = 路径末端值（沿用旧 E9 字段命名以兼容下游）
        final_values = mc_result.paths[:, -1] if mc_result.paths is not None else None
        # 破产概率（终值 < 阈值）
        bankruptcy_prob = (
            float(np.mean(final_values < bankruptcy_threshold))
            if final_values is not None
            else float("nan")
        )
        # P0 整改：通过 MonteCarloResult.is_robust 判定（Sharpe 5% 分位 > 0）
        is_robust = mc_result.is_robust

        logger.info(f"  模拟次数: {n_simulations}")
        logger.info(
            f"  终值均值: {np.mean(final_values):.4f}, 中位数: {np.median(final_values):.4f}"
        )
        logger.info(f"  破产概率(终值<{bankruptcy_threshold}): {bankruptcy_prob:.2%}")
        logger.info(
            f"  稳健性: {'✅稳健 (Sharpe@5%>0)' if is_robust else '⚠️不稳健 (Sharpe@5%≤0)'}"
        )

        # 兼容旧字段：per-sim final_value & max_drawdown 表格
        # 从 path 重新计算 max_drawdown（与旧版口径一致：每条路径的最大回撤）
        if mc_result.paths is not None:
            peak = np.maximum.accumulate(mc_result.paths, axis=1)
            safe_peak = np.where(peak > 0, peak, 1.0)
            per_sim_max_dd = np.min(mc_result.paths / safe_peak - 1.0, axis=1)
        else:
            per_sim_max_dd = np.zeros(n_simulations)

        mc_results = pd.DataFrame(
            {
                "sim_id": range(n_simulations),
                "final_value": final_values,
                "max_drawdown": per_sim_max_dd,
            }
        )
        save_csv(mc_results, output_dir / "e9_monte_carlo_results.csv")

        # 模拟统计数据（兼容旧版字段 + 新增 is_robust / sharpe_quantiles）
        stats_data: Dict[str, list] = {
            "statistic": ["mean", "median", "std", "p5", "p95"],
            "final_value": [
                float(np.mean(final_values)),
                float(np.median(final_values)),
                float(np.std(final_values)),
                float(np.percentile(final_values, 5)),
                float(np.percentile(final_values, 95)),
            ],
            "max_drawdown": [
                float(np.mean(per_sim_max_dd)),
                float(np.median(per_sim_max_dd)),
                float(np.std(per_sim_max_dd)),
                float(np.percentile(per_sim_max_dd, 5)),
                float(np.percentile(per_sim_max_dd, 95)),
            ],
        }
        # P0 整改：把 MonteCarloResult 公共输出落到 stats csv，便于外部分析
        stats_data["sharpe_quantile_05"] = [
            mc_result.sharpe_quantiles.get(0.05, 0.0)
        ] * 5
        stats_data["sharpe_quantile_50"] = [
            mc_result.sharpe_quantiles.get(0.50, 0.0)
        ] * 5
        stats_data["is_robust"] = [is_robust] + [None] * 4  # 仅第一行有意义
        save_csv(pd.DataFrame(stats_data), output_dir / "e9_monte_carlo_stats.csv")

        # 保存完整模拟路径（npz 压缩），供路径分布图 / 历史回放
        if mc_result.paths is not None:
            np.savez_compressed(
                output_dir / "e9_monte_carlo_paths.npz",
                paths=mc_result.paths,
                initial_dates=eq_sorted["date"].iloc[1:].to_numpy(),
                n_simulations=n_simulations,
                random_seed=random_seed,
                trading_days_per_year=trading_days_per_year,
            )
            logger.info(
                f"  路径已保存: e9_monte_carlo_paths.npz shape={mc_result.paths.shape}"
            )

        return mc_results
    except Exception as e:
        logger.error(f"  蒙特卡洛模拟失败: {e}")
        return None
