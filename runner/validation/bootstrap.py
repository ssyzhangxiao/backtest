"""
Bootstrap 置信区间验证模块。

对回测收益序列进行 Bootstrap 重采样，
估计 Sharpe 等指标的置信区间。
委托 PyBrokerBacktestRunner.bootstrap_metrics 和
utils/metrics.MetricsCalculator.bootstrap_confidence_interval，
不重复实现 Bootstrap 重采样逻辑。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.engine.pybroker_data_source import PyBrokerDataSource
from core.config.strategy_profiles import StrategyLibrary
from runner.backtest.runner import get_pybroker_runner, safe_run_backtest
from runner.common.utils import save_csv

_DEFAULT_N_SAMPLES = 5000


def run_bootstrap_validation(
    data_source: PyBrokerDataSource,
    config: BacktestConfig,
    lib: StrategyLibrary,
    output_dir: Path,
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    cross_sectional: bool = False,
    n_samples: int = _DEFAULT_N_SAMPLES,
    **kwargs,
) -> Dict[str, Any]:
    """
    Bootstrap 置信区间验证。

    对回测收益序列进行 Bootstrap 重采样，估计 Sharpe 等指标的置信区间。
    优先使用系统 Bootstrap，失败则回退到 MetricsCalculator。

    Args:
        data_source: 数据源
        config: 回测配置（BacktestConfig）
        lib: 策略库
        output_dir: 输出目录
        best_params: 优化后的最优参数
        cross_sectional: 是否使用横截面打分模式
        n_samples: Bootstrap 采样次数

    Returns:
        {"sharpe_samples": ..., "confidence_intervals": ...} 字典
    """
    logger.info("Bootstrap 置信区间验证")
    strategy_names = config.strategy_names
    default_strategy = strategy_names[:1] if strategy_names else ["trend"]

    # 构建兼容的配置字典
    config_dict = {
        "backtest": {
            "initial_cash": config.initial_cash,
            "full_start_date": config.full_start,
            "full_end_date": config.full_end,
        },
        "symbols": config.symbols,
        "strategies": [{"name": s} for s in default_strategy],
    }

    runner = get_pybroker_runner(data_source, config_dict, strategies=default_strategy)
    result = safe_run_backtest(
        runner, config.full_start, config.full_end,
        "Bootstrap_base",
    )

    if result is None or result.equity_curve is None or result.equity_curve.empty:
        logger.warning("Bootstrap: 无净值数据，跳过")
        return {"sharpe_samples": [], "confidence_intervals": pd.DataFrame()}

    # 优先使用系统 Bootstrap
    bootstrap_result = _try_system_bootstrap(runner, n_samples)

    # 系统失败时回退到 MetricsCalculator
    if bootstrap_result is None:
        bootstrap_result = _try_metrics_calculator_bootstrap(result, n_samples)

    if bootstrap_result is None:
        return {"sharpe_samples": [], "confidence_intervals": pd.DataFrame()}

    sharpe_samples, df_ci = _process_bootstrap_result(bootstrap_result, output_dir)
    return {"sharpe_samples": sharpe_samples, "confidence_intervals": df_ci}


def _try_system_bootstrap(
    runner,
    n_samples: int,
) -> Optional[Any]:
    """
    尝试使用 PyBrokerBacktestRunner 的系统 Bootstrap。

    Args:
        runner: 回测运行器
        n_samples: 采样次数

    Returns:
        Bootstrap 结果，失败返回 None
    """
    try:
        result = runner.bootstrap_metrics(n_samples=n_samples)
        logger.info(f"  系统Bootstrap完成: {n_samples} 样本")
        return result
    except Exception as e:
        logger.warning(f"  系统Bootstrap失败: {e}, 回退到MetricsCalculator")
        return None


def _try_metrics_calculator_bootstrap(
    result,
    n_samples: int,
) -> Optional[Any]:
    """
    回退到 MetricsCalculator 的 Bootstrap。

    Args:
        result: 回测结果
        n_samples: 采样次数

    Returns:
        Bootstrap 结果，失败返回 None
    """
    try:
        from utils.metrics import MetricsCalculator
        equity = result.equity_curve["equity"].values
        bootstrap_result = MetricsCalculator.bootstrap_confidence_interval(
            equity, n_samples=n_samples,
        )
        logger.info(f"  MetricsCalculator Bootstrap完成: {n_samples} 样本")
        return bootstrap_result
    except Exception as e:
        logger.error(f"  MetricsCalculator Bootstrap也失败: {e}")
        return None


def _process_bootstrap_result(
    bootstrap_result: Any,
    output_dir: Path,
) -> Tuple[List[float], pd.DataFrame]:
    """
    处理 Bootstrap 结果，统一格式并保存。

    Args:
        bootstrap_result: 原始 Bootstrap 结果
        output_dir: 输出目录

    Returns:
        (sharpe_samples, 置信区间DataFrame)
    """
    # 结构化结果：{metric: {mean, ci_lower, ci_upper}}
    if isinstance(bootstrap_result, dict):
        first_val = next(iter(bootstrap_result.values()), None)
        if isinstance(first_val, dict) and "mean" in first_val:
            rows: List[Dict[str, Any]] = []
            for metric_name, vals in bootstrap_result.items():
                if isinstance(vals, dict):
                    rows.append({"metric": metric_name, **vals})
            df_ci = pd.DataFrame(rows)
            save_csv(df_ci, output_dir / "bootstrap_confidence_intervals.csv")
            sharpe_ci = bootstrap_result.get("sharpe", {})
            logger.info(f"  Bootstrap置信区间: {sharpe_ci}")
            return [], df_ci

        # 兼容老格式：直接取第一个列表值
        sharpe_samples: List[float] = []
        for val in bootstrap_result.values():
            if isinstance(val, list) and len(val) > 0:
                sharpe_samples = val
                break

        if sharpe_samples:
            df_samples = pd.DataFrame({"sharpe": sharpe_samples})
            save_csv(df_samples, output_dir / "bootstrap_samples.csv")
            return sharpe_samples, df_samples

    return [], pd.DataFrame()
