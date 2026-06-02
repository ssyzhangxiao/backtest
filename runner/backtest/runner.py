"""
回测执行封装模块。

委托 core/engine/backtest_runner.py 的 PyBrokerBacktestRunner，
不重新实现回测逻辑。
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from core.config import BacktestConfig
from core.engine.backtest_runner import (
    PyBrokerBacktestRunner,
    PyBrokerResult,
)
from core.engine.pybroker_data_source import PyBrokerDataSource
from runner.common.errors import BacktestError

# 回测配置缓存
_bt_config_cache: Dict[str, BacktestConfig] = {}


def build_backtest_config(config: Dict[str, Any]) -> BacktestConfig:
    """
    构建 BacktestConfig 对象（带缓存）。

    消除重复#7：直接从 yaml 字典构建，与 BacktestConfig.from_yaml() 对齐。

    Args:
        config: 完整配置字典

    Returns:
        BacktestConfig 实例
    """
    cache_key = f"{id(config)}"
    if cache_key in _bt_config_cache:
        return _bt_config_cache[cache_key]

    bt_cfg = config["backtest"]
    risk_cfg: Dict[str, Any] = config.get("risk_management", {})
    bt_config = BacktestConfig(
        initial_cash=bt_cfg.get("initial_cash", 1_000_000),
        commission_rate=bt_cfg.get("commission_rate", bt_cfg.get("commission", 0.0001)),
        slippage_rate=bt_cfg.get("slippage_rate", bt_cfg.get("slippage", 0.0001)),
        stop_loss_pct=risk_cfg.get("stop_loss_pct", bt_cfg.get("stop_loss_pct", 0.03)),
        max_position_pct=risk_cfg.get("position_limit_pct", bt_cfg.get("max_position_pct", 0.15)),
        max_total_position_pct=risk_cfg.get("total_position_limit", bt_cfg.get("max_total_position_pct", 0.6)),
        in_sample_end=bt_cfg.get("in_sample_end_date"),
        factor_weights=config.get("factor_weights", {}),
        entry_threshold=float(bt_cfg.get("entry_threshold", 0.05)),
        min_position_pct=float(bt_cfg.get("min_position_pct", 0.0)),
        use_cross_section=bool(bt_cfg.get("use_cross_section", True)),
        use_rank_score=bool(bt_cfg.get("use_rank_score", True)),
        use_rolling_ic=bool(bt_cfg.get("use_rolling_ic", True)),
        top_n_symbols=int(bt_cfg.get("top_n_symbols", 5)),
    )
    bt_config.rebalance_days = bt_cfg.get("rebalance_freq", 3)
    _bt_config_cache[cache_key] = bt_config
    return bt_config


def get_pybroker_runner(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    strategies: Optional[List[str]] = None,
) -> PyBrokerBacktestRunner:
    """
    创建 PyBroker 回测运行器。

    Args:
        data_source: 数据源
        config: 配置字典
        strategies: 策略名称列表

    Returns:
        PyBrokerBacktestRunner 实例
    """
    bt_config = build_backtest_config(config)
    symbols = config.get("symbols", [])
    runner = PyBrokerBacktestRunner(data_source, bt_config, target_symbols=symbols)
    if strategies:
        runner.register_strategies(strategies)
    return runner


def safe_run_backtest(
    runner: PyBrokerBacktestRunner,
    start_date: str,
    end_date: str,
    experiment_name: str,
    **kwargs,
) -> Optional[PyBrokerResult]:
    """
    安全执行回测，捕获异常并记录日志。

    Args:
        runner: 回测运行器
        start_date: 开始日期
        end_date: 结束日期
        experiment_name: 实验名称（用于日志）
        **kwargs: 传给 runner.run 的额外参数

    Returns:
        回测结果，失败返回 None
    """
    try:
        return runner.run(start_date=start_date, end_date=end_date, **kwargs)
    except Exception as e:
        logger.error(f"{experiment_name} 回测执行失败: {e}", exc_info=True)
        return None


def setup_logging(config: Dict[str, Any]) -> None:
    """配置日志系统，失败时降级为控制台输出。"""
    log_config: Dict[str, Any] = config.get("logging", {})
    if not log_config:
        logger.add(sys.stdout, level="INFO")
        return

    try:
        log_dir = Path(log_config.get("log_dir", "logs"))
        log_dir.mkdir(exist_ok=True)
        logger.remove()
        logger.add(
            log_dir / log_config.get("log_file", "backtest.log"),
            rotation=log_config.get("rotation", "100 MB"),
            retention=log_config.get("retention", "30 days"),
            level=log_config.get("log_level", "INFO"),
        )
        logger.add(log_dir / log_config.get("error_file", "error.log"), level="ERROR")
        logger.add(sys.stdout, level=log_config.get("log_level", "INFO"))
    except Exception as e:
        logger.add(sys.stdout, level="INFO")
        logger.warning(f"日志文件配置失败，仅使用控制台输出: {e}")


def run_experiment_safely(name: str, func, *args) -> Any:
    """
    安全执行单个实验，捕获异常不中断整体流程。

    Args:
        name: 实验名称
        func: 实验函数
        *args: 传递给实验函数的参数

    Returns:
        实验结果，失败返回 None
    """
    logger.info("=" * 60)
    logger.info(f"{name}")
    try:
        return func(*args)
    except Exception as e:
        logger.error(f"{name} 执行失败: {e}", exc_info=True)
        return None
