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
from core.execution.backtest_runner import (
    PyBrokerBacktestRunner,
    PyBrokerResult,
)
from core.engine.pybroker_data_source import PyBrokerDataSource


def build_backtest_config(config: Dict[str, Any]) -> BacktestConfig:
    """
    从完整配置字典构建 BacktestConfig 实例（无缓存，每次重新构造）。

    规则2：字段命名与 config.yaml 保持一致（initial_capital / commission_rate / slippage_rate）。

    Args:
        config: 完整配置字典（含 backtest / risk_management / factor_weights 等子段）

    Returns:
        BacktestConfig 实例
    """
    bt_cfg = config["backtest"]
    risk_cfg: Dict[str, Any] = config.get("risk_management", {})
    bt_config = BacktestConfig(
        initial_cash=bt_cfg.get("initial_capital", 1_000_000),
        commission_rate=bt_cfg.get("commission_rate", bt_cfg.get("commission", 0.0001)),
        slippage_rate=bt_cfg.get("slippage_rate", bt_cfg.get("slippage", 0.0001)),
        stop_loss_pct=risk_cfg.get("stop_loss_pct", bt_cfg.get("stop_loss_pct", 0.03)),
        max_position_pct=risk_cfg.get(
            "position_limit_pct", bt_cfg.get("max_position_pct", 0.15)
        ),
        max_total_position_pct=risk_cfg.get(
            "total_position_limit", bt_cfg.get("max_total_position_pct", 0.6)
        ),
        in_sample_end=bt_cfg.get("in_sample_end_date"),
        factor_weights=config.get("factor_weights", {}),
        entry_threshold=float(bt_cfg.get("entry_threshold", 0.05)),
        min_position_pct=float(bt_cfg.get("min_position_pct", 0.0)),
        use_cross_section=bool(bt_cfg.get("use_cross_section", True)),
        use_rank_score=bool(bt_cfg.get("use_rank_score", True)),
        use_rolling_ic=bool(bt_cfg.get("use_rolling_ic", True)),
    )
    bt_config.rebalance_days = bt_cfg.get("rebalance_freq", 3)
    return bt_config


def get_pybroker_runner(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    strategies: Optional[List[str]] = None,
    target_symbols: Optional[List[str]] = None,
) -> PyBrokerBacktestRunner:
    """
    创建 PyBroker 回测运行器。

    Args:
        data_source: 数据源
        config: 配置字典
        strategies: 策略名称列表
        target_symbols: 目标品种子集；None 时使用 config["symbols"]。
            外层 per-symbol 循环必须传入 [sym] 以避免跑全品种组合。

    Returns:
        PyBrokerBacktestRunner 实例
    """
    bt_config = build_backtest_config(config)
    # 修复 per-symbol 隔离 bug：
    # 外层循环传入 target_symbols=[sym] 时，仅对该品种做回测；
    # 不传时维持原行为，使用 config["symbols"] 全品种组合。
    symbols = (
        target_symbols if target_symbols is not None else config.get("symbols", [])
    )
    runner = PyBrokerBacktestRunner(data_source, bt_config, target_symbols=symbols)
    if strategies:
        runner.register_strategies(strategies)
        # 修复 best_params 失效 bug：把 config["strategies"][*].params 注入到
        # custom_params，让 PyBroker run() 的 sub_params.update(custom_params) 生效。
        custom = {}
        for sname in strategies:
            for s_cfg in config.get("strategies", []) or []:
                if s_cfg.get("name") == sname and s_cfg.get("params"):
                    custom[sname] = dict(s_cfg["params"])
                    break
        if custom:
            runner.set_custom_params(custom)  # type: ignore[attr-defined]
            logger.debug("已注入最优参数到 PyBroker: %s", list(custom.keys()))
    return runner


def safe_run_backtest(
    runner: PyBrokerBacktestRunner,
    start_date: str,
    end_date: str,
    experiment_name: str,
    initial_cash: Optional[float] = None,
    **kwargs,
) -> Optional[PyBrokerResult]:
    """
    安全执行回测，捕获异常并记录日志。

    Args:
        runner: 回测运行器
        start_date: 开始日期
        end_date: 结束日期
        experiment_name: 实验名称（用于日志）
        initial_cash: 可选的初始资金覆盖；为 None 时沿用 runner 配置
        **kwargs: 传给 runner.run 的额外参数

    Returns:
        回测结果，失败返回 None
    """
    try:
        run_kwargs = dict(kwargs)
        if initial_cash is not None:
            run_kwargs["initial_cash"] = initial_cash
        return runner.run(start_date=start_date, end_date=end_date, **run_kwargs)
    except Exception as e:
        # 2026-06-11 修复：使用 logger.exception() 等价于 error+exc_info+diagnose，
        # 确保完整 traceback 写入日志（不再因 exc_info=True 写法在某些 loguru
        # sink 配置下不输出堆栈）。同时把异常对象 raise_with_traceback 抛回，
        # 让编排层（experiments/）try/except 能拿到原始异常 + 完整 traceback。
        logger.exception(f"{experiment_name} 回测执行失败: {e}")
        return None


def setup_logging(config: Dict[str, Any]) -> None:
    """
    配置日志系统，失败时降级为控制台输出。

    Args:
        config: 完整配置字典（读取 logging 子段）
    """
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
