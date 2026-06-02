"""
实验模块：E1-E11 实验实现。

提供统一入口 run_experiment() 和 get_experiment_runner()，
供 Pipeline 编排器调用。
"""

from typing import Any, Callable, Dict, Optional

from runner.backtest.experiments.e1_e5 import (
    run_e1_single_strategy_baselines,
    run_e2_equal_weight,
    run_e3_dynamic_weight,
    run_e5_multi_symbol,
)
from runner.backtest.experiments.e6_e11 import (
    run_e6_walkforward,
    run_e7_out_of_sample,
    run_e8_bootstrap,
    run_e9_monte_carlo,
    run_e10_html_report,
    run_e11_factor_analysis,
)

_EXPERIMENT_MAP: Dict[str, Callable] = {
    "e1": run_e1_single_strategy_baselines,
    "e2": run_e2_equal_weight,
    "e3": run_e3_dynamic_weight,
    "e5": run_e5_multi_symbol,
    "e6": run_e6_walkforward,
    "e7": run_e7_out_of_sample,
    "e8": run_e8_bootstrap,
    "e9": run_e9_monte_carlo,
    "e10": run_e10_html_report,
    "e11": run_e11_factor_analysis,
}


def get_experiment_runner(name: str) -> Callable:
    """
    获取实验执行函数。

    Args:
        name: 实验名称（如 "e1", "e2"）

    Returns:
        实验执行函数

    Raises:
        ValueError: 实验名称不存在
    """
    name_lower = name.lower()
    if name_lower not in _EXPERIMENT_MAP:
        raise ValueError(f"未知实验: {name}，可用: {list(_EXPERIMENT_MAP.keys())}")
    return _EXPERIMENT_MAP[name_lower]


def run_experiment(
    name: str,
    config,
    data_source,
    raw_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    执行指定实验。

    Args:
        name: 实验名称或 "all"
        config: BacktestConfig 实例
        data_source: PyBrokerDataSource 实例
        raw_config: 原始配置字典（部分实验需要）

    Returns:
        实验结果
    """
    from pathlib import Path
    from runner.common.utils import save_csv

    if raw_config is None:
        raw_config = {}

    output_dir = Path(raw_config.get("output", {}).get("output_dir", "results"))
    output_dir.mkdir(exist_ok=True)

    if name.lower() == "all":
        results = {}
        for exp_name, func in _EXPERIMENT_MAP.items():
            if exp_name == "e10":
                continue
            try:
                results[exp_name] = func(data_source, raw_config, output_dir)
            except Exception as e:
                from loguru import logger

                logger.error(f"实验 {exp_name} 失败: {e}")
        return results
    else:
        func = get_experiment_runner(name)
        if name.lower() == "e10":
            return func(raw_config, {}, output_dir)
        return func(data_source, raw_config, output_dir)
