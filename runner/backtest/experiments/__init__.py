"""
实验模块：E1-E11 实验实现。

提供统一入口 run_experiment() 和 get_experiment_runner()，
供 Pipeline 编排器调用。
"""

from typing import Any, Callable, Dict, Optional

# 拆分后的 6 个实验模块（按工作内容命名，规则 8）：
# - e1_baselines          E1 单策略多品种基线
# - e2_e3_fusion          E2 等权融合 + E3 环境动态加权
# - e4_e5_portfolio       E4 风险平价 + E5 多品种分散
# - e6_e7_validation      E6 WalkForward + E7 样本外
# - e8_e9_resampling      E8 Bootstrap + E9 蒙特卡洛
# - e10_e11_reporting     E10 HTML 报告 + E11 因子分析
from runner.backtest.experiments.e1_baselines import (
    run_e1_single_strategy_baselines,
)
from runner.backtest.experiments.e2_e3_fusion import (
    run_e2_equal_weight,
    run_e3_dynamic_weight,
)
from runner.backtest.experiments.e4_e5_portfolio import (
    run_e4_risk_parity,
    run_e5_multi_symbol,
)
from runner.backtest.experiments.e6_e7_validation import (
    run_e6_walkforward,
    run_e7_out_of_sample,
)
from runner.backtest.experiments.e8_e9_resampling import (
    run_e8_bootstrap,
    run_e9_monte_carlo,
)
from runner.backtest.experiments.e10_e11_reporting import (
    run_e10_html_report,
    run_e11_factor_analysis,
)

_EXPERIMENT_MAP: Dict[str, Callable] = {
    "e1": run_e1_single_strategy_baselines,
    "e2": run_e2_equal_weight,
    "e3": run_e3_dynamic_weight,
    "e4": run_e4_risk_parity,
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
    cross_sectional: bool = False,
    strategy: Optional[str] = None,
) -> Any:
    """
    执行指定实验。

    Args:
        name: 实验名称或 "all"
        config: BacktestConfig 实例
        data_source: PyBrokerDataSource 实例
        raw_config: 原始配置字典（部分实验需要）
        cross_sectional: 是否启用多策略横截面打分模式
        strategy: 指定策略名称，None 表示自动选择

    Returns:
        实验结果
    """
    from pathlib import Path
    from runner.common.utils import save_csv

    if raw_config is None:
        raw_config = {}
    # 将模式参数注入 raw_config，供实验函数使用
    raw_config["_cross_sectional"] = cross_sectional
    raw_config["_strategy"] = strategy

    output_dir = Path(raw_config.get("output", {}).get("output_dir", "results"))
    output_dir.mkdir(exist_ok=True)

    if name.lower() == "all":
        results = {}
        for exp_name, func in _EXPERIMENT_MAP.items():
            if exp_name == "e10":
                continue
            try:
                # E9 需要 BacktestConfig 而非 raw_config dict
                if exp_name == "e9":
                    from runner.backtest.runner import build_backtest_config

                    bt_config = build_backtest_config(raw_config)
                    results[exp_name] = func(data_source, bt_config, output_dir)
                else:
                    results[exp_name] = func(data_source, raw_config, output_dir)
            except Exception as e:
                from loguru import logger

                logger.error(f"实验 {exp_name} 失败: {e}")
        return results
    else:
        func = get_experiment_runner(name)
        if name.lower() == "e10":
            return func(raw_config, {}, output_dir)
        if name.lower() == "e9":
            from runner.backtest.runner import build_backtest_config

            bt_config = build_backtest_config(raw_config)
            return func(data_source, bt_config, output_dir)
        return func(data_source, raw_config, output_dir)
