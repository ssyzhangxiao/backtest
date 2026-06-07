"""
网格搜索模块。

对单个策略执行参数网格搜索，返回按 Sharpe 排序的结果。
包含参数扰动稳定性测试。
"""

from itertools import product
from typing import Any, Dict

import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.engine.backtest_runner import PyBrokerBacktestRunner
from core.config.strategy_profiles import StrategyLibrary
from runner.optimization import copy_config
from runner.optimization.window_search import _MAX_OPT_PROGRESS


def grid_search_single_strategy(
    strategy_name: str,
    param_space: Dict[str, Any],
    ds,
    lib: StrategyLibrary,
    config: BacktestConfig,
) -> pd.DataFrame:
    """
    对单个策略执行参数网格搜索，返回按 Sharpe 排序的结果。

    包含参数扰动稳定性测试，复合排序：Sharpe * (0.4 + 0.6 * stability)。

    Args:
        strategy_name: 策略名称
        param_space: 参数搜索空间
        ds: 数据源
        lib: 策略库
        config: 回测配置（BacktestConfig）

    Returns:
        按复合得分排序的结果 DataFrame
    """
    keys = list(param_space.keys())
    values = list(param_space.values())
    combos = list(product(*values))
    total = len(combos)

    logger.info(f"\n  策略: {strategy_name} | 参数组合数: {total}")

    # 构建回测用的 BacktestConfig（复用传入的 config，仅覆盖交易参数）
    bt_config = BacktestConfig(
        initial_cash=config.initial_cash,
        commission_rate=config.commission_rate,
        slippage_rate=config.slippage_rate,
    )

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # 期限结构策略的入场/出场阈值合法性检查
        if strategy_name == "term_structure" and params.get("entry_threshold", 0) <= params.get("exit_threshold", 999):
            continue

        try:
            runner = PyBrokerBacktestRunner(ds, bt_config)
            runner.register_strategies([strategy_name])

            result = runner.run(
                config.full_start,
                config.in_sample_end,
                custom_params={strategy_name: params},
            )
            kpi = dict(result.metrics)
            kpi.update(params)
            results.append(kpi)

        except Exception as e:
            logger.debug(f"    组合 {params} 失败: {e}")
            continue

        if (i + 1) % _MAX_OPT_PROGRESS == 0 or i + 1 == total:
            logger.info(f"    进度: {i+1}/{total}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Top N 参数做扰动测试
    stability_scores = {}
    top_n = min(10, len(df))
    df_temp = df.sort_values("sharpe", ascending=False).head(top_n)
    for _, row in df_temp.iterrows():
        params_base = {k: row[k] for k in keys if k in row}
        score = param_stability_test(strategy_name, params_base, ds, lib, config)
        param_key = _params_to_key(params_base)
        stability_scores[param_key] = score

    def _get_stability(row):
        pk = _params_to_key({k: row[k] for k in keys if k in row})
        return stability_scores.get(pk, 0.5)

    df["stability"] = df.apply(_get_stability, axis=1)
    df["composite_score"] = df["sharpe"] * (0.4 + 0.6 * df["stability"])
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df


def _params_to_key(params: Dict[str, Any]) -> str:
    """将参数字典转为排序后的键值字符串。"""
    return ",".join(f"{k}={params[k]}" for k in sorted(params.keys()))


def param_stability_test(
    strategy_name: str,
    params: Dict[str, Any],
    ds,
    lib: StrategyLibrary,
    config: BacktestConfig,
    perturb_ratio: float = 0.10,
) -> float:
    """
    参数扰动测试：每个参数 ±10%，看 Sharpe 变化幅度。

    Args:
        strategy_name: 策略名
        params: 基准参数
        ds: 数据源
        lib: 策略库
        config: 回测配置（BacktestConfig）
        perturb_ratio: 扰动比例

    Returns:
        稳定性分数 (0~1)，1 表示最稳定
    """
    base_sharpe = None
    sharpe_changes = []

    # 构建回测用的 BacktestConfig（完整复制所有字段）
    bt_config = copy_config(config)

    try:
        runner = PyBrokerBacktestRunner(ds, bt_config)
        runner.register_strategies([strategy_name])
        result = runner.run(
            config.full_start, config.in_sample_end,
            custom_params={strategy_name: params},
        )
        base_sharpe = float(result.metrics.get("sharpe", 0))
    except Exception:
        return 0.3

    if base_sharpe is None or abs(base_sharpe) < 1e-8:
        return 0.3

    for key in list(params.keys()):
        orig = params[key]
        if not isinstance(orig, (int, float)):
            continue
        delta = abs(orig) * perturb_ratio
        for direction in [1, -1]:
            try:
                test_params = dict(params)
                test_params[key] = orig + delta * direction
                if strategy_name == "term_structure":
                    if test_params.get("entry_threshold", 0) <= test_params.get("exit_threshold", 999):
                        continue

                runner2 = PyBrokerBacktestRunner(ds, bt_config)
                runner2.register_strategies([strategy_name])
                result2 = runner2.run(
                    config.full_start, config.in_sample_end,
                    custom_params={strategy_name: test_params},
                )
                test_sharpe = float(result2.metrics.get("sharpe", 0))
                change = abs(test_sharpe - base_sharpe) / max(abs(base_sharpe), 1e-8)
                sharpe_changes.append(change)
            except Exception:
                pass

    if not sharpe_changes:
        return 0.5

    avg_change = sum(sharpe_changes) / len(sharpe_changes)
    stability = max(0.0, min(1.0, 1.0 - avg_change / 0.5))
    return stability
