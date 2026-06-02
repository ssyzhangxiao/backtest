"""
窗口搜索模块。

P0-A: 对 Top N 参数组合，遍历不同训练窗口长度做 WalkForward。
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.engine.backtest_runner import PyBrokerBacktestRunner
from core.strategy_registry import StrategyLibrary
from runner.common.utils import safe_float

# P0-A: 窗口搜索网格
_INSAMPLE_BAR_RANGE = list(range(126, 505, 63))
_TEST_TRAIN_RATIO = 1 / 4
_STEP_TEST_RATIO = 1
_MAX_OPT_PROGRESS = 5


def window_search_single_strategy(
    strategy_name: str,
    top_params_list: List[Dict[str, Any]],
    ds,
    lib: StrategyLibrary,
    opt_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    对 Top N 参数组合，遍历不同训练窗口长度做 WalkForward。

    窗口配置：
      - 训练窗口: 126, 189, 252, 315, 378, 441, 504（步长63）
      - 测试窗口: 训练窗口 / 4
      - 步进: 测试窗口长度

    Args:
        strategy_name: 策略名
        top_params_list: Top N 参数组合列表
        ds: 数据源
        lib: 策略库
        opt_cfg: 优化配置

    Returns:
        DataFrame，每行 = (参数组合, 窗口配置, WF指标)
    """
    logger.info(f"\n  窗口搜索: {strategy_name} | 参数组合数: {len(top_params_list)} | 窗口配置数: {len(_INSAMPLE_BAR_RANGE)}")

    results = []
    total = len(top_params_list) * len(_INSAMPLE_BAR_RANGE)
    progress = 0

    for params in top_params_list:
        for train_bars in _INSAMPLE_BAR_RANGE:
            test_bars = max(5, int(train_bars * _TEST_TRAIN_RATIO))
            step_bars = max(5, int(test_bars * _STEP_TEST_RATIO))

            progress += 1
            try:
                config = BacktestConfig(
                    initial_cash=opt_cfg["initial_cash"],
                    commission_rate=opt_cfg["commission_rate"],
                    slippage_rate=opt_cfg["slippage_rate"],
                    wf_train_bars=train_bars,
                    wf_test_bars=test_bars,
                    wf_step_bars=step_bars,
                )
                runner = PyBrokerBacktestRunner(ds, config)
                runner.register_strategies([strategy_name])

                wf_result = runner.walkforward(
                    opt_cfg["full_start"],
                    opt_cfg["in_sample_end"],
                )

                window_sharpes = []
                window_returns = []
                for w in wf_result.windows:
                    m = w.get("metrics", {})
                    s = safe_float(m.get("sharpe"))
                    if s is not None:
                        window_sharpes.append(s)
                    r = safe_float(m.get("total_return_pct"))
                    if r is not None:
                        window_returns.append(r)

                row = {
                    "train_bars": train_bars,
                    "test_bars": test_bars,
                    "step_bars": step_bars,
                    "n_windows": len(wf_result.windows),
                    "wf_avg_sharpe": np.mean(window_sharpes) if window_sharpes else 0.0,
                    "wf_min_sharpe": min(window_sharpes) if window_sharpes else 0.0,
                    "wf_avg_return": np.mean(window_returns) if window_returns else 0.0,
                    "wf_positive_ratio": sum(1 for s in window_sharpes if s > 0) / max(len(window_sharpes), 1),
                }
                row.update(params)
                results.append(row)

            except Exception as e:
                logger.debug(f"    窗口搜索失败 (train={train_bars}): {e}")
                continue

            if progress % _MAX_OPT_PROGRESS == 0 or progress == total:
                logger.info(f"    窗口搜索进度: {progress}/{total}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("wf_avg_sharpe", ascending=False).reset_index(drop=True)
    return df


def rolling_validate(
    strategy_name: str,
    best_params: Dict[str, Any],
    ds,
    lib: StrategyLibrary,
    opt_cfg: Dict[str, Any],
    train_bars: int = 252,
    test_bars: int = 63,
    step_bars: int = 21,
) -> Dict[str, Any]:
    """
    使用指定窗口配置做 WalkForward 验证。

    Args:
        strategy_name: 策略名
        best_params: 最优参数
        ds: 数据源
        lib: 策略库
        opt_cfg: 优化配置
        train_bars: 训练窗口长度
        test_bars: 测试窗口长度
        step_bars: 步进长度

    Returns:
        验证结果字典
    """
    logger.info(f"\n  滚动窗口验证: {strategy_name} | train={train_bars}, test={test_bars}, step={step_bars}")

    try:
        config = BacktestConfig(
            initial_cash=opt_cfg["initial_cash"],
            commission_rate=opt_cfg["commission_rate"],
            slippage_rate=opt_cfg["slippage_rate"],
            wf_train_bars=train_bars,
            wf_test_bars=test_bars,
            wf_step_bars=step_bars,
        )
        runner = PyBrokerBacktestRunner(ds, config)
        runner.register_strategies([strategy_name])

        wf_result = runner.walkforward(
            opt_cfg["full_start"],
            opt_cfg["in_sample_end"],
        )

        window_sharpes = []
        window_returns = []
        for w in wf_result.windows:
            m = w.get("metrics", {})
            if "sharpe" in m and safe_float(m["sharpe"]) is not None:
                window_sharpes.append(safe_float(m["sharpe"]))
            if "total_return_pct" in m and safe_float(m["total_return_pct"]) is not None:
                window_returns.append(safe_float(m["total_return_pct"]))

        result = {
            "n_windows": len(wf_result.windows),
            "avg_sharpe": np.mean(window_sharpes) if window_sharpes else 0.0,
            "avg_return_pct": np.mean(window_returns) if window_returns else 0.0,
            "min_sharpe": min(window_sharpes) if window_sharpes else 0.0,
            "sharpe_std": np.std(window_sharpes) if len(window_sharpes) > 1 else 0.0,
            "positive_sharpe_ratio": sum(1 for s in window_sharpes if s > 0) / max(len(window_sharpes), 1),
        }

        logger.info(f"    窗口数: {result['n_windows']}")
        logger.info(f"    平均Sharpe: {result['avg_sharpe']:.4f}")
        logger.info(f"    正Sharpe比例: {result['positive_sharpe_ratio']:.1%}")

        return result

    except Exception as e:
        logger.error(f"    滚动验证失败: {e}")
        return {}


def out_of_sample_test(
    strategy_name: str,
    best_params: Dict[str, Any],
    ds,
    lib: StrategyLibrary,
    opt_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    样本外测试。

    Args:
        strategy_name: 策略名
        best_params: 最优参数
        ds: 数据源
        lib: 策略库
        opt_cfg: 优化配置

    Returns:
        样本外 KPI 字典
    """
    logger.info(f"\n  样本外测试: {strategy_name}")

    try:
        config = BacktestConfig(
            initial_cash=opt_cfg["initial_cash"],
            commission_rate=opt_cfg["commission_rate"],
            slippage_rate=opt_cfg["slippage_rate"],
        )
        runner = PyBrokerBacktestRunner(ds, config)
        runner.register_strategies([strategy_name])

        result = runner.run(
            opt_cfg.get("out_sample_start", "2024-01-01"),
            opt_cfg["full_end"],
            custom_params={strategy_name: best_params},
        )
        kpi = dict(result.metrics)

        logger.info(f"    样本外收益: {safe_float(kpi.get('total_return_pct', 0)):.2f}%")
        logger.info(f"    样本外Sharpe: {safe_float(kpi.get('sharpe', 0)):.4f}")
        logger.info(f"    样本外回撤: {safe_float(kpi.get('max_drawdown_pct', 0)):.2f}%")

        return kpi

    except Exception as e:
        logger.error(f"    样本外测试失败: {e}")
        return {}
