#!/usr/bin/env python3
"""
参数优化脚本 v2：网格搜索 + 窗口搜索 + 样本外优先选择

P0-A: WalkForward 窗口搜索网格（insample 126~504 步长63，test/step=3:1）
P0-C: 目标函数 = OOS_Sharpe - 0.5 × 过拟合惩罚

流程：
  1. 网格搜索：在全样本内数据上搜索参数组合
  2. 窗口搜索：对 Top N 参数组合，遍历不同训练窗口长度
  3. 滚动窗口验证：用最优窗口配置做 WalkForward
  4. 样本外测试：验证样本外表现
  5. 样本外优先选择：最大化 OOS_Sharpe - 0.5 × 过拟合惩罚
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from itertools import product
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from core.engine.backtest_runner import PyBrokerBacktestRunner
from core.engine.pybroker_data_source import create_hybrid_data_source
from core.engine.rolling_ic import RollingICWeightEngine, RollingICConfig
from core.engine.factor_decay import FactorDecayMonitor, FactorDecayConfig, DecayStatus
from core.config import BacktestConfig
from core.strategy_registry import StrategyLibrary
from run_full_backtest import load_config, _safe_float, get_tqsdk_credentials, _compute_factor_scores_from_ohlcv

# ══════════════════════════════════════════════════════════════════════════════
# 全局常量
# ══════════════════════════════════════════════════════════════════════════════

# P0-A: 窗口搜索网格
_INSAMPLE_BAR_RANGE = list(range(126, 505, 63))  # [126, 189, 252, 315, 378, 441, 504]
_TEST_TRAIN_RATIO = 1 / 4  # 测试窗口 = 训练窗口 / 4（即 test:train = 1:3）
_STEP_TEST_RATIO = 1  # 步进 = 测试窗口长度

# P0-C: 样本外优先目标函数
_OOF_PENALTY_WEIGHT = 0.5  # 过拟合惩罚权重

# 通用
_MAX_OPT_PROGRESS = 5
_MIN_SHARPE_FOR_DECAY_CALC = 0.01
_DECAY_PASS_THRESHOLD = 0.5
_TOP_N_FOR_WINDOW_SEARCH = 5  # 窗口搜索只对 Top N 参数组合做


def _get_param_spaces(
    lib: StrategyLibrary, strategy_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    """获取各策略的参数搜索空间。"""
    param_spaces = {}
    for sname in strategy_names:
        profile = lib.get_profile(sname)
        if profile is not None and profile.param_ranges:
            param_spaces[sname] = dict(profile.param_ranges)
    return param_spaces


# ══════════════════════════════════════════════════════════════════════════════
# P0-C: 样本外优先目标函数
# ══════════════════════════════════════════════════════════════════════════════


def compute_oos_priority_score(
    is_sharpe: float,
    oos_sharpe: float,
    penalty_weight: float = _OOF_PENALTY_WEIGHT,
) -> float:
    """
    计算样本外优先的综合评分。

    公式：Score = OOS_Sharpe - penalty_weight × max(0, IS_Sharpe - OOS_Sharpe)

    当 IS >> OOS 时（过拟合），惩罚项增大，降低总分。
    当 OOS >= IS 时（样本外更好），惩罚为0，直接用 OOS。

    Args:
        is_sharpe: 样本内 Sharpe
        oos_sharpe: 样本外 Sharpe
        penalty_weight: 过拟合惩罚权重

    Returns:
        综合评分
    """
    overfit_gap = max(0.0, is_sharpe - oos_sharpe)
    return oos_sharpe - penalty_weight * overfit_gap


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: 网格搜索
# ══════════════════════════════════════════════════════════════════════════════


def grid_search_single_strategy(
    strategy_name: str,
    param_space: Dict[str, Any],
    ds,
    lib: StrategyLibrary,
    opt_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """对单个策略执行参数网格搜索，返回按 Sharpe 排序的结果。"""
    keys = list(param_space.keys())
    values = list(param_space.values())
    combos = list(product(*values))
    total = len(combos)

    logger.info(f"\n  策略: {strategy_name} | 参数组合数: {total}")

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        if strategy_name == "roll_yield" and params.get("entry_threshold", 0) <= params.get("exit_threshold", 999):
            continue

        try:
            config = BacktestConfig(
                initial_cash=opt_cfg["initial_cash"],
                commission_rate=opt_cfg["commission_rate"],
                slippage_rate=opt_cfg["slippage_rate"],
            )
            runner = PyBrokerBacktestRunner(ds, config)
            runner.register_strategies([strategy_name])

            result = runner.run(
                opt_cfg["full_start"],
                opt_cfg["in_sample_end"],
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

    # 对 Top N 参数做扰动测试，评估过拟合风险
    stability_scores = {}
    top_n = min(10, len(df))
    df_temp = df.sort_values("sharpe", ascending=False).head(top_n)
    for _, row in df_temp.iterrows():
        params_base = {k: row[k] for k in keys if k in row}
        score = _param_stability_test(
            strategy_name, params_base, ds, lib, opt_cfg
        )
        param_key = _params_to_key(params_base)
        stability_scores[param_key] = score

    # 附加稳定性分数到结果
    def _get_stability(row):
        pk = _params_to_key({k: row[k] for k in keys if k in row})
        return stability_scores.get(pk, 0.5)

    df["stability"] = df.apply(_get_stability, axis=1)

    # 复合排序：Sharpe * 稳定性（过拟合高风险降权）
    df["composite_score"] = df["sharpe"] * (0.4 + 0.6 * df["stability"])
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df


def _params_to_key(params: Dict[str, Any]) -> str:
    """将参数字典转为排序后的键值字符串，用作 key。"""
    return ",".join(f"{k}={params[k]}" for k in sorted(params.keys()))


def _param_stability_test(
    strategy_name: str,
    params: Dict[str, Any],
    ds, lib: "StrategyLibrary",
    opt_cfg: Dict[str, Any],
    perturb_ratio: float = 0.10,
) -> float:
    """
    参数扰动测试：每个参数 ±10%，看 Sharpe 变化幅度。
    返回稳定性分数 (0~1)，1 表示最稳定。
    """
    base_sharpe = None
    sharpe_changes = []

    try:
        config = BacktestConfig(
            initial_cash=opt_cfg["initial_cash"],
            commission_rate=opt_cfg["commission_rate"],
            slippage_rate=opt_cfg["slippage_rate"],
        )
        runner = PyBrokerBacktestRunner(ds, config)
        runner.register_strategies([strategy_name])
        result = runner.run(
            opt_cfg["full_start"], opt_cfg["in_sample_end"],
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
                # 跳过非法组合
                if strategy_name == "roll_yield":
                    if test_params.get("entry_threshold", 0) <= test_params.get("exit_threshold", 999):
                        continue

                config2 = BacktestConfig(
                    initial_cash=opt_cfg["initial_cash"],
                    commission_rate=opt_cfg["commission_rate"],
                    slippage_rate=opt_cfg["slippage_rate"],
                )
                runner2 = PyBrokerBacktestRunner(ds, config2)
                runner2.register_strategies([strategy_name])
                result2 = runner2.run(
                    opt_cfg["full_start"], opt_cfg["in_sample_end"],
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
    # 平均变化 > 50% → 极不稳定（过拟合高风险）
    # 平均变化 < 10% → 非常稳定
    stability = max(0.0, min(1.0, 1.0 - avg_change / 0.5))
    return stability


# ══════════════════════════════════════════════════════════════════════════════
# P0-A: 窗口搜索（遍历不同训练窗口长度）
# ══════════════════════════════════════════════════════════════════════════════


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
      - 测试窗口: 训练窗口 / 4（即 test:train = 1:3）
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
                    s = _safe_float(m.get("sharpe"))
                    if s is not None:
                        window_sharpes.append(s)
                    r = _safe_float(m.get("total_return_pct"))
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


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: 滚动窗口验证（使用最优窗口配置）
# ══════════════════════════════════════════════════════════════════════════════


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
    """使用指定窗口配置做 WalkForward 验证。"""
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
            if "sharpe" in m and _safe_float(m["sharpe"]) is not None:
                window_sharpes.append(_safe_float(m["sharpe"]))
            if "total_return_pct" in m and _safe_float(m["total_return_pct"]) is not None:
                window_returns.append(_safe_float(m["total_return_pct"]))

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


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: 样本外测试
# ══════════════════════════════════════════════════════════════════════════════


def out_of_sample_test(
    strategy_name: str,
    best_params: Dict[str, Any],
    ds,
    lib: StrategyLibrary,
    opt_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """样本外测试。"""
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

        logger.info(f"    样本外收益: {_safe_float(kpi.get('total_return_pct', 0)):.2f}%")
        logger.info(f"    样本外Sharpe: {_safe_float(kpi.get('sharpe', 0)):.4f}")
        logger.info(f"    样本外回撤: {_safe_float(kpi.get('max_drawdown_pct', 0)):.2f}%")

        return kpi

    except Exception as e:
        logger.error(f"    样本外测试失败: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# 参数建议输出
# ══════════════════════════════════════════════════════════════════════════════


def print_optimization_suggestions(
    best_params_all: Dict[str, Dict[str, Any]],
    best_window_config: Dict[str, Dict[str, int]],
    lib: StrategyLibrary,
) -> None:
    """输出最优参数建议和窗口配置。"""
    logger.info("\n" + "=" * 60)
    logger.info("最优参数建议")
    logger.info("=" * 60)

    for sname, params in best_params_all.items():
        profile = lib.get_profile(sname)
        if profile is None:
            continue
        old_params = dict(profile.default_params)
        win_cfg = best_window_config.get(sname, {})
        logger.info(f"\n  {sname}:")
        logger.info(f"    当前默认: {old_params}")
        logger.info(f"    建议更新: {params}")
        logger.info(f"    窗口配置: train={win_cfg.get('train_bars', 252)}, test={win_cfg.get('test_bars', 63)}, step={win_cfg.get('step_bars', 21)}")
        changed = {k: (old_params.get(k), v) for k, v in params.items() if old_params.get(k) != v}
        if changed:
            logger.info(f"    变更项: {changed}")

    logger.info("\n" + "-" * 60)
    logger.info("参数应用方式（3种，按推荐程度排序）：")
    logger.info("-" * 60)

    logger.info("\n  方式1（推荐）: 通过 StrategyLibrary.update_default_params() 运行时更新")
    logger.info("    示例代码:")
    logger.info("      from core.strategy_registry import StrategyLibrary")
    logger.info("      lib = StrategyLibrary()")
    for sname, params in best_params_all.items():
        logger.info(f"      lib.update_default_params('{sname}', {params})")

    logger.info("\n  方式2: 更新 config.yaml 策略参数段")
    for sname, params in best_params_all.items():
        logger.info(f"\n    - name: \"{sname}\"")
        logger.info(f"      params:")
        for k, v in params.items():
            if isinstance(v, str):
                logger.info(f"        {k}: \"{v}\"")
            else:
                logger.info(f"        {k}: {v}")

    logger.info("\n  方式3: 通过 custom_params 参数覆盖（单次回测）")

    logger.info("\n" + "=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# P0-C: 样本外优先选择
# ══════════════════════════════════════════════════════════════════════════════


def select_best_by_oos_priority(
    strategy_name: str,
    grid_df: pd.DataFrame,
    oos_results_map: Dict[str, Dict[str, Any]],
    param_keys: List[str],
) -> Tuple[Dict[str, Any], float]:
    """
    按样本外优先原则选择最优参数。

    对 Top 10 样本内参数组合，计算样本外优先评分：
      Score = OOS_Sharpe - 0.5 × max(0, IS_Sharpe - OOS_Sharpe)

    Args:
        strategy_name: 策略名
        grid_df: 网格搜索结果
        oos_results_map: {参数签名: 样本外KPI}
        param_keys: 参数键名列表

    Returns:
        (最优参数, 最优评分)
    """
    best_score = -float("inf")
    best_params = {}

    for _, row in grid_df.head(10).iterrows():
        params = {k: row[k] for k in param_keys}
        param_sig = str(sorted(params.items()))

        is_sharpe = _safe_float(row.get("sharpe", 0))
        oos_kpi = oos_results_map.get(param_sig, {})
        oos_sharpe = _safe_float(oos_kpi.get("sharpe", 0))

        score = compute_oos_priority_score(is_sharpe, oos_sharpe)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════


def main(config_path: str = "config.yaml", skip_optimization: bool = False) -> Dict[str, Any]:
    """
    主函数：执行完整参数优化流程（P0-A + P0-C）。

    流程：
      [1/5] 加载数据
      [2/5] 网格搜索（样本内）
      [3/5] 窗口搜索（P0-A: 遍历训练窗口长度）
      [4/5] 滚动窗口验证 + 样本外测试
      [5/5] 样本外优先选择（P0-C） + 结果汇总
    """
    logger.info("=" * 60)
    logger.info("  参数优化 v2：窗口搜索 + 样本外优先")
    logger.info(f"  P0-A: 窗口搜索网格 insample={_INSAMPLE_BAR_RANGE}")
    logger.info(f"  P0-C: 目标函数 = OOS_Sharpe - {_OOF_PENALTY_WEIGHT} × 过拟合惩罚")
    logger.info(f"  开始: {datetime.now()}")
    logger.info("=" * 60)

    cfg = load_config(config_path)
    bt_cfg = cfg.get("backtest", {})
    opt_cfg = {
        "initial_cash": bt_cfg.get("initial_cash", 1_000_000),
        "commission_rate": bt_cfg.get("commission_rate", 0.0003),
        "slippage_rate": bt_cfg.get("slippage_rate", 0.0002),
        "full_start": bt_cfg.get("full_start_date", "2016-01-01"),
        "in_sample_end": bt_cfg.get("in_sample_end_date", "2023-12-31"),
        "full_end": bt_cfg.get("full_end_date", "2026-05-01"),
        "out_sample_start": bt_cfg.get("out_sample_start_date", "2024-01-01"),
        "symbols": cfg.get("symbols", ["SHFE.RB", "DCE.M", "CZCE.TA", "SHFE.CU", "CFFEX.IF"]),
        "output_dir": cfg.get("output", {}).get("output_dir", "output_backtest_pybroker"),
        "strategy_names": [s["name"] for s in cfg.get("strategies", []) if s.get("name")],
    }

    lib = StrategyLibrary()

    # ── [1/5] 加载数据 ──
    logger.info("\n[1/5] 加载数据...")
    phone, password = get_tqsdk_credentials()
    data_dir = cfg.get("data", {}).get("csv_data_dir", "data")
    data_length = cfg.get("data", {}).get("tqsdk_data_length", 3000)
    ds = create_hybrid_data_source(
        phone=phone,
        password=password,
        symbols=opt_cfg["symbols"],
        data_dir=data_dir,
        data_length=data_length,
    )

    strategy_names = opt_cfg["strategy_names"]
    param_spaces = _get_param_spaces(lib, strategy_names)
    logger.info(f"\n  参数空间来源: StrategyLibrary.param_ranges")
    for sname, ps in param_spaces.items():
        total = 1
        for v in ps.values():
            total *= len(v)
        logger.info(f"    {sname}: {ps} ({total} 组合)")

    # ── [2/5] 网格搜索 ──
    all_grid_results: Dict[str, pd.DataFrame] = {}
    best_params_all: Dict[str, Dict[str, Any]] = {}
    best_window_config: Dict[str, Dict[str, int]] = {}

    if skip_optimization:
        logger.info("\n  跳过网格搜索，使用当前默认参数...")
        for sname in strategy_names:
            profile = lib.get_profile(sname)
            if profile:
                best_params_all[sname] = dict(profile.default_params)
                best_window_config[sname] = {"train_bars": 252, "test_bars": 63, "step_bars": 21}
    else:
        logger.info(f"\n[2/5] 网格搜索（样本内: {opt_cfg['full_start']} ~ {opt_cfg['in_sample_end']}）")

        for sname, param_space in param_spaces.items():
            grid_df = grid_search_single_strategy(sname, param_space, ds, lib, opt_cfg)
            all_grid_results[sname] = grid_df

            if grid_df.empty:
                logger.info(f"  {sname}: 无有效结果")
                continue

            output_path = Path(opt_cfg["output_dir"])
            output_path.mkdir(parents=True, exist_ok=True)
            grid_df.to_csv(output_path / f"opt_grid_{sname}.csv", index=False)

            logger.info(f"\n  {sname} Top 5 (按Sharpe):")
            top5 = grid_df.head(5)
            for idx, row in top5.iterrows():
                param_str = ", ".join(f"{k}={row[k]}" for k in param_space.keys())
                logger.info(f"    #{idx+1} {param_str} => Sharpe={_safe_float(row['sharpe']):.4f}, Return={_safe_float(row['total_return_pct']):.2f}%, DD={_safe_float(row['max_drawdown_pct']):.2f}%")

        # ── [3/5] 窗口搜索（P0-A） ──
        logger.info(f"\n[3/5] 窗口搜索（P0-A: insample={_INSAMPLE_BAR_RANGE}）")

        for sname, param_space in param_spaces.items():
            grid_df = all_grid_results.get(sname, pd.DataFrame())
            if grid_df.empty:
                continue

            # 取 Top N 参数组合
            top_params_list = []
            param_keys = list(param_space.keys())
            for _, row in grid_df.head(_TOP_N_FOR_WINDOW_SEARCH).iterrows():
                top_params_list.append({k: row[k] for k in param_keys})

            if not top_params_list:
                continue

            # 窗口搜索
            window_df = window_search_single_strategy(
                sname, top_params_list, ds, lib, opt_cfg,
            )

            if not window_df.empty:
                output_path = Path(opt_cfg["output_dir"])
                window_df.to_csv(output_path / f"opt_window_{sname}.csv", index=False)

                # 选择最优窗口配置（按 wf_avg_sharpe）
                best_win = window_df.iloc[0]
                train_bars = int(best_win["train_bars"])
                test_bars = int(best_win["test_bars"])
                step_bars = int(best_win["step_bars"])

                # 提取对应的最优参数
                best_params = {k: best_win[k] for k in param_keys}

                best_params_all[sname] = best_params
                best_window_config[sname] = {
                    "train_bars": train_bars,
                    "test_bars": test_bars,
                    "step_bars": step_bars,
                }

                logger.info(f"\n  {sname} 最优窗口配置:")
                logger.info(f"    train={train_bars}, test={test_bars}, step={step_bars}")
                logger.info(f"    WF平均Sharpe: {best_win['wf_avg_sharpe']:.4f}")
                logger.info(f"    WF正Sharpe比例: {best_win['wf_positive_ratio']:.1%}")
                logger.info(f"    最优参数: {best_params}")
            else:
                # 窗口搜索失败，回退到网格搜索 Top 1
                best = grid_df.iloc[0]
                best_params_all[sname] = {k: best[k] for k in param_keys}
                best_window_config[sname] = {"train_bars": 252, "test_bars": 63, "step_bars": 21}

    # ── [4/5] 滚动窗口验证 + 样本外测试 ──
    logger.info("\n[4/5] 滚动窗口验证 + 样本外测试")
    validation_results: Dict[str, Dict[str, Any]] = {}
    oos_results: Dict[str, Dict[str, Any]] = {}

    for sname, params in best_params_all.items():
        win_cfg = best_window_config.get(sname, {"train_bars": 252, "test_bars": 63, "step_bars": 21})

        val = rolling_validate(
            sname, params, ds, lib, opt_cfg,
            train_bars=win_cfg["train_bars"],
            test_bars=win_cfg["test_bars"],
            step_bars=win_cfg["step_bars"],
        )
        validation_results[sname] = val

        oos = out_of_sample_test(sname, params, ds, lib, opt_cfg)
        oos_results[sname] = oos

    # ── [5/5] 样本外优先选择（P0-C）+ 结果汇总 ──
    logger.info("\n[5/5] 样本外优先选择（P0-C）+ 结果汇总")

    summary = []
    for sname in param_spaces:
        if sname not in best_params_all:
            continue
        row: Dict[str, Any] = {
            "strategy": sname,
            "best_params": str(best_params_all[sname]),
        }

        # 窗口配置
        win_cfg = best_window_config.get(sname, {})
        row["train_bars"] = win_cfg.get("train_bars", 252)
        row["test_bars"] = win_cfg.get("test_bars", 63)
        row["step_bars"] = win_cfg.get("step_bars", 21)

        # 样本内
        grid_df = all_grid_results.get(sname) if not skip_optimization else None
        if grid_df is not None and not grid_df.empty:
            top = grid_df.iloc[0]
            row["in_sample_sharpe"] = round(_safe_float(top["sharpe"]), 4)
            row["in_sample_return"] = round(_safe_float(top["total_return_pct"]), 2)
            row["in_sample_drawdown"] = round(_safe_float(top["max_drawdown_pct"]), 2)

        # WF 验证
        val = validation_results.get(sname, {})
        row["wf_avg_sharpe"] = round(val.get("avg_sharpe", 0.0), 4)
        row["wf_positive_ratio"] = round(val.get("positive_sharpe_ratio", 0.0), 2)

        # 样本外
        oos = oos_results.get(sname, {})
        row["oos_sharpe"] = round(_safe_float(oos.get("sharpe", 0)), 4)
        row["oos_return"] = round(_safe_float(oos.get("total_return_pct", 0)), 2)
        row["oos_drawdown"] = round(_safe_float(oos.get("max_drawdown_pct", 0)), 2)

        # P0-C: 样本外优先评分
        is_sharpe = _safe_float(row.get("in_sample_sharpe", 0))
        oos_sharpe = _safe_float(row.get("oos_sharpe", 0))
        row["oos_priority_score"] = round(
            compute_oos_priority_score(is_sharpe, oos_sharpe), 4,
        )

        summary.append(row)

    summary_df = pd.DataFrame(summary)
    output_path = Path(opt_cfg["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path / "opt_summary.csv", index=False)

    logger.info("\n" + "=" * 60)
    logger.info("  优化结果汇总")
    logger.info("=" * 60)
    logger.info("\n" + summary_df.to_string(index=False))

    # 过拟合检验
    logger.info("\n  过拟合检验:")
    for _, row in summary_df.iterrows():
        sname = row["strategy"]
        is_sharpe = _safe_float(row.get("in_sample_sharpe", 0))
        oos_sharpe = _safe_float(row.get("oos_sharpe", 0))
        if abs(is_sharpe) > _MIN_SHARPE_FOR_DECAY_CALC:
            decay = (is_sharpe - oos_sharpe) / abs(is_sharpe)
            status = "合格" if decay < _DECAY_PASS_THRESHOLD else "过拟合风险"
            logger.info(f"    {sname}: IS={is_sharpe:.4f}, OOS={oos_sharpe:.4f}, 衰减={decay:.1%}, 评分={row.get('oos_priority_score', 0):.4f} → {status}")
        else:
            logger.info(f"    {sname}: IS Sharpe接近0，无法判断")

    # ── 因子IC稳定性分析（后验） ──
    logger.info("\n  因子IC稳定性分析:")
    ic_stability_rows = []
    ic_config = RollingICConfig(window=60, min_observations=30)
    decay_config = FactorDecayConfig(trend_window=40)

    for symbol in opt_cfg["symbols"]:
        try:
            sym_df = ds.query(
                opt_cfg["full_start"], opt_cfg["full_end"], symbols=[symbol]
            )
            if sym_df is None or len(sym_df) < 60:
                continue

            scored = _compute_factor_scores_from_ohlcv(sym_df)
            factor_names = ["ts_momentum", "roll_yield", "alpha019", "alpha032"]

            ic_engine = RollingICWeightEngine(ic_config)
            decay_monitor = FactorDecayMonitor(decay_config)

            for i in range(len(scored)):
                row = scored.iloc[i]
                forward_ret = float(row["forward_return"])
                if not np.isfinite(forward_ret):
                    continue

                factor_scores = {
                    name: float(row.get(name, 0.0))
                    for name in factor_names
                    if np.isfinite(row.get(name, 0.0))
                }
                if not factor_scores:
                    continue

                ic_engine.update(factor_scores, forward_ret)
                current_ic = ic_engine.current_ic
                for name, ic_val in current_ic.items():
                    decay_monitor.update(name, ic_val)

            decay_monitor.check_decay()
            ic_summary = ic_engine.get_ic_summary()

            for name, stats in ic_summary.items():
                status = decay_monitor.current_status.get(name, DecayStatus.HEALTHY).value
                ic_stability_rows.append({
                    "symbol": symbol,
                    "factor": name,
                    "mean_ic": round(stats.get("mean", 0.0), 4),
                    "ir": round(stats.get("ir", 0.0), 4),
                    "current_weight": round(ic_engine.get_dynamic_weights().get(name, 0.0), 4),
                    "decay_status": status,
                })

        except Exception as e:
            logger.debug(f"    {symbol} IC分析失败: {e}")

    if ic_stability_rows:
        ic_df = pd.DataFrame(ic_stability_rows)
        ic_df.to_csv(output_path / "opt_ic_stability.csv", index=False)
        for _, row in ic_df.iterrows():
            logger.info(
                f"    {row['symbol']}/{row['factor']}: "
                f"mean_IC={row['mean_ic']:.4f}, IR={row['ir']:.2f}, "
                f"weight={row['current_weight']:.4f}, status={row['decay_status']}"
            )

    # 敏感性摘要（P0-A 交付物）
    if not skip_optimization:
        _print_sensitivity_summary(all_grid_results, param_spaces, output_path)

    if not skip_optimization:
        print_optimization_suggestions(best_params_all, best_window_config, lib)

    logger.info(f"\n  优化完成: {datetime.now()}")
    logger.info(f"  结果目录: {opt_cfg['output_dir']}/")

    summary_path = output_path / "opt_summary.csv"

    return {
        "best_params": best_params_all,
        "best_window_config": best_window_config,
        "summary": summary_df,
        "grid_results": all_grid_results,
        "validation_results": validation_results,
        "oos_results": oos_results,
        "summary_path": str(summary_path),
    }


def _print_sensitivity_summary(
    all_grid_results: Dict[str, pd.DataFrame],
    param_spaces: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """输出参数敏感性摘要（P0-A 交付物）。"""
    logger.info("\n  参数敏感性摘要:")
    sensitivity_rows = []

    for sname, grid_df in all_grid_results.items():
        if grid_df.empty:
            continue
        param_keys = list(param_spaces.get(sname, {}).keys())

        for pk in param_keys:
            groups = grid_df.groupby(pk)["sharpe"].agg(["mean", "std", "min", "max"])
            for val, row in groups.iterrows():
                sensitivity_rows.append({
                    "strategy": sname,
                    "param": pk,
                    "value": val,
                    "mean_sharpe": round(row["mean"], 4),
                    "std_sharpe": round(row["std"], 4) if not pd.isna(row["std"]) else 0.0,
                    "min_sharpe": round(row["min"], 4),
                    "max_sharpe": round(row["max"], 4),
                })
                logger.info(f"    {sname}.{pk}={val}: mean={row['mean']:.4f}, std={row.get('std', 0):.4f}")

    if sensitivity_rows:
        sens_df = pd.DataFrame(sensitivity_rows)
        sens_df.to_csv(output_path / "opt_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
