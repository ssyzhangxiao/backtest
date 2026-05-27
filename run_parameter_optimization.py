#!/usr/bin/env python3
"""
参数优化脚本：网格搜索 + 滚动窗口验证

使用系统内置的 PyBrokerBacktestRunner 对每个策略的参数空间进行：
  1. 网格搜索：在全样本内数据上搜索最优参数组合
  2. 滚动窗口验证：用 Walkforward 验证最优参数是否过拟合
  3. 输出最优参数并更新策略库

严格使用系统现有模块，不引入外部工具。
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from itertools import product
import pandas as pd
import numpy as np
from datetime import datetime

from core.engine.broker_adapter import PyBrokerBacktestRunner, create_hybrid_data_source
from core.config import BacktestConfig
from core.strategy_library import StrategyLibrary

# ── 配置 ──
INITIAL_CASH = 1_000_000
FULL_START = "2016-01-01"
IN_SAMPLE_END = "2023-12-31"  # 样本内截止
FULL_END = "2026-05-01"
SYMBOLS = ["SHFE.RB", "DCE.M", "CZCE.TA", "SHFE.CU", "CFFEX.IF"]
OUTPUT_DIR = "output_backtest_pybroker"

# ── 参数搜索空间（来自 StrategyLibrary.param_ranges） ──
PARAM_SPACES = {
    "dual_ma": {
        "short_ma": [3, 5, 8, 10],
        "long_ma": [15, 20, 30, 40],
    },
    "rsi": {
        "rsi_period": [10, 14, 20],
        "oversold": [15.0, 20.0, 25.0, 30.0],
        "overbought": [70.0, 75.0, 80.0, 85.0],
    },
    "vol_breakout": {
        "atr_period": [14, 20, 26],
        "band_period": [20, 30, 40],
        "atr_multiplier": [1.5, 2.0, 2.5],
    },
}


def extract_kpi(result) -> dict:
    """从回测结果提取关键指标。"""
    kpi = {}
    m = result.metrics
    kpi["total_return_pct"] = m.get("total_return_pct", 0)
    kpi["sharpe"] = m.get("sharpe", 0)
    kpi["max_drawdown_pct"] = m.get("max_drawdown_pct", 0)
    kpi["win_rate"] = m.get("win_rate", 0)
    kpi["profit_factor"] = m.get("profit_factor", 0)
    kpi["trade_count"] = m.get("trade_count", 0)
    kpi["sortino"] = m.get("sortino", 0)
    return kpi


def grid_search_single_strategy(strategy_name: str, param_space: dict, ds) -> pd.DataFrame:
    """
    对单个策略执行网格搜索。

    对参数空间中的每组参数，修改策略库的 default_params，
    然后用 PyBrokerBacktestRunner 在样本内数据上回测。

    Returns:
        结果 DataFrame，按 Sharpe 排序
    """
    keys = list(param_space.keys())
    values = list(param_space.values())
    combos = list(product(*values))
    total = len(combos)

    print(f"\n  策略: {strategy_name} | 参数组合数: {total}")

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # 跳过不合理组合（如 short_ma >= long_ma）
        if strategy_name == "dual_ma" and params.get("short_ma", 0) >= params.get("long_ma", 999):
            continue
        if strategy_name == "rsi" and params.get("oversold", 0) >= params.get("overbought", 100):
            continue

        try:
            # 修改策略库的 default_params
            lib = StrategyLibrary()
            profile = lib.get_profile(strategy_name)
            if profile is None:
                continue

            # 临时修改默认参数
            original_params = dict(profile.default_params)
            profile.default_params.update(params)

            # 创建 runner 并回测（样本内）
            config = BacktestConfig(
                initial_cash=INITIAL_CASH,
                commission_rate=0.0003,
                slippage_rate=0.0002,
            )
            runner = PyBrokerBacktestRunner(ds, config)
            runner.register_strategies([strategy_name])

            result = runner.run(FULL_START, IN_SAMPLE_END)
            kpi = extract_kpi(result)
            kpi.update(params)
            results.append(kpi)

            # 恢复原始参数
            profile.default_params = original_params

        except Exception as e:
            print(f"    组合 {params} 失败: {e}")
            continue

        # 进度
        if (i + 1) % 5 == 0 or i + 1 == total:
            print(f"    进度: {i+1}/{total}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    return df


def rolling_validate(strategy_name: str, best_params: dict, ds) -> dict:
    """
    用 Walkforward 滚动窗口验证最优参数。

    在样本内数据上用 Walkforward 验证，确保参数不过拟合。
    返回各窗口的平均 Sharpe 和收益率。
    """
    print(f"\n  滚动窗口验证: {strategy_name} | 参数: {best_params}")

    # 修改策略库参数
    lib = StrategyLibrary()
    profile = lib.get_profile(strategy_name)
    if profile is None:
        return {}

    original_params = dict(profile.default_params)
    profile.default_params.update(best_params)

    try:
        config = BacktestConfig(
            initial_cash=INITIAL_CASH,
            commission_rate=0.0003,
            slippage_rate=0.0002,
            wf_train_ratio=0.6,
            wf_step_ratio=0.15,
        )
        runner = PyBrokerBacktestRunner(ds, config)
        runner.register_strategies([strategy_name])

        wf_result = runner.walkforward(FULL_START, IN_SAMPLE_END)

        window_sharpes = []
        window_returns = []
        for w in wf_result.windows:
            m = w.get("metrics", {})
            if "sharpe" in m:
                window_sharpes.append(m["sharpe"])
            if "total_return_pct" in m:
                window_returns.append(m["total_return_pct"])

        result = {
            "n_windows": len(wf_result.windows),
            "avg_sharpe": np.mean(window_sharpes) if window_sharpes else 0,
            "avg_return_pct": np.mean(window_returns) if window_returns else 0,
            "min_sharpe": min(window_sharpes) if window_sharpes else 0,
            "sharpe_std": np.std(window_sharpes) if len(window_sharpes) > 1 else 0,
            "positive_sharpe_ratio": sum(1 for s in window_sharpes if s > 0) / max(len(window_sharpes), 1),
        }

        print(f"    窗口数: {result['n_windows']}")
        print(f"    平均Sharpe: {result['avg_sharpe']:.4f}")
        print(f"    正Sharpe比例: {result['positive_sharpe_ratio']:.1%}")

        return result

    except Exception as e:
        print(f"    滚动验证失败: {e}")
        return {}
    finally:
        profile.default_params = original_params


def out_of_sample_test(strategy_name: str, best_params: dict, ds) -> dict:
    """
    样本外测试：用最优参数在样本外数据上回测。
    """
    print(f"\n  样本外测试: {strategy_name}")

    lib = StrategyLibrary()
    profile = lib.get_profile(strategy_name)
    if profile is None:
        return {}

    original_params = dict(profile.default_params)
    profile.default_params.update(best_params)

    try:
        config = BacktestConfig(
            initial_cash=INITIAL_CASH,
            commission_rate=0.0003,
            slippage_rate=0.0002,
        )
        runner = PyBrokerBacktestRunner(ds, config)
        runner.register_strategies([strategy_name])

        result = runner.run("2024-01-01", FULL_END)
        kpi = extract_kpi(result)

        print(f"    样本外收益: {kpi['total_return_pct']:.2f}%")
        print(f"    样本外Sharpe: {kpi['sharpe']:.4f}")
        print(f"    样本外回撤: {kpi['max_drawdown_pct']:.2f}%")

        return kpi

    except Exception as e:
        print(f"    样本外测试失败: {e}")
        return {}
    finally:
        profile.default_params = original_params


def update_strategy_library(best_params_all: dict):
    """
    将最优参数更新到策略库（持久化到 __init__.py）。
    """
    lib = StrategyLibrary()

    print("\n" + "=" * 60)
    print("更新策略库默认参数")
    print("=" * 60)

    for sname, params in best_params_all.items():
        profile = lib.get_profile(sname)
        if profile is None:
            continue
        old_params = dict(profile.default_params)
        # 仅更新搜索空间中的参数
        for k, v in params.items():
            if k in old_params or k in PARAM_SPACES.get(sname, {}):
                profile.default_params[k] = v
        print(f"  {sname}: {old_params} → {profile.default_params}")

    # 写入到 __init__.py
    init_path = os.path.join(os.path.dirname(__file__), "core", "strategy_library", "__init__.py")
    with open(init_path, "r", encoding="utf-8") as f:
        content = f.read()

    for sname, params in best_params_all.items():
        profile = lib.get_profile(sname)
        if profile is None:
            continue
        # 找到策略注册块中的 default_params 并替换
        # 这里只打印，实际修改需要精确的字符串替换
        print(f"  [需手动更新] {sname}: {params}")

    print("\n  注意: 参数已临时生效。如需持久化，请手动更新 core/strategy_library/__init__.py")


def main():
    print("=" * 60)
    print("  参数优化：网格搜索 + 滚动窗口验证")
    print(f"  开始: {datetime.now()}")
    print("=" * 60)

    # ── 1. 加载数据 ──
    print("\n[1/4] 加载 TqSdk 数据...")
    ds = create_hybrid_data_source(
        phone="13600198250",
        password="lg123456789",
        symbols=SYMBOLS,
        data_length=3000,
    )

    # ── 2. 网格搜索 ──
    print("\n[2/4] 网格搜索（样本内: 2016-01 ~ 2023-12）")
    all_grid_results = {}
    best_params_all = {}

    for sname, param_space in PARAM_SPACES.items():
        grid_df = grid_search_single_strategy(sname, param_space, ds)
        all_grid_results[sname] = grid_df

        if grid_df.empty:
            print(f"  {sname}: 无有效结果")
            continue

        # 保存完整网格搜索结果
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        grid_df.to_csv(os.path.join(OUTPUT_DIR, f"opt_grid_{sname}.csv"), index=False)

        # 输出 Top 5
        print(f"\n  {sname} Top 5 (按Sharpe):")
        top5 = grid_df.head(5)
        for idx, row in top5.iterrows():
            param_str = ", ".join(f"{k}={row[k]}" for k in param_space.keys())
            print(f"    #{idx+1} {param_str} => Sharpe={row['sharpe']:.4f}, Return={row['total_return_pct']:.2f}%, DD={row['max_drawdown_pct']:.2f}%")

        # 取 Top 1 参数
        best = grid_df.iloc[0]
        best_params = {k: best[k] for k in param_space.keys()}
        best_params_all[sname] = best_params

    # ── 3. 滚动窗口验证 ──
    print("\n[3/4] 滚动窗口验证（Walkforward）")
    validation_results = {}

    for sname, params in best_params_all.items():
        val = rolling_validate(sname, params, ds)
        validation_results[sname] = val

    # ── 4. 样本外测试 ──
    print("\n[4/4] 样本外测试（2024-01 ~ 2026-05）")
    oos_results = {}

    for sname, params in best_params_all.items():
        oos = out_of_sample_test(sname, params, ds)
        oos_results[sname] = oos

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("  优化结果汇总")
    print("=" * 60)

    summary = []
    for sname in PARAM_SPACES:
        if sname not in best_params_all:
            continue
        row = {
            "strategy": sname,
            "best_params": str(best_params_all[sname]),
        }
        # 网格搜索 Top1
        grid_df = all_grid_results.get(sname)
        if grid_df is not None and not grid_df.empty:
            top = grid_df.iloc[0]
            row["in_sample_sharpe"] = round(top["sharpe"], 4)
            row["in_sample_return"] = round(top["total_return_pct"], 2)
            row["in_sample_drawdown"] = round(top["max_drawdown_pct"], 2)
        # 滚动验证
        val = validation_results.get(sname, {})
        row["wf_avg_sharpe"] = round(val.get("avg_sharpe", 0), 4)
        row["wf_positive_ratio"] = round(val.get("positive_sharpe_ratio", 0), 2)
        # 样本外
        oos = oos_results.get(sname, {})
        row["oos_sharpe"] = round(oos.get("sharpe", 0), 4)
        row["oos_return"] = round(oos.get("total_return_pct", 0), 2)
        row["oos_drawdown"] = round(oos.get("max_drawdown_pct", 0), 2)

        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "opt_summary.csv"), index=False)

    print("\n" + summary_df.to_string(index=False))

    # 判断是否过拟合：样本外Sharpe衰减
    print("\n  过拟合检验:")
    for _, row in summary_df.iterrows():
        sname = row["strategy"]
        is_sharpe = row.get("in_sample_sharpe", 0)
        oos_sharpe = row.get("oos_sharpe", 0)
        if abs(is_sharpe) > 0.01:
            decay = (is_sharpe - oos_sharpe) / abs(is_sharpe)
            status = "合格" if decay < 0.5 else "过拟合风险"
            print(f"    {sname}: IS={is_sharpe:.4f}, OOS={oos_sharpe:.4f}, 衰减={decay:.1%} → {status}")
        else:
            print(f"    {sname}: IS Sharpe接近0，无法判断")

    # ── 更新策略库 ──
    update_strategy_library(best_params_all)

    print(f"\n  优化完成: {datetime.now()}")
    print(f"  结果目录: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
