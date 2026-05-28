#!/usr/bin/env python3
"""
参数优化脚本：网格搜索 + 滚动窗口验证

使用系统内置的 PyBrokerBacktestRunner 对每个策略的参数空间进行：
  1. 网格搜索：在全样本内数据上搜索最优参数组合
  2. 滚动窗口验证：用 Walkforward 验证最优参数是否过拟合
  3. 输出最优参数建议

严格使用系统现有模块，不引入外部工具。
所有参数空间来自 StrategyLibrary.param_ranges，
参数覆盖通过 PyBrokerBacktestRunner.run(custom_params=...) 实现。
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from itertools import product
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from core.engine.broker_adapter import PyBrokerBacktestRunner, create_hybrid_data_source
from core.config import BacktestConfig
from core.strategy_library import StrategyLibrary


def _load_opt_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bt = cfg.get("backtest", {})
    return {
        "initial_cash": bt.get("initial_cash", 1_000_000),
        "commission_rate": bt.get("commission_rate", 0.0003),
        "slippage_rate": bt.get("slippage_rate", 0.0002),
        "full_start": bt.get("full_start_date", "2016-01-01"),
        "in_sample_end": bt.get("in_sample_end_date", "2023-12-31"),
        "full_end": bt.get("full_end_date", "2026-05-01"),
        "out_sample_start": bt.get("out_sample_start_date", "2024-01-01"),
        "symbols": cfg.get("symbols", ["SHFE.RB", "DCE.M", "CZCE.TA", "SHFE.CU", "CFFEX.IF"]),
        "output_dir": cfg.get("output", {}).get("output_dir", "output_backtest_pybroker"),
        "strategy_names": [s["name"] for s in cfg.get("strategies", []) if s.get("name")],
    }


def _get_param_spaces(lib: StrategyLibrary, strategy_names: list) -> dict:
    param_spaces = {}
    for sname in strategy_names:
        profile = lib.get_profile(sname)
        if profile is not None and profile.param_ranges:
            param_spaces[sname] = dict(profile.param_ranges)
    return param_spaces


def grid_search_single_strategy(
    strategy_name: str, param_space: dict, ds, lib: StrategyLibrary, opt_cfg: dict
) -> pd.DataFrame:
    keys = list(param_space.keys())
    values = list(param_space.values())
    combos = list(product(*values))
    total = len(combos)

    logger.info(f"\n  策略: {strategy_name} | 参数组合数: {total}")

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        if strategy_name == "dual_ma" and params.get("short_ma", 0) >= params.get("long_ma", 999):
            continue
        if strategy_name == "rsi" and params.get("oversold", 0) >= params.get("overbought", 100):
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
            logger.info(f"    组合 {params} 失败: {e}")
            continue

        if (i + 1) % 5 == 0 or i + 1 == total:
            logger.info(f"    进度: {i+1}/{total}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    return df


def rolling_validate(
    strategy_name: str, best_params: dict, ds, lib: StrategyLibrary, opt_cfg: dict
) -> dict:
    logger.info(f"\n  滚动窗口验证: {strategy_name} | 参数: {best_params}")

    try:
        config = BacktestConfig(
            initial_cash=opt_cfg["initial_cash"],
            commission_rate=opt_cfg["commission_rate"],
            slippage_rate=opt_cfg["slippage_rate"],
            wf_train_ratio=0.6,
            wf_step_ratio=0.15,
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

        logger.info(f"    窗口数: {result['n_windows']}")
        logger.info(f"    平均Sharpe: {result['avg_sharpe']:.4f}")
        logger.info(f"    正Sharpe比例: {result['positive_sharpe_ratio']:.1%}")

        return result

    except Exception as e:
        logger.info(f"    滚动验证失败: {e}")
        return {}


def out_of_sample_test(
    strategy_name: str, best_params: dict, ds, lib: StrategyLibrary, opt_cfg: dict
) -> dict:
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

        logger.info(f"    样本外收益: {kpi.get('total_return_pct', 0):.2f}%")
        logger.info(f"    样本外Sharpe: {kpi.get('sharpe', 0):.4f}")
        logger.info(f"    样本外回撤: {kpi.get('max_drawdown_pct', 0):.2f}%")

        return kpi

    except Exception as e:
        logger.info(f"    样本外测试失败: {e}")
        return {}


def print_optimization_suggestions(best_params_all: dict, lib: StrategyLibrary):
    logger.info("\n" + "=" * 60)
    logger.info("最优参数建议")
    logger.info("=" * 60)

    for sname, params in best_params_all.items():
        profile = lib.get_profile(sname)
        if profile is None:
            continue
        old_params = dict(profile.default_params)
        logger.info(f"\n  {sname}:")
        logger.info(f"    当前默认: {old_params}")
        logger.info(f"    建议更新: {params}")
        changed = {k: (old_params.get(k), v) for k, v in params.items() if old_params.get(k) != v}
        if changed:
            logger.info(f"    变更项: {changed}")

    logger.info("\n" + "-" * 60)
    logger.info("参数应用方式（3种，按推荐程度排序）：")
    logger.info("-" * 60)

    logger.info("\n  方式1（推荐）: 通过 StrategyLibrary.update_default_params() 运行时更新")
    logger.info("    优点: 即时生效，无需修改源代码，可回滚")
    logger.info("    示例代码:")
    logger.info("      from core.strategy_library import StrategyLibrary")
    logger.info("      lib = StrategyLibrary()")
    for sname, params in best_params_all.items():
        logger.info(f"      lib.update_default_params('{sname}', {params})")
    logger.info("      # 更新后所有使用 StrategyLibrary 的回测将自动采用新参数")

    logger.info("\n  方式2: 更新 config.yaml 策略参数段")
    logger.info("    优点: 参数与代码分离，版本可控")
    yaml_params = lib.export_params_to_yaml(list(best_params_all.keys()))
    for sname in best_params_all:
        if sname in yaml_params:
            yaml_params[sname].update(best_params_all[sname])
    logger.info("    建议将以下内容更新到 config.yaml 的 strategies 段：")
    for sname, params in best_params_all.items():
        logger.info(f"\n    - name: \"{sname}\"")
        logger.info(f"      params:")
        for k, v in params.items():
            if isinstance(v, str):
                logger.info(f"        {k}: \"{v}\"")
            else:
                logger.info(f"        {k}: {v}")

    logger.info("\n  方式3: 通过 custom_params 参数覆盖（单次回测）")
    logger.info("    优点: 不修改任何持久化配置，仅影响当前回测")
    logger.info("    示例代码:")
    logger.info("      runner = PyBrokerBacktestRunner(ds, config)")
    for sname, params in best_params_all.items():
        logger.info(f"      runner.register_strategies(['{sname}'])")
        logger.info(f"      result = runner.run(start, end, custom_params={{'{sname}': {params}}})")

    logger.info("\n" + "=" * 60)
    logger.info("  注意: 方式1和方式2的参数变更在系统重启后不会持久化")
    logger.info("  如需永久生效，请将方式2的参数更新到 config.yaml")
    logger.info("=" * 60)


def main():
    logger.info("=" * 60)
    logger.info("  参数优化：网格搜索 + 滚动窗口验证")
    logger.info(f"  开始: {datetime.now()}")
    logger.info("=" * 60)

    opt_cfg = _load_opt_config()
    lib = StrategyLibrary()

    logger.info("\n[1/4] 加载数据...")
    phone = os.getenv("TQSDK_PHONE")
    password = os.getenv("TQSDK_PASSWORD")
    ds = create_hybrid_data_source(
        phone=phone,
        password=password,
        symbols=opt_cfg["symbols"],
        data_length=3000,
    )

    strategy_names = opt_cfg["strategy_names"]
    param_spaces = _get_param_spaces(lib, strategy_names)
    logger.info(f"\n  参数空间来源: StrategyLibrary.param_ranges")
    for sname, ps in param_spaces.items():
        total = 1
        for v in ps.values():
            total *= len(v)
        logger.info(f"    {sname}: {ps} ({total} 组合)")

    logger.info(f"\n[2/4] 网格搜索（样本内: {opt_cfg['full_start']} ~ {opt_cfg['in_sample_end']}）")
    all_grid_results = {}
    best_params_all = {}

    for sname, param_space in param_spaces.items():
        grid_df = grid_search_single_strategy(sname, param_space, ds, lib, opt_cfg)
        all_grid_results[sname] = grid_df

        if grid_df.empty:
            logger.info(f"  {sname}: 无有效结果")
            continue

        os.makedirs(opt_cfg["output_dir"], exist_ok=True)
        grid_df.to_csv(os.path.join(opt_cfg["output_dir"], f"opt_grid_{sname}.csv"), index=False)

        logger.info(f"\n  {sname} Top 5 (按Sharpe):")
        top5 = grid_df.head(5)
        for idx, row in top5.iterrows():
            param_str = ", ".join(f"{k}={row[k]}" for k in param_space.keys())
            logger.info(f"    #{idx+1} {param_str} => Sharpe={row['sharpe']:.4f}, Return={row['total_return_pct']:.2f}%, DD={row['max_drawdown_pct']:.2f}%")

        best = grid_df.iloc[0]
        best_params = {k: best[k] for k in param_space.keys()}
        best_params_all[sname] = best_params

    logger.info("\n[3/4] 滚动窗口验证（Walkforward）")
    validation_results = {}

    for sname, params in best_params_all.items():
        val = rolling_validate(sname, params, ds, lib, opt_cfg)
        validation_results[sname] = val

    logger.info(f"\n[4/4] 样本外测试（{opt_cfg.get('out_sample_start', '2024-01-01')} ~ {opt_cfg['full_end']}）")
    oos_results = {}

    for sname, params in best_params_all.items():
        oos = out_of_sample_test(sname, params, ds, lib, opt_cfg)
        oos_results[sname] = oos

    logger.info("\n" + "=" * 60)
    logger.info("  优化结果汇总")
    logger.info("=" * 60)

    summary = []
    for sname in param_spaces:
        if sname not in best_params_all:
            continue
        row = {
            "strategy": sname,
            "best_params": str(best_params_all[sname]),
        }
        grid_df = all_grid_results.get(sname)
        if grid_df is not None and not grid_df.empty:
            top = grid_df.iloc[0]
            row["in_sample_sharpe"] = round(top["sharpe"], 4)
            row["in_sample_return"] = round(top["total_return_pct"], 2)
            row["in_sample_drawdown"] = round(top["max_drawdown_pct"], 2)
        val = validation_results.get(sname, {})
        row["wf_avg_sharpe"] = round(val.get("avg_sharpe", 0), 4)
        row["wf_positive_ratio"] = round(val.get("positive_sharpe_ratio", 0), 2)
        oos = oos_results.get(sname, {})
        row["oos_sharpe"] = round(oos.get("sharpe", 0), 4)
        row["oos_return"] = round(oos.get("total_return_pct", 0), 2)
        row["oos_drawdown"] = round(oos.get("max_drawdown_pct", 0), 2)

        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "opt_summary.csv"), index=False)

    logger.info("\n" + summary_df.to_string(index=False))

    logger.info("\n  过拟合检验:")
    for _, row in summary_df.iterrows():
        sname = row["strategy"]
        is_sharpe = row.get("in_sample_sharpe", 0)
        oos_sharpe = row.get("oos_sharpe", 0)
        if abs(is_sharpe) > 0.01:
            decay = (is_sharpe - oos_sharpe) / abs(is_sharpe)
            status = "合格" if decay < 0.5 else "过拟合风险"
            logger.info(f"    {sname}: IS={is_sharpe:.4f}, OOS={oos_sharpe:.4f}, 衰减={decay:.1%} → {status}")
        else:
            logger.info(f"    {sname}: IS Sharpe接近0，无法判断")

    print_optimization_suggestions(best_params_all, lib)

    logger.info(f"\n  优化完成: {datetime.now()}")
    logger.info(f"  结果目录: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
