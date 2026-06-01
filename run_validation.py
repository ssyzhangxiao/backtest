#!/usr/bin/env python3
"""
验证脚本 v2：参数优化 + WalkForward + 样本外 + 蒙特卡洛

严格调用系统标准模块，与 run_full_backtest.py / run_parameter_optimization.py
保持一致的接口规范和执行逻辑。

任务1: 参数优化 + WalkForward 搜索网格验证（P0-A + P0-C）
任务2: 2016-2020 训练 / 2021-2025 样本外验证（P1-B）
    2a: WalkForward 滚动验证（单策略，run_e6_walkforward）
    2b: 样本内外对比（单策略汇总，run_e7_out_of_sample）
    2c: 环境分布统计
    2d: 训练期回测（固定参数 vs 环境感知参数）
    2e: 验证期按年切片
任务3: 蒙特卡洛 1000 次鲁棒性测试（P2-D）
"""

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import warnings
warnings.filterwarnings("ignore")

from loguru import logger

from core.engine.backtest_runner import (
    PyBrokerBacktestRunner,
)
from core.engine.pybroker_data_source import (
    PyBrokerDataSource,
    create_hybrid_data_source,
)
from core.engine.rolling_ic import RollingICWeightEngine, RollingICConfig
from core.engine.factor_decay import (
    FactorDecayMonitor, FactorDecayConfig, DecayStatus,
)
from core.config import BacktestConfig
from core.strategy_registry import StrategyLibrary
from run_full_backtest import (
    load_config,
    get_tqsdk_credentials,
    _safe_float,
    _is_valid_number,
    save_csv,
    _compute_factor_scores_from_ohlcv,
    run_e6_walkforward,
    run_e7_out_of_sample,
    run_e9_monte_carlo,
)
from run_parameter_optimization import (
    grid_search_single_strategy,
    window_search_single_strategy,
    rolling_validate,
    out_of_sample_test,
    compute_oos_priority_score,
    _get_param_spaces,
    _INSAMPLE_BAR_RANGE as OPT_INSAMPLE_BAR_RANGE,
    _OOF_PENALTY_WEIGHT,
    _TOP_N_FOR_WINDOW_SEARCH,
)
from core.market_regime import MarketRegime

# ══════════════════════════════════════════════════════════════════════════════
# 全局常量（可被 config.yaml 覆盖）
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_TRAIN_START = "2016-01-01"
_DEFAULT_TRAIN_END = "2020-12-31"
_DEFAULT_TEST_START = "2021-01-01"
_DEFAULT_TEST_END = "2025-12-31"
_N_MONTE_CARLO = 1000
_RANDOM_SEED = 42
_DEFAULT_BANKRUPTCY_THRESHOLD = 0.8


def _build_opt_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """构建优化配置，从 config.yaml 读取日期范围（有则用，无则用默认值）。"""
    bt = cfg.get("backtest", {})
    return {
        "initial_cash": bt.get("initial_cash", 1_000_000),
        "commission_rate": bt.get("commission_rate", 0.0005),
        "slippage_rate": bt.get("slippage_rate", 0.0005),
        "train_start": bt.get("in_sample_start_date", _DEFAULT_TRAIN_START),
        "train_end": bt.get("in_sample_end_date", _DEFAULT_TRAIN_END),
        "test_start": bt.get("out_sample_start_date", _DEFAULT_TEST_START),
        "test_end": bt.get("out_sample_end_date", _DEFAULT_TEST_END),
        "full_start": bt.get("full_start_date", _DEFAULT_TRAIN_START),
        "full_end": bt.get("full_end_date", _DEFAULT_TEST_END),
        "in_sample_end": bt.get("in_sample_end_date", _DEFAULT_TRAIN_END),
        "out_sample_start": bt.get("out_sample_start_date", _DEFAULT_TEST_START),
        "symbols": cfg.get("symbols", ["SHFE.RB", "DCE.M", "CZCE.TA", "SHFE.CU", "CFFEX.IF"]),
        "output_dir": cfg.get("output", {}).get("output_dir", "output_validation"),
        "strategy_names": [s["name"] for s in cfg.get("strategies", []) if s.get("name")],
        "bankruptcy_threshold": cfg.get("risk_management", {}).get("bankruptcy_threshold", _DEFAULT_BANKRUPTCY_THRESHOLD),
    }


def _load_data(cfg: Dict[str, Any], opt_cfg: Dict[str, Any]) -> PyBrokerDataSource:
    """
    加载数据：tqsdk 优先，与 run_full_backtest 保持一致。

    create_hybrid_data_source 内部已实现 tqsdk 优先 + CSV fallback，
    无需手动过滤日期（tqsdk 返回的数据即覆盖最近市场）。
    """
    logger.info("加载数据（TqSdk 优先）...")
    phone, password = get_tqsdk_credentials()
    data_length = cfg.get("data", {}).get("tqsdk_data_length", 4000)
    ds = create_hybrid_data_source(
        phone=phone,
        password=password,
        symbols=opt_cfg["symbols"],
        data_dir=cfg.get("data", {}).get("csv_data_dir", "data"),
        data_length=data_length,
    )
    pybroker_df = ds.to_pybroker_df()
    if pybroker_df is not None and not pybroker_df.empty:
        # 日期范围校验
        if "date" in pybroker_df.columns:
            data_min = pybroker_df["date"].min()
            data_max = pybroker_df["date"].max()
            logger.info(f"  数据日期范围: {data_min} ~ {data_max}")
            required_start = opt_cfg.get("train_start", _DEFAULT_TRAIN_START)
            required_end = opt_cfg.get("test_end", _DEFAULT_TEST_END)
            if hasattr(data_min, "strftime"):
                if str(data_min)[:10] > required_start:
                    logger.warning(f"  数据起始日期({str(data_min)[:10]})晚于回测起始日期({required_start})，回测范围将被截断")
                if str(data_max)[:10] < required_end:
                    logger.warning(f"  数据结束日期({str(data_max)[:10]})早于回测结束日期({required_end})，回测范围将被截断")
            logger.info(f"  数据加载完成: {len(pybroker_df)} 行, {pybroker_df['symbol'].nunique()} 品种")
    return ds


# ══════════════════════════════════════════════════════════════════════════════
# 任务1: 参数优化 + WalkForward 搜索网格验证
# ══════════════════════════════════════════════════════════════════════════════


def task1_walkforward_grid(
    ds: PyBrokerDataSource,
    opt_cfg: Dict[str, Any],
    lib: StrategyLibrary,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    参数优化 + WalkForward 搜索网格验证（P0-A + P0-C）。

    调用 run_parameter_optimization 的标准接口：
      1. grid_search_single_strategy: 网格搜索
      2. window_search_single_strategy: 窗口搜索
      3. rolling_validate: 滚动窗口验证
      4. compute_oos_priority_score: 样本外优先评分
    """
    logger.info("=" * 60)
    logger.info("任务1: 参数优化 + WalkForward 搜索网格验证")
    logger.info(f"  搜索网格: {OPT_INSAMPLE_BAR_RANGE}")
    logger.info(f"  样本外优先惩罚权重: {_OOF_PENALTY_WEIGHT}")
    logger.info("=" * 60)

    strategy_names = opt_cfg["strategy_names"]
    param_spaces = _get_param_spaces(lib, strategy_names)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1a: 网格搜索 ──
    all_grid_results: Dict[str, pd.DataFrame] = {}
    best_params_all: Dict[str, Dict[str, Any]] = {}
    best_window_config: Dict[str, Dict[str, int]] = {}

    for sname, param_space in param_spaces.items():
        logger.info(f"\n  网格搜索: {sname}")
        grid_df = grid_search_single_strategy(sname, param_space, ds, lib, opt_cfg)
        all_grid_results[sname] = grid_df

        if not grid_df.empty:
            grid_df.to_csv(output_dir / f"task1_grid_{sname}.csv", index=False)
            top1 = grid_df.iloc[0]
            param_keys = list(param_space.keys())
            best_params_all[sname] = {k: top1[k] for k in param_keys}
            logger.info(f"    Top1: Sharpe={_safe_float(top1.get('sharpe', 0)):.4f}")
        else:
            profile = lib.get_profile(sname)
            best_params_all[sname] = dict(profile.default_params) if profile else {}

    # ── 1b: 窗口搜索（P0-A） ──
    logger.info("\n  窗口搜索（P0-A）:")
    for sname, param_space in param_spaces.items():
        grid_df = all_grid_results.get(sname, pd.DataFrame())
        if grid_df.empty:
            best_window_config[sname] = {"train_bars": 252, "test_bars": 63, "step_bars": 21}
            continue

        param_keys = list(param_space.keys())
        top_params_list = []
        for _, row in grid_df.head(_TOP_N_FOR_WINDOW_SEARCH).iterrows():
            top_params_list.append({k: row[k] for k in param_keys})

        if not top_params_list:
            best_window_config[sname] = {"train_bars": 252, "test_bars": 63, "step_bars": 21}
            continue

        window_df = window_search_single_strategy(
            sname, top_params_list, ds, lib, opt_cfg,
        )

        if not window_df.empty:
            window_df.to_csv(output_dir / f"task1_window_{sname}.csv", index=False)
            best_win = window_df.iloc[0]
            train_bars = int(best_win["train_bars"])
            test_bars = int(best_win["test_bars"])
            step_bars = int(best_win["step_bars"])
            best_params_all[sname] = {k: best_win[k] for k in param_keys}
            best_window_config[sname] = {
                "train_bars": train_bars,
                "test_bars": test_bars,
                "step_bars": step_bars,
            }
            logger.info(f"    {sname}: train={train_bars}, test={test_bars}, step={step_bars}")
        else:
            best_window_config[sname] = {"train_bars": 252, "test_bars": 63, "step_bars": 21}

    # ── 1c: 滚动窗口验证 + 样本外测试 ──
    logger.info("\n  滚动窗口验证 + 样本外测试:")
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

    # ── 1d: 新旧配置对比 ──
    logger.info("\n  新旧 WalkForward 配置对比:")
    compare_rows = []
    for sname in param_spaces:
        win_cfg = best_window_config.get(sname, {"train_bars": 252, "test_bars": 63, "step_bars": 21})

        new_sharpes = _run_wf_windows(
            sname, ds, opt_cfg,
            train_bars=win_cfg["train_bars"],
            test_bars=win_cfg["test_bars"],
            step_bars=win_cfg["step_bars"],
        )
        old_sharpes = _run_wf_windows(
            sname, ds, opt_cfg,
            train_bars=504, test_bars=126, step_bars=42,
        )

        new_avg = np.mean(new_sharpes) if new_sharpes else 0.0
        new_min = min(new_sharpes) if new_sharpes else 0.0
        old_avg = np.mean(old_sharpes) if old_sharpes else 0.0
        old_min = min(old_sharpes) if old_sharpes else 0.0
        avg_ratio = new_avg / old_avg if abs(old_avg) > 1e-6 else 0.0
        min_ratio = new_min / old_min if abs(old_min) > 1e-6 else 0.0

        is_sharpe = _safe_float(all_grid_results.get(sname, pd.DataFrame()).iloc[0].get("sharpe", 0)) if not all_grid_results.get(sname, pd.DataFrame()).empty else 0.0
        oos_sharpe = _safe_float(oos_results.get(sname, {}).get("sharpe", 0))
        oos_score = compute_oos_priority_score(is_sharpe, oos_sharpe)

        row = {
            "strategy": sname,
            "best_params": str(best_params_all.get(sname, {})),
            "train_bars": win_cfg["train_bars"],
            "test_bars": win_cfg["test_bars"],
            "step_bars": win_cfg["step_bars"],
            "new_avg_sharpe": round(new_avg, 4),
            "new_min_sharpe": round(new_min, 4),
            "old_avg_sharpe": round(old_avg, 4),
            "old_min_sharpe": round(old_min, 4),
            "avg_ratio": round(avg_ratio, 4),
            "min_ratio": round(min_ratio, 4),
            "avg_check": "≥95%" if avg_ratio >= 0.95 else "<95%",
            "min_check": "≥90%" if min_ratio >= 0.90 else "<90%",
            "is_sharpe": round(is_sharpe, 4),
            "oos_sharpe": round(oos_sharpe, 4),
            "oos_priority_score": round(oos_score, 4),
        }
        compare_rows.append(row)
        logger.info(f"    {sname}: 新avg={new_avg:.4f}, 旧avg={old_avg:.4f}, 比值={avg_ratio:.2%}")

    df_compare = pd.DataFrame(compare_rows)
    df_compare.to_csv(output_dir / "task1_wf_compare.csv", index=False)

    # ── 1e: 窗口敏感性网格 ──
    grid_rows = []
    for sname in param_spaces:
        for train_bars in OPT_INSAMPLE_BAR_RANGE:
            test_bars = max(5, int(train_bars * 0.25))
            step_bars = max(5, test_bars)
            sharpes = _run_wf_windows(sname, ds, opt_cfg, train_bars, test_bars, step_bars)
            avg_s = np.mean(sharpes) if sharpes else 0.0
            grid_rows.append({
                "strategy": sname,
                "train_bars": train_bars,
                "test_bars": test_bars,
                "step_bars": step_bars,
                "avg_sharpe": round(avg_s, 4),
                "n_windows": len(sharpes),
            })

    df_grid = pd.DataFrame(grid_rows)
    df_grid.to_csv(output_dir / "task1_wf_grid.csv", index=False)

    _plot_wf_comparison(df_compare, output_dir / "task1_wf_compare.png")
    _plot_wf_sensitivity(df_grid, output_dir / "task1_wf_sensitivity.png")

    return {
        "compare": df_compare,
        "grid": df_grid,
        "best_params": best_params_all,
        "best_window_config": best_window_config,
        "validation_results": validation_results,
        "oos_results": oos_results,
    }


def _run_wf_windows(
    strategy_name: str,
    ds: PyBrokerDataSource,
    opt_cfg: Dict[str, Any],
    train_bars: int = 252,
    test_bars: int = 63,
    step_bars: int = 21,
) -> List[float]:
    """执行 WalkForward 并返回各窗口 Sharpe 列表。"""
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

        sharpes = []
        for w in wf_result.windows:
            m = w.get("metrics", {})
            s = _safe_float(m.get("sharpe", 0))
            sharpes.append(s)
        return sharpes
    except Exception as e:
        logger.error(f"    WF失败 ({strategy_name}, train={train_bars}): {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 任务2: 2016-2020 训练 / 2021-2025 样本外验证
# ══════════════════════════════════════════════════════════════════════════════


def task2_train_test_split(
    ds: PyBrokerDataSource,
    opt_cfg: Dict[str, Any],
    lib: StrategyLibrary,
    output_dir: Path,
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    训练/验证期划分验证（P1-B）。

    调用 run_full_backtest 的标准接口：
      - run_e6_walkforward: 单策略 WalkForward 滚动验证（按品种分别执行）
      - run_e7_out_of_sample: 单策略汇总样本内外对比
    同时补充环境分布统计和按年切片验证。
    """
    train_start = opt_cfg.get("train_start", _DEFAULT_TRAIN_START)
    train_end = opt_cfg.get("train_end", _DEFAULT_TRAIN_END)
    test_start = opt_cfg.get("test_start", _DEFAULT_TEST_START)
    test_end = opt_cfg.get("test_end", _DEFAULT_TEST_END)

    logger.info("=" * 60)
    logger.info("任务2: 训练/验证期划分验证（P1-B）")
    logger.info(f"  训练期: {train_start} ~ {train_end}")
    logger.info(f"  验证期: {test_start} ~ {test_end}")
    logger.info("=" * 60)

    strategy_names = opt_cfg["strategy_names"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 2a: 构建回测配置（与 run_full_backtest 一致） ──
    val_config = _build_validation_config(opt_cfg)

    # ── 2b: WalkForward 滚动验证（调用 run_full_backtest.run_e6_walkforward） ──
    logger.info("\n  WalkForward 滚动验证:")
    wf_df = run_e6_walkforward(ds, val_config, output_dir)

    # ── 2c: 样本外验证（调用 run_full_backtest.run_e7_out_of_sample） ──
    logger.info("\n  样本外验证:")
    oos_df = run_e7_out_of_sample(ds, val_config, output_dir)

    # ── 2d: 环境分布统计 ──
    logger.info("\n  环境分布统计:")
    env_stats = _compute_environment_stats(ds, opt_cfg)
    if not env_stats.empty:
        env_stats.to_csv(output_dir / "task2_env_stats.csv", index=False)
        logger.info(f"\n{env_stats.to_string(index=False)}")

    # ── 2e: 训练期回测（固定参数 vs 环境感知参数） ──
    logger.info("\n  训练期回测（固定参数 vs 环境感知参数）:")
    train_fixed = _run_period_backtest(
        strategy_names, ds, opt_cfg, train_start, train_end, "fixed",
        best_params=best_params,
    )
    train_regime = _run_period_backtest(
        strategy_names, ds, opt_cfg, train_start, train_end, "regime",
        best_params=best_params,
    )

    # ── 2f: 验证期按年切片 ──
    logger.info("\n  验证期按年切片:")
    yearly_results = []
    start_year = int(train_end[:4])
    end_year = int(test_end[:4])
    for year in range(start_year + 1, end_year + 1):
        start = f"{year}-01-01"
        end = f"{year}-12-31"

        fixed_kpi = _run_period_backtest(
            strategy_names, ds, opt_cfg, start, end, "fixed",
            best_params=best_params,
        )
        regime_kpi = _run_period_backtest(
            strategy_names, ds, opt_cfg, start, end, "regime",
            best_params=best_params,
        )

        for sname in strategy_names:
            fk = fixed_kpi.get(sname, {})
            rk = regime_kpi.get(sname, {})
            yearly_results.append({
                "year": year,
                "strategy": sname,
                "fixed_sharpe": round(_safe_float(fk.get("sharpe", 0)), 4),
                "fixed_return": round(_safe_float(fk.get("total_return_pct", 0)), 2),
                "fixed_drawdown": round(_safe_float(fk.get("max_drawdown_pct", 0)), 2),
                "regime_sharpe": round(_safe_float(rk.get("sharpe", 0)), 4),
                "regime_return": round(_safe_float(rk.get("total_return_pct", 0)), 2),
                "regime_drawdown": round(_safe_float(rk.get("max_drawdown_pct", 0)), 2),
            })

        logger.info(f"    {year}: 完成")

    df_yearly = pd.DataFrame(yearly_results)
    df_yearly.to_csv(output_dir / "task2_yearly_validation.csv", index=False)

    # ── 2g: 参数对比表 ──
    from core.param_manager import V3RegimeParamManager
    param_mgr = V3RegimeParamManager()
    param_table = param_mgr.get_params_comparison_table()
    param_table.to_csv(output_dir / "task2_param_comparison.csv", index=False)

    # ── 2h: 汇总 ──
    summary_rows = []
    for sname in strategy_names:
        fk = train_fixed.get(sname, {})
        rk = train_regime.get(sname, {})
        summary_rows.append({
            "strategy": sname,
            "train_fixed_sharpe": round(_safe_float(fk.get("sharpe", 0)), 4),
            "train_regime_sharpe": round(_safe_float(rk.get("sharpe", 0)), 4),
            "train_fixed_dd": round(_safe_float(fk.get("max_drawdown_pct", 0)), 2),
            "train_regime_dd": round(_safe_float(rk.get("max_drawdown_pct", 0)), 2),
        })
    df_summary = pd.DataFrame(summary_rows)

    _plot_yearly_comparison(df_yearly, output_dir / "task2_yearly_comparison.png")
    _plot_env_distribution(env_stats, output_dir / "task2_env_distribution.png")

    return {
        "walkforward": wf_df,
        "out_of_sample": oos_df,
        "env_stats": env_stats,
        "yearly": df_yearly,
        "param_table": param_table,
        "summary": df_summary,
    }


def _build_validation_config(opt_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """构建与 run_full_backtest 兼容的配置字典。"""
    return {
        "backtest": {
            "initial_cash": opt_cfg["initial_cash"],
            "commission_rate": opt_cfg["commission_rate"],
            "slippage_rate": opt_cfg["slippage_rate"],
            "full_start_date": opt_cfg["full_start"],
            "full_end_date": opt_cfg["full_end"],
            "in_sample_end_date": opt_cfg["in_sample_end"],
            "out_sample_start_date": opt_cfg["out_sample_start"],
        },
        "symbols": opt_cfg["symbols"],
        "strategies": [{"name": s} for s in opt_cfg["strategy_names"]],
        "risk_management": {
            "stop_loss_pct": 0.05,
            "position_limit_pct": 0.4,
            "total_position_limit": 0.8,
        },
        "factor_weights": {},
        "monte_carlo": {
            "n_simulations": _N_MONTE_CARLO,
            "random_seed": _RANDOM_SEED,
        },
        "output": {"output_dir": str(opt_cfg["output_dir"])},
    }


def _compute_environment_stats(
    ds: PyBrokerDataSource,
    opt_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """计算各品种的环境分布统计（5类环境）。"""
    from scripts.analysis_runner import V3RegimeAwareRunner
    regime_runner = V3RegimeAwareRunner()
    all_stats = []

    for sym in opt_cfg["symbols"]:
        try:
            df = ds.to_pybroker_df()
            if df is None or df.empty:
                continue
            sym_df = df[df["symbol"] == sym] if "symbol" in df.columns else df
            if sym_df.empty or "close" not in sym_df.columns:
                continue
            if "high" not in sym_df.columns or "low" not in sym_df.columns:
                logger.warning(f"  缺少 high/low 列 ({sym})，跳过 v3 环境检测")
                continue

            df_with_regime = regime_runner.detect_regime_series(sym_df)
            dist = regime_runner.get_regime_distribution(df_with_regime)

            row = {"symbol": sym}
            all_regimes = [
                "trend_up", "trend_down", "range_bound", "high_volatility",
                "low_volatility",
            ]
            for regime in all_regimes:
                row[regime] = round(dist.get(regime, 0.0), 4)
            all_stats.append(row)
        except Exception as e:
            logger.warning(f"  环境统计计算失败 ({sym}): {e}")

    return pd.DataFrame(all_stats)


def _run_period_backtest(
    strategy_names: List[str],
    ds: PyBrokerDataSource,
    opt_cfg: Dict[str, Any],
    start: str,
    end: str,
    mode: str = "fixed",
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    执行某时段的回测，返回各策略KPI。

    mode="fixed": 固定参数（或优化参数）回测
    mode="regime": 环境感知回测，使用 RegimeAwareRunner 根据市场环境动态切换参数
    """
    results = {}

    if mode == "regime":
        from core.param_manager import V3RegimeParamManager
        from scripts.analysis_runner import V3RegimeAwareRunner
        param_manager = V3RegimeParamManager()
        regime_runner = V3RegimeAwareRunner(param_manager=param_manager)

        df = ds.to_pybroker_df()
        if df is None or df.empty:
            logger.warning("  环境感知回测跳过：数据为空")
            return results

        try:
            config = BacktestConfig(
                initial_cash=opt_cfg["initial_cash"],
                commission_rate=opt_cfg["commission_rate"],
                slippage_rate=opt_cfg["slippage_rate"],
            )
            runner = PyBrokerBacktestRunner(ds, config)
            runner.register_strategies(strategy_names)

            regime_result = regime_runner.run_with_regime_switch(
                runner, df, strategy_names, start, end,
            )
            regime_metrics = regime_result.get("metrics", {})
            if regime_metrics:
                results["regime_combo"] = dict(regime_metrics)
                logger.info(f"  环境感知回测完成: {len(strategy_names)}策略, 环境: {regime_result.get('regime', 'unknown')}")
        except Exception as e:
            logger.warning(f"  环境感知回测失败 ({start}~{end}): {e}")
            results["regime_combo"] = {}
        return results

    # mode == "fixed": 逐策略固定参数回测
    for sname in strategy_names:
        try:
            config = BacktestConfig(
                initial_cash=opt_cfg["initial_cash"],
                commission_rate=opt_cfg["commission_rate"],
                slippage_rate=opt_cfg["slippage_rate"],
            )
            runner = PyBrokerBacktestRunner(ds, config)
            runner.register_strategies([sname])

            custom_params = None
            if best_params and sname in best_params:
                custom_params = {sname: best_params[sname]}

            result = runner.run(start, end, custom_params=custom_params)
            results[sname] = dict(result.metrics)
        except Exception as e:
            logger.warning(f"  固定参数回测失败 ({sname}, {start}~{end}): {e}")
            results[sname] = {}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 任务3: 蒙特卡洛 1000 次鲁棒性测试
# ══════════════════════════════════════════════════════════════════════════════


def task3_monte_carlo(
    ds: PyBrokerDataSource,
    opt_cfg: Dict[str, Any],
    lib: StrategyLibrary,
    output_dir: Path,
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    蒙特卡洛 1000 次鲁棒性测试（P2-D 验收项）。

    调用 run_full_backtest.run_e9_monte_carlo 的标准接口，
    同时对每个策略单独执行蒙特卡洛并输出详细分布。
    """
    full_start = opt_cfg.get("full_start", _DEFAULT_TRAIN_START)
    full_end = opt_cfg.get("full_end", _DEFAULT_TEST_END)
    bankruptcy_threshold = opt_cfg.get("bankruptcy_threshold", _DEFAULT_BANKRUPTCY_THRESHOLD)

    logger.info("=" * 60)
    logger.info("任务3: 蒙特卡洛 1000 次鲁棒性测试")
    logger.info(f"  模拟次数: {_N_MONTE_CARLO}")
    logger.info(f"  破产阈值: {bankruptcy_threshold:.1%}")
    logger.info("=" * 60)

    strategy_names = list(opt_cfg["strategy_names"])

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 3a: 调用 run_full_backtest 的标准蒙特卡洛 ──
    val_config = _build_validation_config(opt_cfg)
    val_config["strategies"] = [{"name": s} for s in strategy_names]
    mc_base_df = run_e9_monte_carlo(ds, val_config, output_dir)

    # ── 3b: 逐策略蒙特卡洛详细分析 ──
    all_mc_results = {}

    for sname in strategy_names:
        logger.info(f"\n  策略: {sname}")

        try:
            config = BacktestConfig(
                initial_cash=opt_cfg["initial_cash"],
                commission_rate=opt_cfg["commission_rate"],
                slippage_rate=opt_cfg["slippage_rate"],
            )
            runner = PyBrokerBacktestRunner(ds, config)
            runner.register_strategies([sname])

            custom_params = None
            if best_params and sname in best_params:
                custom_params = {sname: best_params[sname]}

            result = runner.run(full_start, full_end, custom_params=custom_params)
            eq = result.equity_curve

            if eq is None or eq.empty:
                logger.warning(f"  {sname}: 无净值数据，跳过")
                continue

            eq_sorted = eq.sort_values("date")
            returns = eq_sorted["equity"].pct_change().dropna()
            returns = returns[returns.apply(_is_valid_number)]

            if len(returns) == 0:
                logger.warning(f"  {sname}: 无有效收益率，跳过")
                continue

            mc_result = _run_monte_carlo_sim(
                returns, n_simulations=_N_MONTE_CARLO, seed=_RANDOM_SEED,
                bankruptcy_threshold=bankruptcy_threshold,
            )
            all_mc_results[sname] = mc_result

            logger.info(f"    终值均值: {mc_result['final_mean']:.4f}")
            logger.info(f"    终值中位数: {mc_result['final_median']:.4f}")
            logger.info(f"    破产概率(终值<{bankruptcy_threshold:.1f}): {mc_result['bankruptcy_prob']:.2%}")
            logger.info(f"    最大回撤均值: {mc_result['avg_max_dd']:.2%}")
            logger.info(f"    月胜率均值: {mc_result['avg_monthly_win_rate']:.1%}")

        except Exception as e:
            logger.error(f"  {sname} 蒙特卡洛失败: {e}")

    # ── 3c: 汇总表 ──
    summary_rows = []
    for sname, mc in all_mc_results.items():
        summary_rows.append({
            "strategy": sname,
            "final_mean": round(mc["final_mean"], 4),
            "final_median": round(mc["final_median"], 4),
            "final_5pct": round(mc["final_5pct"], 4),
            "final_95pct": round(mc["final_95pct"], 4),
            "bankruptcy_prob": round(mc["bankruptcy_prob"], 4),
            "avg_max_dd": round(mc["avg_max_dd"], 4),
            "avg_monthly_win_rate": round(mc["avg_monthly_win_rate"], 4),
            "calmar_mean": round(mc["calmar_mean"], 4),
        })

    df_mc = pd.DataFrame(summary_rows)
    df_mc.to_csv(output_dir / "task3_monte_carlo_summary.csv", index=False)

    # 保存详细模拟数据
    for sname, mc in all_mc_results.items():
        detail = pd.DataFrame({
            "sim_id": range(_N_MONTE_CARLO),
            "final_value": mc["final_values"],
            "max_drawdown": mc["max_drawdowns"],
        })
        detail.to_csv(output_dir / f"task3_mc_detail_{sname}.csv", index=False)

    _plot_monte_carlo_distribution(all_mc_results, output_dir / "task3_mc_distribution.png")

    return {"summary": df_mc, "details": all_mc_results, "base": mc_base_df}


def _run_monte_carlo_sim(
    returns: pd.Series,
    n_simulations: int = 1000,
    seed: int = 42,
    bankruptcy_threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    执行蒙特卡洛模拟。

    Args:
        returns: 日收益率序列
        n_simulations: 模拟次数
        seed: 随机种子
        bankruptcy_threshold: 破产阈值（终值低于此比例视为破产）
    """
    rng = np.random.default_rng(seed)
    ret_array = returns.values
    n_days = len(ret_array)

    sim_equities = np.zeros((n_simulations, n_days + 1))
    sim_equities[:, 0] = 1.0

    for i in range(n_simulations):
        sampled = rng.choice(ret_array, size=n_days, replace=True)
        sim_equities[i, 1:] = np.cumprod(1.0 + sampled)

    final_values = sim_equities[:, -1]

    peak_equities = np.maximum.accumulate(sim_equities, axis=1)
    peak_safe = np.where(peak_equities > 0, peak_equities, 1.0)
    drawdowns = sim_equities / peak_safe - 1.0
    max_drawdowns = np.min(drawdowns, axis=1)

    # 使用 pandas 重采样计算月胜率（基于模拟曲线）
    monthly_win_rates = []
    for i in range(n_simulations):
        eq = pd.Series(sim_equities[i])
        if hasattr(returns, "index") and isinstance(returns.index, pd.DatetimeIndex):
            try:
                eq.index = returns.index
                monthly_eq = eq.resample("ME").last().dropna()
                monthly_ret = monthly_eq.pct_change().dropna()
            except (ValueError, TypeError):
                monthly_ret = pd.Series(dtype=float)
        else:
            monthly_ret = pd.Series(dtype=float)

        if len(monthly_ret) > 0:
            monthly_win_rates.append(float((monthly_ret > 0).mean()))
        else:
            n_months = max(1, n_days // 21)
            wins = 0
            for m in range(n_months):
                start_idx = m * 21
                end_idx = min((m + 1) * 21, n_days)
                if end_idx < n_days + 1 and eq.iloc[end_idx] > eq.iloc[start_idx]:
                    wins += 1
            monthly_win_rates.append(wins / n_months)

    monthly_win_rate = float(np.mean(monthly_win_rates)) if monthly_win_rates else 0.0

    annual_returns = (final_values ** (252 / n_days) - 1)
    calmar_ratios = annual_returns / np.abs(max_drawdowns + 1e-10)

    bankruptcy_prob = float(np.mean(final_values < bankruptcy_threshold))

    return {
        "final_values": final_values,
        "max_drawdowns": max_drawdowns,
        "final_mean": float(np.mean(final_values)),
        "final_median": float(np.median(final_values)),
        "final_5pct": float(np.percentile(final_values, 5)),
        "final_95pct": float(np.percentile(final_values, 95)),
        "bankruptcy_prob": bankruptcy_prob,
        "bankruptcy_threshold": bankruptcy_threshold,
        "avg_max_dd": float(np.mean(max_drawdowns)),
        "avg_monthly_win_rate": monthly_win_rate,
        "calmar_mean": float(np.mean(calmar_ratios)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 绘图函数
# ══════════════════════════════════════════════════════════════════════════════


def _plot_wf_comparison(df: pd.DataFrame, path: Path) -> None:
    """绘制 WalkForward 新旧配置对比图。"""
    try:
        if df.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        strategies = df["strategy"].tolist()
        x = np.arange(len(strategies))
        width = 0.35

        axes[0].bar(x - width / 2, df["new_avg_sharpe"], width, label="新配置", alpha=0.8)
        axes[0].bar(x + width / 2, df["old_avg_sharpe"], width, label="旧配置", alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(strategies, rotation=45, ha="right")
        axes[0].set_title("平均Sharpe对比")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(x - width / 2, df["new_min_sharpe"], width, label="新配置", alpha=0.8)
        axes[1].bar(x + width / 2, df["old_min_sharpe"], width, label="旧配置", alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(strategies, rotation=45, ha="right")
        axes[1].set_title("最低Sharpe对比")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  已保存: {path}")
    except Exception as e:
        logger.error(f"  绘图失败: {e}")


def _plot_wf_sensitivity(df: pd.DataFrame, path: Path) -> None:
    """绘制 WalkForward 窗口敏感性分析图。"""
    try:
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        for sname in df["strategy"].unique():
            sub = df[df["strategy"] == sname]
            ax.plot(sub["train_bars"], sub["avg_sharpe"], marker="o", label=sname)
        ax.set_xlabel("训练窗口长度（交易日）")
        ax.set_ylabel("平均Sharpe")
        ax.set_title("WalkForward 窗口敏感性分析")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  已保存: {path}")
    except Exception as e:
        logger.error(f"  绘图失败: {e}")


def _plot_yearly_comparison(df: pd.DataFrame, path: Path) -> None:
    """绘制按年验证对比图。"""
    try:
        if df.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for sname in df["strategy"].unique():
            sub = df[df["strategy"] == sname]
            axes[0].plot(sub["year"], sub["fixed_sharpe"], marker="o", label=f"{sname}(固定)")
            axes[0].plot(sub["year"], sub["regime_sharpe"], marker="s", linestyle="--", label=f"{sname}(环境)")
        axes[0].set_title("按年Sharpe对比")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        for sname in df["strategy"].unique():
            sub = df[df["strategy"] == sname]
            axes[1].plot(sub["year"], sub["fixed_drawdown"], marker="o", label=f"{sname}(固定)")
            axes[1].plot(sub["year"], sub["regime_drawdown"], marker="s", linestyle="--", label=f"{sname}(环境)")
        axes[1].set_title("按年最大回撤对比")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  已保存: {path}")
    except Exception as e:
        logger.error(f"  绘图失败: {e}")


def _plot_env_distribution(env_stats: pd.DataFrame, path: Path) -> None:
    """绘制环境分布图。"""
    try:
        if env_stats.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        regimes = [
            "trend_up", "trend_down", "range_bound", "high_volatility",
            "low_volatility", "breakout", "exhaustion_bull", "exhaustion_bear",
        ]
        x = env_stats["symbol"]
        bottom = np.zeros(len(x))
        for regime in regimes:
            if regime in env_stats.columns:
                vals = env_stats[regime].values
                ax.bar(x, vals, bottom=bottom, label=regime, alpha=0.7)
                bottom += vals
        ax.set_title("各品种市场环境分布")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  已保存: {path}")
    except Exception as e:
        logger.error(f"  绘图失败: {e}")


def _plot_monte_carlo_distribution(mc_results: Dict, path: Path) -> None:
    """绘制蒙特卡洛分布图。"""
    try:
        n = len(mc_results)
        if n == 0:
            return
        fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for i, (sname, mc) in enumerate(mc_results.items()):
            axes[i, 0].hist(mc["final_values"], bins=50, alpha=0.7, edgecolor="black")
            axes[i, 0].axvline(1.0, color="red", linestyle="--", label="盈亏平衡")
            axes[i, 0].set_title(f"{sname} 终值分布")
            axes[i, 0].legend()

            axes[i, 1].hist(mc["max_drawdowns"], bins=50, alpha=0.7, edgecolor="black", color="orange")
            axes[i, 1].set_title(f"{sname} 最大回撤分布")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  已保存: {path}")
    except Exception as e:
        logger.error(f"  绘图失败: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 因子IC稳定性与衰减分析（验证期集成）
# ══════════════════════════════════════════════════════════════════════════════


def _factor_ic_stability_analysis(
    ds: PyBrokerDataSource,
    opt_cfg: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    因子IC稳定性分析：对比训练期和验证期的因子IC变化。

    对每个品种独立计算因子得分、滚动IC和衰减状态，
    输出IC稳定性CSV和汇总图表。
    """
    train_start = opt_cfg.get("train_start", _DEFAULT_TRAIN_START)
    train_end = opt_cfg.get("train_end", _DEFAULT_TRAIN_END)
    test_start = opt_cfg.get("test_start", _DEFAULT_TEST_START)
    test_end = opt_cfg.get("test_end", _DEFAULT_TEST_END)
    symbols = opt_cfg["symbols"]
    factor_names = ["ts_momentum", "roll_yield", "alpha019", "alpha032"]

    logger.info("=" * 60)
    logger.info("因子IC稳定性分析（滚动IC + 衰减监控）")
    logger.info(f"  因子: {factor_names}")
    logger.info(f"  训练期: {train_start} ~ {train_end}")
    logger.info(f"  验证期: {test_start} ~ {test_end}")
    logger.info("=" * 60)

    ic_config = RollingICConfig(window=60, forward_period=5, ema_alpha=0.1, min_observations=30)
    decay_config = FactorDecayConfig(
        trend_window=40, ic_healthy_threshold=0.03, ic_dead_threshold=0.01,
        max_consecutive_decline=5, decay_slope_threshold=-0.001,
    )

    all_results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    for symbol in symbols:
        logger.info(f"  分析品种: {symbol}")
        try:
            sym_df = ds.query(train_start, test_end, symbols=[symbol])
            if sym_df is None or len(sym_df) < 60:
                logger.warning(f"    {symbol}: 数据不足，跳过")
                continue

            scored = _compute_factor_scores_from_ohlcv(sym_df)
            ic_engine = RollingICWeightEngine(ic_config)
            decay_monitor = FactorDecayMonitor(decay_config)

            for i in range(len(scored)):
                row = scored.iloc[i]
                forward_ret = float(row["forward_return"])
                if not _is_valid_number(forward_ret):
                    continue

                factor_scores = {
                    name: float(row.get(name, 0.0))
                    for name in factor_names
                    if _is_valid_number(row.get(name, 0.0))
                }
                if not factor_scores:
                    continue

                ic_engine.update(factor_scores, forward_ret)
                current_ic = ic_engine.current_ic
                for name, ic_val in current_ic.items():
                    if _is_valid_number(ic_val):
                        decay_monitor.update(name, ic_val)

            decay_monitor.check_decay()
            ic_summary = ic_engine.get_ic_summary()
            final_weights = ic_engine.get_dynamic_weights()

            for name, stats in ic_summary.items():
                status = decay_monitor.current_status.get(name, DecayStatus.HEALTHY).value
                summary_rows.append({
                    "symbol": symbol,
                    "factor": name,
                    "mean_ic": round(stats.get("mean", 0.0), 6),
                    "std_ic": round(stats.get("std", 0.0), 6),
                    "ir": round(stats.get("ir", 0.0), 4),
                    "current_ic": round(stats.get("current", 0.0), 6),
                    "current_weight": round(final_weights.get(name, 0.0), 4),
                    "decay_status": status,
                })

            # 分别统计训练期和验证期IC
            train_scored = scored[
                (scored["date"] >= train_start) & (scored["date"] <= train_end)
            ]
            test_scored = scored[
                (scored["date"] >= test_start) & (scored["date"] <= test_end)
            ]

            train_ic_engine = RollingICWeightEngine(ic_config)
            test_ic_engine = RollingICWeightEngine(ic_config)

            for _, row_data in train_scored.iterrows():
                fwd = float(row_data["forward_return"])
                if not _is_valid_number(fwd):
                    continue
                fscores = {
                    name: float(row_data.get(name, 0.0))
                    for name in factor_names
                    if _is_valid_number(row_data.get(name, 0.0))
                }
                if fscores:
                    train_ic_engine.update(fscores, fwd)

            for _, row_data in test_scored.iterrows():
                fwd = float(row_data["forward_return"])
                if not _is_valid_number(fwd):
                    continue
                fscores = {
                    name: float(row_data.get(name, 0.0))
                    for name in factor_names
                    if _is_valid_number(row_data.get(name, 0.0))
                }
                if fscores:
                    test_ic_engine.update(fscores, fwd)

            train_summary = train_ic_engine.get_ic_summary()
            test_summary = test_ic_engine.get_ic_summary()

            for name in factor_names:
                train_mean = train_summary.get(name, {}).get("mean", 0.0)
                test_mean = test_summary.get(name, {}).get("mean", 0.0)
                ic_drop = (test_mean - train_mean) / (abs(train_mean) + 1e-10) if abs(train_mean) > 1e-6 else 0.0
                logger.info(
                    f"    {name}: train_IC={train_mean:.4f}, test_IC={test_mean:.4f}, "
                    f"衰减={ic_drop:.1%}, status={decay_monitor.current_status.get(name, DecayStatus.HEALTHY).value}"
                )

            all_results[symbol] = {
                "ic_summary": ic_summary,
                "final_weights": final_weights,
                "decay_status": decay_monitor.current_status,
                "train_ic": train_summary,
                "test_ic": test_summary,
            }

        except Exception as e:
            logger.error(f"    {symbol} 因子分析失败: {e}")

    # 汇总CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        save_csv(summary_df, output_dir / "factor_ic_stability.csv")

        logger.info("\n  因子IC稳定性汇总:")
        for _, row_data in summary_df.iterrows():
            logger.info(
                f"    {row_data['symbol']}/{row_data['factor']}: "
                f"mean_IC={row_data['mean_ic']:.4f}, IR={row_data['ir']:.2f}, "
                f"weight={row_data['current_weight']:.4f}, status={row_data['decay_status']}"
            )

        # 图表
        _plot_factor_ic_stability(
            summary_df, output_dir / "factor_ic_stability.png",
        )

    return {"summary_rows": summary_rows, "details": all_results}


def _plot_factor_ic_stability(df: pd.DataFrame, path: Path) -> None:
    """绘制因子IC稳定性对比图。"""
    try:
        if df.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：各因子IC均值（按品种分组柱状）
        pivot_mean = df.pivot_table(
            values="mean_ic", index="symbol", columns="factor", aggfunc="mean",
        )
        if not pivot_mean.empty:
            pivot_mean.plot(kind="bar", ax=axes[0], alpha=0.8)
            axes[0].set_title("各因子IC均值（按品种）")
            axes[0].set_ylabel("Mean IC")
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis="x", rotation=45)

        # 右图：IC信息比率散点
        for sym in df["symbol"].unique():
            sub = df[df["symbol"] == sym]
            axes[1].scatter(sub["ir"], sub["current_weight"], label=sym, s=50, alpha=0.8)
        axes[1].set_xlabel("IC IR（信息比率）")
        axes[1].set_ylabel("当前权重")
        axes[1].set_title("因子IR vs 动态权重")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  已保存: {path}")
    except Exception as e:
        logger.error(f"  绘图失败: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════


def main(
    config_path: str = "config.yaml",
    tasks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    主函数：参数优化 + WalkForward + 样本外 + 蒙特卡洛。

    流程：
      [1/4] 加载数据（tqsdk 优先）
      [2/4] 任务1: 参数优化 + WalkForward 搜索网格验证
      [3/4] 任务2: 2016-2020 训练 / 2021-2025 样本外验证
      [4/4] 任务3: 蒙特卡洛 1000 次鲁棒性测试

    Args:
        config_path: 配置文件路径
        tasks: 要执行的任务列表，None表示全部执行
               可选: ["task1", "task2", "task3"]
    """
    logger.info("=" * 80)
    logger.info("  验证脚本 v2：参数优化 + WalkForward + 样本外 + 蒙特卡洛")
    logger.info(f"  开始: {datetime.now()}")
    logger.info("=" * 80)

    cfg = load_config(config_path)
    opt_cfg = _build_opt_cfg(cfg)
    output_dir = Path(opt_cfg["output_dir"]) / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    lib = StrategyLibrary()

    train_start = opt_cfg.get("train_start", _DEFAULT_TRAIN_START)
    train_end = opt_cfg.get("train_end", _DEFAULT_TRAIN_END)
    test_start = opt_cfg.get("test_start", _DEFAULT_TEST_START)
    test_end = opt_cfg.get("test_end", _DEFAULT_TEST_END)
    logger.info(f"  训练期: {train_start} ~ {train_end}")
    logger.info(f"  验证期: {test_start} ~ {test_end}")

    # ── [1/4] 加载数据 ──
    ds = _load_data(cfg, opt_cfg)

    if tasks is None:
        tasks = ["task1", "task2", "task3"]

    results: Dict[str, Any] = {}
    best_params: Optional[Dict[str, Dict[str, Any]]] = None

    # ── [2/4] 任务1: 参数优化 + WalkForward ──
    if "task1" in tasks:
        logger.info("\n" + "=" * 80)
        t1_result = task1_walkforward_grid(ds, opt_cfg, lib, output_dir)
        results["task1"] = t1_result
        best_params = t1_result.get("best_params")

        # 将优化参数应用到 StrategyLibrary
        if best_params:
            logger.info("\n  应用优化参数到 StrategyLibrary:")
            for sname, params in best_params.items():
                lib.update_default_params(sname, params)
                logger.info(f"    {sname}: {params}")

    # ── [3/4] 任务2: 样本外验证 ──
    if "task2" in tasks:
        logger.info("\n" + "=" * 80)
        results["task2"] = task2_train_test_split(
            ds, opt_cfg, lib, output_dir, best_params=best_params,
        )

    # ── [4/4] 任务3: 蒙特卡洛 ──
    if "task3" in tasks:
        logger.info("\n" + "=" * 80)
        results["task3"] = task3_monte_carlo(
            ds, opt_cfg, lib, output_dir, best_params=best_params,
        )

    # ── 因子IC稳定性与衰减分析 ──
    logger.info("\n" + "=" * 80)
    results["factor_ic"] = _factor_ic_stability_analysis(ds, opt_cfg, output_dir)

    # ── 汇总报告 ──
    results["_train_end"] = train_end
    results["_test_start"] = test_start
    results["_full_start"] = opt_cfg.get("full_start", _DEFAULT_TRAIN_START)
    results["_full_end"] = opt_cfg.get("full_end", _DEFAULT_TEST_END)
    _print_summary(results, output_dir)

    # ── 生成HTML分析报告 ──
    logger.info("\n" + "=" * 60)
    logger.info("  生成验证分析报告...")
    try:
        from core.report_builder import generate_report as build_report
        report_path = build_report(
            output_dir=str(output_dir),
            title="量化回测验证分析报告",
            subtitle=f"WalkForward + 样本外验证 + 蒙特卡洛 · {datetime.now().strftime('%Y-%m-%d')}",
            report_name="validation_report.html",
        )
        logger.info(f"  验证报告已生成: {report_path}")
    except Exception as e:
        logger.error(f"  验证报告生成失败: {e}")

    logger.info("\n" + "=" * 80)
    logger.info(f"  验证完成: {datetime.now()}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info("=" * 80)

    return results


def _print_summary(results: Dict[str, Any], output_dir: Path) -> None:
    """输出验证汇总。"""
    logger.info("\n" + "=" * 60)
    logger.info("  验证汇总")
    logger.info("=" * 60)

    if "task1" in results:
        t1 = results["task1"]
        compare = t1.get("compare", pd.DataFrame())
        if not compare.empty:
            logger.info("\n  任务1 - WalkForward 对比:")
            for _, row in compare.iterrows():
                logger.info(
                    f"    {row['strategy']}: "
                    f"新avg={row.get('new_avg_sharpe', 'N/A')}, "
                    f"旧avg={row.get('old_avg_sharpe', 'N/A')}, "
                    f"比值={row.get('avg_ratio', 'N/A')}, "
                    f"OOS评分={row.get('oos_priority_score', 'N/A')}"
                )

    if "task2" in results:
        t2 = results["task2"]
        yearly = t2.get("yearly", pd.DataFrame())
        if not yearly.empty:
            logger.info("\n  任务2 - 按年验证:")
            for sname in yearly["strategy"].unique():
                sub = yearly[yearly["strategy"] == sname]
                avg_fixed = sub["fixed_sharpe"].mean()
                avg_regime = sub["regime_sharpe"].mean()
                logger.info(f"    {sname}: 固定avg={avg_fixed:.4f}, 环境avg={avg_regime:.4f}")

    if "task3" in results:
        t3 = results["task3"]
        mc_summary = t3.get("summary", pd.DataFrame())
        if not mc_summary.empty:
            logger.info("\n  任务3 - 蒙特卡洛:")
            for _, row in mc_summary.iterrows():
                logger.info(
                    f"    {row['strategy']}: "
                    f"终值均值={row.get('final_mean', 'N/A')}, "
                    f"破产概率={row.get('bankruptcy_prob', 'N/A')}, "
                    f"回撤均值={row.get('avg_max_dd', 'N/A')}"
                )

    # 保存汇总到文件
    summary_path = output_dir / "validation_summary.txt"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            full_start = results.get("_full_start", _DEFAULT_TRAIN_START)
            full_end = results.get("_full_end", _DEFAULT_TEST_END)
            train_end_str = results.get("_train_end", _DEFAULT_TRAIN_END)
            test_start_str = results.get("_test_start", _DEFAULT_TEST_START)
            f.write(f"验证完成时间: {datetime.now()}\n")
            f.write(f"训练期: {full_start} ~ {train_end_str}\n")
            f.write(f"验证期: {test_start_str} ~ {full_end}\n\n")

            if "task1" in results:
                compare = results["task1"].get("compare", pd.DataFrame())
                if not compare.empty:
                    f.write("任务1 - WalkForward 对比:\n")
                    f.write(compare.to_string(index=False))
                    f.write("\n\n")

            if "task3" in results:
                mc_summary = results["task3"].get("summary", pd.DataFrame())
                if not mc_summary.empty:
                    f.write("任务3 - 蒙特卡洛:\n")
                    f.write(mc_summary.to_string(index=False))
                    f.write("\n")

        logger.info(f"  汇总已保存: {summary_path}")
    except Exception as e:
        logger.error(f"  汇总保存失败: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="验证脚本 v2：参数优化 + WalkForward + 样本外 + 蒙特卡洛")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--tasks", nargs="+", default=None,
                        choices=["task1", "task2", "task3"],
                        help="要执行的任务（默认全部）")
    args = parser.parse_args()
    main(config_path=args.config, tasks=args.tasks)
