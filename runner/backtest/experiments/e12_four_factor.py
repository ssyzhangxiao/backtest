"""
实验 E12：四因子 CTA 融合回测（动量 + 期限结构 + 基差动量 + 仓单变化率）。

复用 e2_e3_fusion 的 _run_weighted_fusion 框架，差异：
  - strategies=["four_factor"]（单策略）
  - 不传 use_execute_fusion（单策略不需要融合）
  - 支持 use_receipt 开关（无仓单回退到 3 因子）
  - 通过 ReceiptFetcher.fetch_range 预拉取仓单数据

E12 同时输出两个变体：
  - e12_with_receipt：四因子（启用仓单）
  - e12_no_receipt：三因子（关闭仓单，对照组）

调用范式：
  from runner import Pipeline
  pipe = Pipeline("config.yaml").load_data()
  pipe.run_experiments(["e12"])  # 一次跑两个变体
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from core.engine.pybroker_data_source import PyBrokerDataSource
from core.execution.four_factor_indicators import register_four_factor_indicators
from runner.backtest.runner import get_pybroker_runner, safe_run_backtest
from runner.common.utils import (
    format_metrics,
    handle_backtest_errors,
    sanitize_filename,
    save_csv,
    save_equity_curve,
)


# ═══════════════════════════════════════════════════════════════
# 单策略回测核心（区别于 _run_weighted_fusion：单策略无需融合）
# ═══════════════════════════════════════════════════════════════


def _run_four_factor_single(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
    use_receipt: bool,
    four_factor_weights: Optional[Dict[str, float]] = None,
    basis_window: int = 20,
    receipt_window: int = 20,
) -> pd.DataFrame:
    """
    四因子单策略回测（每个品种独立回测，结果聚合到一张表）。

    Args:
        data_source: PyBrokerDataSource
        config: 原始配置字典
        output_dir: 输出目录
        use_receipt: 是否启用仓单因子
        four_factor_weights: 四因子权重（None 时从 config.four_factor.weights 读）
        basis_window: 基差动量窗口
        receipt_window: 仓单变化率窗口

    Returns:
        每行 = 一个品种的指标 DataFrame
    """
    if use_receipt:
        label = "E12_四因子"
        prefix = "e12"
    else:
        label = "E12_四因子无仓单"
        prefix = "e12_no_receipt"
    logger.info(f"执行{label}实验")

    register_four_factor_indicators()  # 确保已注册

    # 1. 拉取仓单数据（如启用）
    receipt_data_map: Dict[str, Any] = {}
    if use_receipt:
        receipt_data_map = _fetch_receipt_for_symbols(
            config, data_source.symbols,
        )
        logger.info(
            f"[E12 with receipt] 仓单数据：{len(receipt_data_map)} 品种，"
            f"keys={list(receipt_data_map.keys())[:5]}",
        )
    else:
        logger.info(f"[E12 no receipt] 不加载仓单（{label}）")

    # 2. 构造 sub_params（透传给 PyBroker 指标构建器）
    weights = four_factor_weights or config.get("four_factor", {}).get("weights", {})
    custom_params: Dict[str, Dict[str, Any]] = {
        "four_factor": {
            "weights": dict(weights),
            "basis_window": basis_window,
            "receipt_window": receipt_window,
            "receipt_data": receipt_data_map,
        },
    }
    # 注入到 config 中，供 get_pybroker_runner 透传
    config = dict(config)
    config["_custom_strategy_params"] = custom_params

    # 3. 逐品种回测
    symbols: List[str] = config.get("symbols", [])
    bt_cfg = config["backtest"]
    all_results: List[Dict[str, Any]] = []

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        # 每品种独立 custom_params（含该品种的 receipt_data）
        per_sym_custom: Dict[str, Dict[str, Any]] = {
            "four_factor": {
                **custom_params["four_factor"],
                "symbol": sym,  # 关键：让 builder 闭包捕获该品种
            },
        }
        runner = get_pybroker_runner(
            data_source, config,
            strategies=["four_factor"],
            target_symbols=[sym],
        )
        # 注入 custom_params（单品种也支持）
        if hasattr(runner, "set_custom_params"):
            runner.set_custom_params(per_sym_custom)
        else:
            runner._custom_params = per_sym_custom
        # 2026-06-19：手动将仓单数据预加载到 runner 的 factor_pool，
        # 解决 dict-style config 下 four_factor_enabled 不可识别的问题。
        if use_receipt and receipt_data_map:
            _inject_receipt_to_runner(runner, receipt_data_map)

        result = safe_run_backtest(
            runner,
            bt_cfg["full_start_date"],
            bt_cfg["full_end_date"],
            f"{label[:3]}_{sym}",
            use_execute_fusion=False,  # 单策略不需要融合
        )

        if result is None:
            all_results.append(
                {
                    "symbol": sym, "strategy": "four_factor",
                    "experiment": label, "error": "回测失败",
                },
            )
            continue

        m = format_metrics(result.metrics)
        result_row: Dict[str, Any] = {
            "symbol": sym, "strategy": "four_factor",
            "experiment": label, "error": None,
        }
        result_row.update(m)
        all_results.append(result_row)

        logger.info(
            f"  {label} {sym}: return={m.get('total_return_pct', 'N/A')} "
            f"sharpe={m.get('sharpe', 'N/A')}",
        )

        eq = result.equity_curve
        if eq is not None and not eq.empty:
            save_equity_curve(
                eq.assign(symbol=sym, four_factor_mode=label),
                output_dir,
                f"{prefix}_equity_{sanitize_filename(sym.replace('.', '_'))}",
            )

    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    save_csv(
        df,
        output_dir / f"{prefix}_metrics.csv",
    )
    return df


def _inject_receipt_to_runner(runner: Any, receipt_data_map: Dict[str, Any]) -> None:
    """
    将仓单数据预加载到 runner 的 factor_pool。

    兼容两种 runner 结构：
      - PyBrokerBacktestRunner: 通过 _blueprint_builder.signal_abstraction.pool
      - 其他: 通过 _signal_abstraction.pool / factor_pool
    """
    pool = None
    bb = getattr(runner, "_blueprint_builder", None)
    if bb is not None:
        sa = getattr(bb, "signal_abstraction", None)
        if sa is not None:
            pool = getattr(sa, "pool", None)
    if pool is None:
        sa = getattr(runner, "_signal_abstraction", None) or getattr(runner, "signal_abstraction", None)
        if sa is not None:
            pool = getattr(sa, "pool", None)
    if pool is None:
        pool = getattr(runner, "factor_pool", None)
    if pool is None:
        logger.warning("  [E12] runner 无 factor_pool，跳过 receipt 预加载")
        return
    try:
        pool.preload_receipt_data(receipt_data=receipt_data_map, receipt_window=20, basis_window=20)
        logger.info(f"  [E12] 已注入仓单数据：{len(receipt_data_map)} 品种")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"  [E12] preload_receipt_data 失败: {e}")


def _fetch_receipt_for_symbols(
    config: Dict[str, Any],
    symbols: List[str],
) -> Dict[str, Any]:
    """
    拉取或读取仓单缓存，返回 {symbol: pd.Series}。

    优先读本地缓存，缺失时尝试 AKShare 拉取。
    拉取失败时返回空字典（receipt_change 信号将保持 0，回退到 3 因子）。
    """
    cache_dir = Path(
        config.get("four_factor", {}).get(
            "receipt_cache_dir", "data/receipt_cache",
        ),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. 优先读本地缓存（只要有数据就用，不强求 50% 覆盖率）
    try:
        from core.data.receipt_fetcher import load_receipt_cache

        cached = load_receipt_cache(cache_dir, symbols=symbols)
        if cached:  # 至少 1 个品种有缓存 → 优先用缓存（缺失品种保持空，回退 3 因子）
            logger.info(
                f"[四因子] 仓单缓存命中：{len(cached)}/{len(symbols)} 品种",
            )
            return cached
    except Exception as e:  # noqa: BLE001
        logger.debug(f"读仓单缓存失败: {e}")

    # 2. 尝试 AKShare 拉取（生产环境）
    try:
        from core.data.receipt_fetcher import ReceiptFetcher
        from datetime import datetime, timedelta

        bt = config.get("backtest", {})
        end_date = bt.get("full_end_date", "2026-05-31")
        start_date = bt.get("full_start_date", "2020-01-01")
        # 限制最近 2 年避免拉取过久
        end_dt = pd.Timestamp(end_date)
        start_dt = max(
            pd.Timestamp(start_date),
            end_dt - timedelta(days=730),
        )
        fetcher = ReceiptFetcher(config={
            "cache_dir": str(cache_dir),
            "enable_online": False,  # 沙盒默认在线拉取关闭（与 receipt.enable_online 对齐）
            "cache_ttl_days": 7,
            "request_interval_min": 1.0,
            "request_interval_max": 4.0,
            "retry_times": 3,
        })
        df = fetcher.fetch_range(
            symbols=symbols,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
        )
        if df is None or df.empty:
            return {}
        out: Dict[str, Any] = {}
        for sym in symbols:
            series = ReceiptFetcher.get_receipt_series(sym, df)
            if not series.empty:
                out[sym] = series
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"[四因子] AKShare 拉取仓单失败（{e}），回退到 3 因子",
        )
        return {}


# ═══════════════════════════════════════════════════════════════
# 实验入口（与 e1~e11 同款签名）
# ═══════════════════════════════════════════════════════════════


@handle_backtest_errors(return_value=pd.DataFrame())
def run_e12_four_factor(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E12：四因子 CTA 融合回测。

    一次性跑两个变体（with_receipt / no_receipt），合并返回一张表。
    """
    logger.info("E12：四因子 CTA 融合回测（双变体：with/without receipt）")

    cfg = config.get("four_factor", {}) or {}
    weights = cfg.get("weights", {})
    basis_window = int(cfg.get("basis_window", 20))
    receipt_window = int(cfg.get("receipt_window", 20))

    # 变体 1：启用仓单
    df_with = _run_four_factor_single(
        data_source=data_source,
        config=config,
        output_dir=output_dir,
        use_receipt=True,
        four_factor_weights=weights,
        basis_window=basis_window,
        receipt_window=receipt_window,
    )

    # 变体 2：关闭仓单（对照组）
    df_no = _run_four_factor_single(
        data_source=data_source,
        config=config,
        output_dir=output_dir,
        use_receipt=False,
        four_factor_weights=weights,
        basis_window=basis_window,
        receipt_window=receipt_window,
    )

    # 合并
    combined = pd.concat([df_with, df_no], ignore_index=True)
    save_csv(combined, output_dir / "e12_combined_metrics.csv")
    return combined
