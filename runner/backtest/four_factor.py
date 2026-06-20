"""四因子 CTA 回测模块（2026-06-19）。

执行流程：
  1. 加载四因子权重（来自 config.yaml 的 four_factor 段）
  2. 加载仓单数据（AKShare ReceiptFetcher，并行接入）
  3. 构造 signal_abstraction.SignalAbstractionLayer 实例
  4. 通过 Pipeline.run_backtest("e1") 委托现有基线回测框架执行
  5. 收集指标 → 与 6 策略基线对比

返回：
  dict，含 metrics / per_symbol / mode

对比报告：
  6 策略基线 vs 四因子（含/不含仓单）→ Sharpe / 年化 / MDD 三表
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.data.receipt_fetcher import ReceiptFetcher
from core.execution.signal_abstraction import SignalAbstractionLayer
from core.execution.factor_pool import UnifiedFactorPool


def _convert_to_symbol_dict(data: Any) -> Dict[str, pd.DataFrame]:
    """将 PyBrokerDataSource / 单 DataFrame 转为 {symbol: ohlcv DataFrame}。

    支持输入：
      - PyBrokerDataSource：按 symbol 列拆分
      - Dict[symbol, DataFrame]：原样返回
      - DataFrame：按 symbol 列拆分（若存在）
    """
    if data is None:
        return {}
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if v is not None}
    # PyBrokerDataSource / DataFrame：含 symbol 列
    if hasattr(data, "to_pybroker_df"):
        df = data.to_pybroker_df()
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        return {}
    if df is None or df.empty or "symbol" not in df.columns:
        return {}
    return {
        sym: sub.drop(columns=["symbol"]).reset_index(drop=True)
        for sym, sub in df.groupby("symbol")
    }


def _has_far_close(data: Any, symbol: str) -> bool:
    """检查品种是否有 far_close 列（基差动量依赖）。"""
    if data is None:
        return False
    if isinstance(data, dict):
        df = data.get(symbol)
    elif isinstance(data, pd.DataFrame):
        df = data[data["symbol"] == symbol] if "symbol" in data.columns else data
        if not df.empty:
            df = df.drop(columns=["symbol"])
    else:
        df = None
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    return "far_close" in df.columns and df["far_close"].notna().sum() > 0


def _load_receipt_data(
    config: BacktestConfig,
    symbols: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, pd.Series]:
    """加载所有品种的仓单数据（缺失品种返回空 Series）。"""
    cache_dir = config.four_factor_receipt_cache_dir
    receipt_data: Dict[str, pd.Series] = {}
    try:
        fetcher = ReceiptFetcher(cache_dir=cache_dir)
    except ImportError as e:
        logger.warning("AKShare 不可用，跳过仓单数据加载: %s", e)
        return {sym: pd.Series(dtype=float) for sym in symbols}

    for sym in symbols:
        try:
            df = fetcher.fetch_range(
                symbols=[sym], start_date=start_date, end_date=end_date,
            )
            if df is None or df.empty:
                receipt_data[sym] = pd.Series(dtype=float)
                continue
            sub = df[df["symbol"] == sym] if "symbol" in df.columns else df
            if "date" in sub.columns and "receipt" in sub.columns:
                series = pd.Series(
                    sub["receipt"].values,
                    index=pd.to_datetime(sub["date"]),
                    name=sym,
                ).sort_index()
            else:
                series = pd.Series(dtype=float, name=sym)
            receipt_data[sym] = series
        except Exception as e:
            logger.warning("加载 %s 仓单数据失败: %s", sym, e)
            receipt_data[sym] = pd.Series(dtype=float)
    return receipt_data


def prepare_four_factor_layer(
    config: BacktestConfig,
    data: Dict[str, pd.DataFrame],
    use_receipt: bool = True,
) -> Dict[str, Any]:
    """构造四因子融合所需的全部组件（不执行回测）。

    Args:
        config: BacktestConfig
        data: {symbol: ohlcv DataFrame}
        use_receipt: 是否启用仓单因子

    Returns:
        {
          "factor_pool": UnifiedFactorPool,
          "signal_layer": SignalAbstractionLayer,
          "per_symbol_weights": {symbol: {factor: weight}},
          "mode": "four_factor" | "three_factor" | "two_factor",
          "use_receipt": bool,
        }
    """
    if not config.four_factor_enabled:
        raise ValueError("config.four_factor_enabled=False，请先启用四因子")

    # 转换为 {symbol: ohlcv DataFrame} 格式（支持 PyBrokerDataSource）
    data_dict = _convert_to_symbol_dict(data)
    symbols = list(data_dict.keys())
    if not symbols:
        raise ValueError("无有效数据：data 既不是 dict 也不含 symbol 列")
    base_weights = config.four_factor_weights or {
        "donchian_breakout": 0.30,
        "carry": 0.25,
        "basis_momentum": 0.25,
        "receipt_change": 0.20,
    }

    # 1. 加载仓单数据
    receipt_data: Dict[str, pd.Series] = {}
    if use_receipt:
        try:
            start = config.train_start
            end = config.test_end
            receipt_data = _load_receipt_data(config, symbols, start, end)
        except Exception as e:
            logger.warning("仓单数据加载失败，回退到 3 因子: %s", e)
            use_receipt = False

    # 2. 构造 factor_pool + signal_abstraction
    factor_pool = UnifiedFactorPool(config)
    factor_pool.preload_receipt_data(
        receipt_data=receipt_data,
        receipt_window=config.four_factor_receipt_window,
        basis_window=config.four_factor_basis_window,
    )
    sig_layer = SignalAbstractionLayer(
        factor_pool=factor_pool,
        xs_position_base=0.25,  # 方向二：b=0.25
        xs_position_ceiling=1.0,
        xs_opposite_penalty=0.4,  # 方向二：p=0.4
    )

    # 3. 按品种数据可用性调整权重
    per_symbol_weights: Dict[str, Dict[str, float]] = {}
    available_count = {"basis": 0, "receipt": 0}
    for sym in symbols:
        has_basis = _has_far_close(data_dict, sym)
        has_receipt = use_receipt and sym in receipt_data and not receipt_data[sym].empty
        per_symbol_weights[sym] = SignalAbstractionLayer.compute_four_factor_weights(
            has_basis=has_basis, has_receipt=has_receipt, base_weights=base_weights,
        )
        if has_basis:
            available_count["basis"] += 1
        if has_receipt:
            available_count["receipt"] += 1

    # 4. 判定 mode
    if available_count["basis"] == 0 and available_count["receipt"] == 0:
        mode = "two_factor"
    elif available_count["receipt"] == 0:
        mode = "three_factor"
    else:
        mode = "four_factor"

    logger.info(
        "[四因子] 准备完成：mode=%s, basis可用=%d, receipt可用=%d",
        mode, available_count["basis"], available_count["receipt"],
    )

    return {
        "factor_pool": factor_pool,
        "signal_layer": sig_layer,
        "per_symbol_weights": per_symbol_weights,
        "mode": mode,
        "use_receipt": use_receipt,
    }


def run_four_factor_backtest(
    config: BacktestConfig,
    data: Dict[str, pd.DataFrame],
    raw_config: Optional[Dict[str, Any]] = None,
    use_receipt: bool = True,
) -> Dict[str, Any]:
    """执行四因子 CTA 回测（规则 17：委托 core/ 不重写回测）。

    实际回测执行委托给 runner.backtest.experiments.run_experiment("e1")，
    本函数仅构造四因子融合所需的 signal_abstraction layer 与 per_symbol 权重，
    并返回可被外部 e1 实验框架消费的字典。

    Args:
        config: BacktestConfig 实例
        data: {symbol: ohlcv DataFrame} 数据源
        raw_config: 原始 yaml dict
        use_receipt: 是否启用仓单因子

    Returns:
        {
          "mode": "four_factor" | "three_factor" | "two_factor",
          "use_receipt": bool,
          "weights_per_symbol": {symbol: {factor: weight}},
          "factor_pool": UnifiedFactorPool,
          "signal_layer": SignalAbstractionLayer,
          "metrics": {sharpe, annual_return, max_drawdown}（占位，实际由 e1 框架填充）,
        }
    """
    prepared = prepare_four_factor_layer(config, data, use_receipt=use_receipt)
    return {
        "mode": prepared["mode"],
        "use_receipt": prepared["use_receipt"],
        "weights_per_symbol": prepared["per_symbol_weights"],
        "factor_pool": prepared["factor_pool"],
        "signal_layer": prepared["signal_layer"],
        "metrics": {"sharpe": 0.0, "annual_return": 0.0, "max_drawdown": 0.0},
    }


def build_comparison_report(
    baseline_result: Dict[str, Any],
    four_factor_result: Dict[str, Any],
    four_factor_no_receipt: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """构建 6 策略基线 vs 四因子 对比报告。

    Returns:
        DataFrame：index=[sharpe/annual_return/max_drawdown]，
                  columns=[baseline_6strat, four_factor, four_factor_no_receipt]
    """
    rows = {
        "sharpe": {
            "baseline_6strat": baseline_result.get("metrics", {}).get("sharpe", 0.0),
            "four_factor": four_factor_result.get("metrics", {}).get("sharpe", 0.0),
        },
        "annual_return": {
            "baseline_6strat": baseline_result.get("metrics", {}).get("annual_return", 0.0),
            "four_factor": four_factor_result.get("metrics", {}).get("annual_return", 0.0),
        },
        "max_drawdown": {
            "baseline_6strat": baseline_result.get("metrics", {}).get("max_drawdown", 0.0),
            "four_factor": four_factor_result.get("metrics", {}).get("max_drawdown", 0.0),
        },
    }
    if four_factor_no_receipt is not None:
        rows["sharpe"]["four_factor_no_receipt"] = (
            four_factor_no_receipt.get("metrics", {}).get("sharpe", 0.0)
        )
        rows["annual_return"]["four_factor_no_receipt"] = (
            four_factor_no_receipt.get("metrics", {}).get("annual_return", 0.0)
        )
        rows["max_drawdown"]["four_factor_no_receipt"] = (
            four_factor_no_receipt.get("metrics", {}).get("max_drawdown", 0.0)
        )
    return pd.DataFrame.from_dict(rows, orient="index")


__all__ = [
    "prepare_four_factor_layer",
    "run_four_factor_backtest",
    "build_comparison_report",
]
