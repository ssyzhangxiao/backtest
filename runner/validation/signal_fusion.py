"""
多策略信号融合（固定权重 + 月度滚动夏普更新）。

组合信号 = 0.3*carry + 0.3*vol + 0.2*donchian + 0.1*momentum + 0.05*tsi + 0.05*pair

两种模式:
  - "fixed": 使用上述固定权重（默认）
  - "rolling_sharpe": 按月度滚动窗口的 Sharpe 比率动态更新权重

每个策略独立运行后提取日收益率序列，按权重融合后计算组合绩效。

委托 scripts/run_cta_batch._run_single 执行回测，利用 pybroker.Strategy 的
backtest() 方法获取完整 equity_curve 以支持滚动权重更新。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from runner.common.utils import sanitize_filename

# ── 固定权重（与用户要求一致） ──
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "carry": 0.30,
    "vol_mean_reversion": 0.30,
    "donchian_breakout": 0.20,
    "momentum_ma": 0.10,
    "tsi_garch": 0.05,
    "pair_trading": 0.05,
}

# 默认6策略及其参数
_DEFAULT_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "carry": {"lookback": 60, "entry_z": 1.2, "direction": "long_only", "ema_alpha": 0.3, "use_slope": True},
    "vol_mean_reversion": {"vol_window": 20, "lookback": 252, "entry_z": 1.2, "vol_percentile": 0.7},
    "donchian_breakout": {"entry_lookback": 20, "atr_window": 14, "atr_entry_mult": 0.5, "trend_filter_ma": 60, "momentum_factor": 0.2},
    "momentum_ma": {"rsi_window": 14, "momentum_fast": 5, "momentum_mid": 20, "momentum_slow": 60},
    "tsi_garch": {"reg_window": 60, "min_obs": 30, "t_stat_threshold": 1.5, "cache_update_freq": 5},
    "pair_trading": {"lookback": 60, "entry_z": 2.0, "ols_window": 90, "adf_interval": 20},
}

# ── 滚动夏普参数 ──
_SHARPE_WINDOW = 21  # 月度滚动（约 21 个交易日）
_SHARPE_MIN_PERIODS = 63  # 最少 3 个月数据才开始更新
_WEIGHT_SMOOTH = 0.3  # 权重平滑系数（防止突变）


def _run_single_with_equity(
    symbol: str,
    strategy_name: str,
    strategy_params: Dict[str, Any],
    entry_threshold: float = 0.05,
    initial_cash: float = 1_000_000,
    full_start: str = "2020-01-01",
    test_start: str = "2023-01-01",
) -> Optional[pd.Series]:
    """运行单个策略回测并返回日收益率序列。"""
    try:
        import pybroker
        from pybroker import StrategyConfig
    except ImportError:
        return None

    from core.engine.cta_executor_builder import CTAExecutorBuilder
    from core.strategies.cta.registry import get_cta_strategy

    # 加载数据
    from scripts.run_cta_batch import _load_symbol_data, _WARMUP, _STRATEGY_EXIT_PARAMS, _DEFAULT_EXIT_PARAMS

    df = _load_symbol_data(symbol)
    if df is None:
        return None
    from core.engine.pybroker_data_source import PyBrokerDataSource
    ds = PyBrokerDataSource(df)
    try:
        symbol_data = ds.for_symbol(symbol)
    except ValueError:
        return None
    df_pb = symbol_data.to_pybroker_df()
    if len(df_pb) < 100:
        return None

    try:
        cta = get_cta_strategy(strategy_name, strategy_params)
    except ValueError:
        return None

    # 注入 spread 数据
    if "spread" in df_pb.columns:
        spread_arr = df_pb["spread"].to_numpy(dtype=float, copy=True)
        cta.set_state(symbol, "_spread", spread_arr)
    elif "far_close" in df_pb.columns:
        far = df_pb["far_close"].to_numpy(dtype=float, copy=True)
        close_val = df_pb["close"].to_numpy(dtype=float, copy=True)
        spread_synth = np.where(
            np.isfinite(far) & np.isfinite(close_val) & (close_val > 0),
            (far - close_val) / close_val * 100, np.nan,
        )
        cta.set_state(symbol, "_spread", spread_synth)

    # 退出参数
    ep = {**(dict(_DEFAULT_EXIT_PARAMS))}
    if strategy_name in _STRATEGY_EXIT_PARAMS:
        for k, v in _STRATEGY_EXIT_PARAMS[strategy_name].items():
            ep[k] = v

    builder = CTAExecutorBuilder(
        cta_strategy=cta,
        entry_threshold=entry_threshold,
        max_position_pct=0.3,
        max_holding_days=ep["max_holding_days"],
        atr_stop_multiple=ep["atr_stop_multiple"],
        atr_window=ep["atr_window"],
        stop_loss_pct=ep["stop_loss_pct"],
        global_risk_pct=ep["global_risk_pct"],
        risk_per_trade=ep.get("risk_per_trade", 0.01),
        target_vol=ep.get("target_vol", 0.0),
    )
    executor_fn = builder.build()

    pb_config = StrategyConfig(initial_cash=initial_cash)
    strategy = pybroker.Strategy(df_pb, full_start, test_start, config=pb_config)
    strategy.add_execution(executor_fn, symbols=[symbol])

    try:
        result = strategy.backtest(warmup=_WARMUP)
    except ValueError as e:
        logger.warning("%s %s 回测失败: %s", symbol, strategy_name, e)
        return None

    # 提取 equity 序列
    if hasattr(result, "equity") and result.equity is not None:
        eq = result.equity
        if isinstance(eq, pd.DataFrame) and "equity" in eq.columns:
            eq_sorted = eq.sort_values("date")
            returns = eq_sorted["equity"].pct_change().dropna()
            returns.name = strategy_name
            return returns
    if result.trades is not None and not result.trades.empty:
        # fallback: 从 trades 构造
        trades = result.trades
        if "exit_date" in trades.columns and "pnl" in trades.columns:
            daily = trades.groupby("exit_date")["pnl"].sum()
            daily = daily / initial_cash
            daily.name = strategy_name
            return daily

    return None


def _compute_rolling_sharpe_weights(
    returns_dict: Dict[str, pd.Series],
    window: int = _SHARPE_WINDOW,
    min_periods: int = _SHARPE_MIN_PERIODS,
) -> Dict[str, np.ndarray]:
    """按月度滚动窗口计算各策略 Sharpe 并归一化为权重。"""
    # 对齐日期
    dfs = []
    for name, sr in returns_dict.items():
        sr = sr.copy()
        sr.name = name
        dfs.append(sr)
    aligned = pd.concat(dfs, axis=1, join="inner").dropna()
    if len(aligned) < min_periods + window:
        return {}

    # 滚动 Sharpe
    rolling_sharpe = aligned.rolling(window, min_periods=min_periods).apply(
        lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 1e-10 else 0.0,
        raw=False,
    )
    rolling_sharpe = rolling_sharpe.fillna(0.0)

    # 归一化权重（softmax 风格）
    weights_dict: Dict[str, np.ndarray] = {}
    for col in rolling_sharpe.columns:
        s = rolling_sharpe[col].values
        # 取正部分
        s_pos = np.maximum(s, 0.0)
        weights_dict[col] = s_pos

    return weights_dict


def run_signal_fusion(
    symbols: List[str],
    strategies: Optional[Dict[str, Dict[str, Any]]] = None,
    weights: Optional[Dict[str, float]] = None,
    mode: str = "fixed",
    entry_threshold: float = 0.05,
    initial_cash: float = 1_000_000,
    full_start: str = "2020-01-01",
    test_start: str = "2023-01-01",
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    主入口：多策略信号融合。

    Args:
        symbols: 品种列表
        strategies: 策略名→参数 dict。默认 _DEFAULT_STRATEGIES
        weights: 固定权重。默认 _DEFAULT_WEIGHTS
        mode: "fixed" 或 "rolling_sharpe"
        entry_threshold: 入场阈值
        initial_cash: 初始资金
        full_start: 全量数据起始
        test_start: 回测起始
        output_dir: 输出目录

    Returns:
        汇总 DataFrame: 每行一个品种，含各策略指标 + 融合指标
    """
    strategies = strategies or _DEFAULT_STRATEGIES
    weights = weights or _DEFAULT_WEIGHTS

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    for symbol in symbols:
        logger.info(f"[信号融合] 品种: {symbol} 模式: {mode}")
        returns_dict: Dict[str, pd.Series] = {}
        metrics_dict: Dict[str, Dict[str, float]] = {}

        for sname, sparams in strategies.items():
            sr = _run_single_with_equity(
                symbol, sname, sparams,
                entry_threshold=entry_threshold,
                initial_cash=initial_cash,
                full_start=full_start,
                test_start=test_start,
            )
            if sr is not None and len(sr) > 20:
                returns_dict[sname] = sr
                metrics_dict[sname] = {
                    "total_return": float(sr.sum()),
                    "sharpe": float(sr.mean() / sr.std() * np.sqrt(252)) if sr.std() > 1e-10 else 0.0,
                    "vol": float(sr.std() * np.sqrt(252)),
                    "trades": len(sr),
                }

        if not returns_dict:
            logger.warning(f"  {symbol}: 无可用策略数据")
            continue

        # ── 融合 ──
        if mode == "rolling_sharpe" and len(returns_dict) > 1:
            # 滚动夏普权重
            W = _compute_rolling_sharpe_weights(returns_dict)
            if W and len(next(iter(W.values()))) > 10:
                # 对齐日期后计算加权组合收益
                aligned = pd.concat(
                    [sr.copy().rename(name) for name, sr in returns_dict.items()],
                    axis=1, join="inner",
                ).dropna()
                combined_rets = np.zeros(len(aligned))
                for i in range(len(aligned)):
                    total_w = 0.0
                    for name in aligned.columns:
                        w_i = W.get(name, np.zeros(len(aligned)))
                        if i < len(w_i) and w_i[i] > 0:
                            combined_rets[i] += aligned.iloc[i][name] * w_i[i]
                            total_w += w_i[i]
                    if total_w > 1e-10:
                        combined_rets[i] /= total_w
                combined_returns = pd.Series(combined_rets, index=aligned.index)
            else:
                logger.warning(f"  {symbol}: 滚动 Sharpe 数据不足，回退固定权重")
                combined_returns = _combine_fixed(returns_dict, weights)
        else:
            combined_returns = _combine_fixed(returns_dict, weights)

        # ── 融合指标 ──
        row: Dict[str, Any] = {"symbol": symbol, "mode": mode}
        for sname in strategies:
            m = metrics_dict.get(sname, {})
            row[f"{sname}_return"] = round(m.get("total_return", 0) * 100, 2)
            row[f"{sname}_sharpe"] = round(m.get("sharpe", 0), 2)

        if combined_returns is not None and len(combined_returns) > 5:
            row["fusion_return_pct"] = round(float(combined_returns.sum()) * 100, 2)
            row["fusion_sharpe"] = round(
                float(combined_returns.mean() / combined_returns.std() * np.sqrt(252)),
                2,
            ) if combined_returns.std() > 1e-10 else 0.0
            row["fusion_vol_pct"] = round(
                float(combined_returns.std() * np.sqrt(252) * 100), 2
            )
        else:
            row["fusion_return_pct"] = 0.0
            row["fusion_sharpe"] = 0.0
            row["fusion_vol_pct"] = 0.0

        all_rows.append(row)

        if output_dir:
            # 保存融合净值
            if combined_returns is not None and len(combined_returns) > 5:
                eq_df = pd.DataFrame({
                    "date": combined_returns.index,
                    "fusion_returns": combined_returns.values,
                    "fusion_equity": (1 + combined_returns).cumprod(),
                })
                eq_path = output_dir / f"fusion_{sanitize_filename(symbol)}.csv"
                eq_df.to_csv(eq_path, index=False)

    result_df = pd.DataFrame(all_rows)
    if output_dir:
        summary_path = output_dir / "signal_fusion_summary.csv"
        result_df.to_csv(summary_path, index=False)

    return result_df


def _combine_fixed(
    returns_dict: Dict[str, pd.Series],
    weights: Dict[str, float],
) -> pd.Series:
    """固定权重融合。"""
    aligned = pd.concat(
        [sr.copy().rename(name) for name, sr in returns_dict.items()],
        axis=1, join="inner",
    ).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)

    # 仅用有权重的策略
    cols = [c for c in aligned.columns if c in weights]
    if not cols:
        return pd.Series(dtype=float)

    w_arr = np.array([weights[c] for c in cols])
    w_arr = w_arr / w_arr.sum()
    combined = aligned[cols].values @ w_arr
    return pd.Series(combined, index=aligned.index)


__all__ = ["run_signal_fusion", "_DEFAULT_WEIGHTS", "_DEFAULT_STRATEGIES"]
