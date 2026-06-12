"""
实验 E4 / E5：组合层回测（风险平价 + 多品种分散）。

E4：对每个品种分别运行所有策略获取收益率，滚动 60 日计算风险平价权重，
    融合净值曲线后回测评估组合绩效。

E5：对每个品种独立回测，提取日收益率序列，按配置处理缺失值后等权合并
    为多品种组合，计算组合绩效与相关性矩阵。

委托 runner/backtest/runner.py 执行回测，runner/common/portfolio_utils.py 提供
权重计算与净值融合，runner/strategy/selector.py 提供策略列表。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd
from loguru import logger

from core.engine.pybroker_data_source import PyBrokerDataSource
from core.performance import PerformanceEvaluator
from runner.backtest.runner import get_pybroker_runner, safe_run_backtest
from runner.common.config_utils import get_missing_data_method
from runner.common.portfolio_utils import (
    calculate_risk_parity_weights,
    fuse_equities_by_weights,
)
from runner.common.utils import (
    format_metrics,
    handle_backtest_errors,
    is_valid_number,
    sanitize_filename,
    save_csv,
    save_equity_curve,
)
from runner.strategy.selector import get_strategy_names


# ============================================
# 类型定义
# ============================================


class PortfolioResult(TypedDict):
    """组合回测结果类型"""

    metrics: Dict[str, Any]
    equity: pd.DataFrame


# ============================================
# E4：风险平价融合
# ============================================


@handle_backtest_errors(return_value=pd.DataFrame())
def run_e4_risk_parity(
    data_source: PyBrokerDataSource, config: Dict[str, Any], output_dir: Path
) -> pd.DataFrame:
    """
    E4：风险平价融合实验。

    对每个品种，先单独运行所有策略，然后计算滚动波动率，
    使用风险平价权重融合信号，最后运行回测。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        汇总指标 DataFrame
    """
    logger.info("E4：风险平价融合实验")
    symbols: List[str] = config.get("symbols", [])
    strategy_names = get_strategy_names(config)
    bt_cfg = config["backtest"]
    all_results: List[Dict[str, Any]] = []

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        # 步骤1：单独运行所有策略获取收益率
        strategy_returns = {}
        strategy_equities = {}

        for sname in strategy_names:
            # 修复 per-symbol 隔离 bug：仅对当前品种做回测
            runner = get_pybroker_runner(
                data_source, config, strategies=[sname], target_symbols=[sym]
            )
            result = safe_run_backtest(
                runner,
                bt_cfg["full_start_date"],
                bt_cfg["full_end_date"],
                f"E4_single_{sym}_{sname}",
            )
            if result is not None and result.equity_curve is not None:
                eq = result.equity_curve.sort_values("date").copy()
                eq["date"] = pd.to_datetime(eq["date"])
                eq.set_index("date", inplace=True)
                rets = eq["equity"].pct_change().dropna()
                strategy_returns[sname] = rets
                strategy_equities[sname] = eq["equity"]

        if not strategy_returns:
            logger.warning(f"  {sym}: 无有效策略数据，跳过")
            continue

        # 步骤2：计算风险平价权重
        df_weights = calculate_risk_parity_weights(strategy_returns, window=60)
        save_csv(
            df_weights,
            output_dir / f"e4_weights_{sanitize_filename(sym.replace('.', '_'))}.csv",
        )
        logger.info(f"  {sym}: 已保存权重序列")

        # 步骤3：使用风险平价权重融合净值曲线
        avg_weights = df_weights.mean().to_dict()
        logger.info(f"  {sym}: 平均权重 {avg_weights}")

        # 合并所有策略的净值并使用风险平价权重融合
        df_equities = pd.DataFrame(strategy_equities)
        df_equities = df_equities.fillna(method="ffill").fillna(1.0)

        # P2 整改：调用公共工具 fuse_equities_by_weights 完成净值融合
        # 避免在 e4_e5_portfolio.py 内嵌约 30 行的手工循环
        # 内部自动归一化权重（sum=1.0）+ 跳过零/负净值的边界处理
        fused_equity = fuse_equities_by_weights(df_equities, avg_weights)

        # 计算融合后净值的绩效指标
        initial_capital = float(bt_cfg.get("initial_cash", 1_000_000))
        fused_equity_scaled = fused_equity * initial_capital
        result_df = pd.DataFrame(
            {"date": fused_equity_scaled.index, "equity": fused_equity_scaled.values}
        )

        # 计算绩效指标
        metrics = PerformanceEvaluator.compute_metrics(fused_equity_scaled)
        m = format_metrics(metrics)
        result_row: Dict[str, Any] = {
            "symbol": sym,
            "strategy": None,
            "experiment": "E4_风险平价",
            "error": None,
        }
        result_row.update(m)
        all_results.append(result_row)

        logger.info(
            f"  风险平价: return={m.get('total_return_pct', 'N/A')} "
            f"sharpe={m.get('sharpe', 'N/A')}"
        )

        # 保存融合后的净值曲线
        save_equity_curve(
            result_df.assign(symbol=sym),
            output_dir,
            f"e4_equity_{sanitize_filename(sym.replace('.', '_'))}",
        )

    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    save_csv(df, output_dir / "e4_risk_parity_metrics.csv")
    return df


# ============================================
# E5：多品种分散
# ============================================


@handle_backtest_errors(return_value=None)
def run_e5_multi_symbol(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> Optional[PortfolioResult]:
    """
    E5：多品种分散回测。

    对每个品种独立运行回测，提取日收益率序列，
    日期对齐后等权合并为组合，计算组合绩效指标。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        组合指标与净值，失败返回 None
    """
    logger.info("E5：多品种分散回测")
    symbols: List[str] = config.get("symbols", [])
    strategy_names = get_strategy_names(config)
    bt_cfg = config["backtest"]
    initial_cash = float(bt_cfg.get("initial_cash", 1_000_000))
    missing_method = get_missing_data_method(config)
    strategy_returns_by_symbol: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        # 修复 per-symbol 隔离 bug：仅对当前品种做回测
        runner = get_pybroker_runner(
            data_source, config, strategies=strategy_names, target_symbols=[sym]
        )
        result = safe_run_backtest(
            runner,
            bt_cfg["full_start_date"],
            bt_cfg["full_end_date"],
            f"E5_{sym}",
        )

        if result is None:
            logger.warning(f"  {sym}: 回测失败，跳过")
            continue

        eq = result.equity_curve
        if eq is None or eq.empty:
            logger.warning(f"  {sym}: 无净值数据，跳过")
            continue

        eq_sorted = eq.sort_values("date").copy()
        eq_sorted["date"] = pd.to_datetime(eq_sorted["date"])
        eq_sorted["daily_return"] = eq_sorted["equity"].pct_change()
        rets = eq_sorted[["date", "daily_return"]].dropna()
        rets = rets[rets["daily_return"].apply(is_valid_number)]
        strategy_returns_by_symbol[sym] = rets.set_index("date")

    if len(strategy_returns_by_symbol) == 0:
        logger.warning("E5：无有效品种收益数据，跳过多品种组合计算")
        return None

    logger.info("  计算多品种等权组合...")
    # 合并收益率
    combined_rets: Optional[pd.DataFrame] = None
    for sym, rets_df in strategy_returns_by_symbol.items():
        renamed = rets_df.rename(columns={"daily_return": sym})
        if combined_rets is None:
            combined_rets = renamed
        else:
            combined_rets = combined_rets.join(renamed, how="outer")

    if combined_rets is None or combined_rets.empty:
        logger.error("E5：合并收益率失败")
        return None

    # 根据配置处理缺失值
    if missing_method == "fill_zero":
        combined_rets = combined_rets.fillna(0.0)
    elif missing_method == "drop":
        combined_rets = combined_rets.dropna()
    elif missing_method == "ffill":
        combined_rets = combined_rets.fillna(method="ffill").fillna(0.0)
    else:
        combined_rets = combined_rets.fillna(0.0)

    portfolio_ret = combined_rets.mean(axis=1)
    portfolio_equity = (1.0 + portfolio_ret).cumprod() * initial_cash
    multi_eq = pd.DataFrame(
        {"date": portfolio_equity.index, "equity": portfolio_equity.values}
    )

    multi_metrics = PerformanceEvaluator.compute_metrics(portfolio_equity)
    m = format_metrics(multi_metrics)
    logger.info(
        f"  多品种组合: return={m.get('total_return_pct', 'N/A')} "
        f"sharpe={m.get('sharpe', 'N/A')} "
        f"max_dd={m.get('max_drawdown_pct', 'N/A')}"
    )

    save_equity_curve(multi_eq, output_dir, "e5_multi_symbol_equity")
    corr_matrix = combined_rets.corr()
    save_csv(corr_matrix, output_dir / "e5_correlation_matrix.csv")

    return {"metrics": m, "equity": multi_eq}
