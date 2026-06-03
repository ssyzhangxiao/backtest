"""
实验 E1-E5：基线回测、等权融合、动态加权、多品种分散。

每个实验保持独立函数，委托 runner/backtest/runner.py 执行回测，
委托 runner/common/utils.py 和 runner/strategy/selector.py 处理工具和策略。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
from loguru import logger

from core.engine.pybroker_data_source import PyBrokerDataSource
from core.performance import PerformanceEvaluator
from runner.backtest.runner import get_pybroker_runner, safe_run_backtest
from runner.common.utils import (
    safe_float,
    is_valid_number,
    save_csv,
    format_metrics,
    sanitize_filename,
    save_equity_curve,
    handle_backtest_errors,
)
from runner.common.config_utils import get_missing_data_method
from runner.strategy.selector import get_strategy_names


# ============================================
# 类型定义
# ============================================


class PortfolioResult(TypedDict):
    """组合回测结果类型"""

    metrics: Dict[str, Any]
    equity: pd.DataFrame


# ============================================
# 通用配置常量
# ============================================

_CONFIG_KEY_SYMBOLS = "symbols"
_CONFIG_KEY_BACKTEST = "backtest"
_CONFIG_KEY_INITIAL_CASH = "initial_cash"
_CONFIG_KEY_FULL_START = "full_start_date"
_CONFIG_KEY_FULL_END = "full_end_date"


# ============================================
# 通用组合实验函数（E2/E3）
# ============================================


def _run_equal_weight(
    data_source: PyBrokerDataSource, config: Dict[str, Any], output_dir: Path
) -> pd.DataFrame:
    """
    执行等权融合实验（E2）。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        汇总指标 DataFrame
    """
    logger.info("执行等权融合实验")
    symbols: List[str] = config.get(_CONFIG_KEY_SYMBOLS, [])
    strategy_names = get_strategy_names(config)
    bt_cfg = config[_CONFIG_KEY_BACKTEST]
    all_results: List[Dict[str, Any]] = []

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        runner = get_pybroker_runner(data_source, config, strategies=strategy_names)
        result = safe_run_backtest(
            runner,
            bt_cfg[_CONFIG_KEY_FULL_START],
            bt_cfg[_CONFIG_KEY_FULL_END],
            f"E2_{sym}",
            use_execute_fusion=False,
        )

        if result is None:
            all_results.append(
                {
                    "symbol": sym,
                    "strategy": None,
                    "experiment": "E2_等权融合",
                    "error": "回测失败",
                }
            )
            continue

        m = format_metrics(result.metrics)
        result_row: Dict[str, Any] = {
            "symbol": sym,
            "strategy": None,
            "experiment": "E2_等权融合",
            "error": None,
        }
        result_row.update(m)
        all_results.append(result_row)

        logger.info(
            f"  等权融合: return={m.get('total_return_pct', 'N/A')} "
            f"sharpe={m.get('sharpe', 'N/A')}"
        )

        eq = result.equity_curve
        if eq is not None and not eq.empty:
            save_equity_curve(
                eq.assign(symbol=sym),
                output_dir,
                f"e2_equity_{sanitize_filename(sym.replace('.', '_'))}",
            )

        if result.switch_log is not None and not result.switch_log.empty:
            save_csv(
                result.switch_log.assign(symbol=sym),
                output_dir
                / f"e2_switch_log_{sanitize_filename(sym.replace('.', '_'))}.csv",
            )
            logger.info(f"  {sym}: 已保存 {len(result.switch_log)} 条调仓决策记录")

    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    save_csv(df, output_dir / "e2_equal_weight_metrics.csv")
    return df


def _run_dynamic_weight(
    data_source: PyBrokerDataSource, config: Dict[str, Any], output_dir: Path
) -> pd.DataFrame:
    """
    执行动态加权融合实验（E3）。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        汇总指标 DataFrame
    """
    logger.info("执行动态加权融合实验")
    symbols: List[str] = config.get(_CONFIG_KEY_SYMBOLS, [])
    strategy_names = get_strategy_names(config)
    bt_cfg = config[_CONFIG_KEY_BACKTEST]
    all_results: List[Dict[str, Any]] = []

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        runner = get_pybroker_runner(data_source, config, strategies=strategy_names)
        result = safe_run_backtest(
            runner,
            bt_cfg[_CONFIG_KEY_FULL_START],
            bt_cfg[_CONFIG_KEY_FULL_END],
            f"E3_{sym}",
            use_execute_fusion=True,
        )

        if result is None:
            all_results.append(
                {
                    "symbol": sym,
                    "strategy": None,
                    "experiment": "E3_动态权重",
                    "error": "回测失败",
                }
            )
            continue

        m = format_metrics(result.metrics)
        result_row: Dict[str, Any] = {
            "symbol": sym,
            "strategy": None,
            "experiment": "E3_动态权重",
            "error": None,
        }
        result_row.update(m)
        all_results.append(result_row)

        logger.info(
            f"  动态权重: return={m.get('total_return_pct', 'N/A')} "
            f"sharpe={m.get('sharpe', 'N/A')}"
        )

        eq = result.equity_curve
        if eq is not None and not eq.empty:
            save_equity_curve(
                eq.assign(symbol=sym),
                output_dir,
                f"e3_equity_{sanitize_filename(sym.replace('.', '_'))}",
            )

        if result.switch_log is not None and not result.switch_log.empty:
            save_csv(
                result.switch_log.assign(symbol=sym),
                output_dir
                / f"e3_switch_log_{sanitize_filename(sym.replace('.', '_'))}.csv",
            )
            logger.info(f"  {sym}: 已保存 {len(result.switch_log)} 条调仓决策记录")

        if result.regime_history is not None and not result.regime_history.empty:
            save_csv(
                result.regime_history,
                output_dir
                / f"e3_regime_{sanitize_filename(sym.replace('.', '_'))}.csv",
            )

    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    save_csv(df, output_dir / "e3_dynamic_weight_metrics.csv")
    return df


# ============================================
# E4：风险平价融合
# ============================================


class WeightedSignalFusion:
    """加权信号融合类"""

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

    def combine(self, signals: Dict[str, float]) -> float:
        """
        融合信号。

        Args:
            signals: 策略信号字典

        Returns:
            融合后的信号
        """
        total = 0.0
        weight_sum = 0.0
        for name, signal in signals.items():
            weight = self.weights.get(name, 0.0)
            total += signal * weight
            weight_sum += weight
        return total / weight_sum if weight_sum > 0 else 0.0


def _calculate_rolling_volatility(returns: pd.Series, window: int = 60) -> pd.Series:
    """
    计算滚动波动率。

    Args:
        returns: 收益率序列
        window: 滚动窗口

    Returns:
        滚动波动率序列
    """
    return returns.rolling(window=window, min_periods=window // 2).std()


def _calculate_risk_parity_weights(
    strategy_returns: Dict[str, pd.Series], window: int = 60
) -> pd.DataFrame:
    """
    计算风险平价权重。

    Args:
        strategy_returns: 策略收益率字典
        window: 滚动窗口

    Returns:
        权重 DataFrame
    """
    # 合并所有策略收益率
    df_returns = pd.DataFrame(strategy_returns)
    df_returns = df_returns.fillna(0.0)

    # 计算滚动波动率
    df_vol = pd.DataFrame(index=df_returns.index)
    for name in df_returns.columns:
        df_vol[name] = _calculate_rolling_volatility(df_returns[name], window)

    # 填充初始波动率（使用 ffill 和 bfill 替代 deprecated 的 method）
    df_vol = df_vol.ffill().bfill()

    # 向量化计算风险平价权重（更高效，避免逐行赋值问题）
    inv_vol = 1.0 / (df_vol + 1e-10)
    sum_inv_vol = inv_vol.sum(axis=1)
    df_weights = inv_vol.div(sum_inv_vol, axis=0)

    return df_weights


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
    symbols: List[str] = config.get(_CONFIG_KEY_SYMBOLS, [])
    strategy_names = get_strategy_names(config)
    bt_cfg = config[_CONFIG_KEY_BACKTEST]
    all_results: List[Dict[str, Any]] = []

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        # 步骤1：单独运行所有策略获取收益率
        strategy_returns = {}
        strategy_equities = {}

        for sname in strategy_names:
            runner = get_pybroker_runner(data_source, config, strategies=[sname])
            result = safe_run_backtest(
                runner,
                bt_cfg[_CONFIG_KEY_FULL_START],
                bt_cfg[_CONFIG_KEY_FULL_END],
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
        df_weights = _calculate_risk_parity_weights(strategy_returns, window=60)
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

        # 归一化权重确保总和为1
        total_weight = sum(avg_weights.values())
        normalized_weights = (
            {k: v / total_weight for k, v in avg_weights.items()}
            if total_weight > 0
            else {}
        )

        # 计算融合后的净值
        fused_equity = pd.Series(1.0, index=df_equities.index)
        for date_idx in range(1, len(df_equities)):
            date = df_equities.index[date_idx]
            prev_date = df_equities.index[date_idx - 1]

            # 计算各策略的日收益率
            daily_returns = {}
            for sname in strategy_names:
                if (
                    sname in df_equities.columns
                    and df_equities.loc[prev_date, sname] > 0
                ):
                    daily_returns[sname] = (
                        df_equities.loc[date, sname] / df_equities.loc[prev_date, sname]
                    ) - 1.0

            # 使用平均权重计算融合收益率
            if daily_returns:
                fused_return = 0.0
                for sname, ret in daily_returns.items():
                    weight = normalized_weights.get(sname, 0.0)
                    fused_return += ret * weight
                fused_equity.loc[date] = fused_equity.loc[prev_date] * (
                    1.0 + fused_return
                )
            else:
                fused_equity.loc[date] = fused_equity.loc[prev_date]

        # 计算融合后净值的绩效指标
        initial_capital = float(bt_cfg.get(_CONFIG_KEY_INITIAL_CASH, 1_000_000))
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
# 实验函数
# ============================================


@handle_backtest_errors(return_value=pd.DataFrame())
def run_e1_single_strategy_baselines(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E1：单策略基线回测。

    对每个品种×每个策略单独运行回测，汇总绩效指标。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        汇总指标 DataFrame
    """
    logger.info("E1：单策略基线回测")
    symbols: List[str] = config.get(_CONFIG_KEY_SYMBOLS, [])
    strategy_names = get_strategy_names(config)
    bt_cfg = config[_CONFIG_KEY_BACKTEST]
    all_results: List[Dict[str, Any]] = []

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        for sname in strategy_names:
            runner = get_pybroker_runner(data_source, config, strategies=[sname])
            result = safe_run_backtest(
                runner,
                bt_cfg[_CONFIG_KEY_FULL_START],
                bt_cfg[_CONFIG_KEY_FULL_END],
                f"E1_{sym}_{sname}",
            )

            if result is not None:
                m = format_metrics(result.metrics)
                result_row: Dict[str, Any] = {
                    "symbol": sym,
                    "strategy": sname,
                    "experiment": "E1_单策略基线",
                    "error": None,
                }
                result_row.update(m)
                all_results.append(result_row)
                logger.info(
                    f"  {sname}: return={m.get('total_return_pct', 'N/A')} "
                    f"sharpe={m.get('sharpe', 'N/A')} "
                    f"max_dd={m.get('max_drawdown_pct', 'N/A')}"
                )

                eq = result.equity_curve
                if eq is not None and not eq.empty:
                    save_equity_curve(
                        eq.assign(symbol=sym, strategy=sname),
                        output_dir,
                        f"e1_equity_{sanitize_filename(sym.replace('.', '_'))}_{sname}",
                    )
            else:
                all_results.append(
                    {
                        "symbol": sym,
                        "strategy": sname,
                        "experiment": "E1_单策略基线",
                        "error": "回测失败",
                    }
                )

    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    save_csv(df, output_dir / "e1_baseline_metrics.csv")
    return df


@handle_backtest_errors(return_value=pd.DataFrame())
def run_e2_equal_weight(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E2：等权信号融合回测。

    所有策略信号等权融合，单一品种组合。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        汇总指标 DataFrame
    """
    logger.info("E2：等权信号融合回测")
    return _run_equal_weight(data_source, config, output_dir)


@handle_backtest_errors(return_value=pd.DataFrame())
def run_e3_dynamic_weight(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E3：环境动态加权回测（execute 融合模式）。

    根据市场环境动态分配各策略权重。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        汇总指标 DataFrame
    """
    logger.info("E3：环境动态加权回测（execute 融合模式）")
    return _run_dynamic_weight(data_source, config, output_dir)


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
    symbols: List[str] = config.get(_CONFIG_KEY_SYMBOLS, [])
    strategy_names = get_strategy_names(config)
    bt_cfg = config[_CONFIG_KEY_BACKTEST]
    initial_cash = float(bt_cfg.get(_CONFIG_KEY_INITIAL_CASH, 1_000_000))
    missing_method = get_missing_data_method(config)
    strategy_returns_by_symbol: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        runner = get_pybroker_runner(data_source, config, strategies=strategy_names)
        result = safe_run_backtest(
            runner,
            bt_cfg[_CONFIG_KEY_FULL_START],
            bt_cfg[_CONFIG_KEY_FULL_END],
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
