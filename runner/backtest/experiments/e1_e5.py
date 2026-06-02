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
)
from runner.strategy.selector import get_strategy_names


# ============================================
# Type Definitions (P3 - Type Annotation Standardization)
# ============================================

class ExperimentResult(TypedDict):
    """标准化实验结果记录类型。"""
    symbol: Optional[str]
    strategy: Optional[str]
    experiment: Optional[str]
    error: Optional[str]


class PortfolioResult(TypedDict):
    """组合回测结果类型。"""
    metrics: Dict[str, Any]
    equity: pd.DataFrame


# ============================================
# Common Configuration Constants (P2 - Hardcoded Keys)
# ============================================

_CONFIG_KEY_SYMBOLS = "symbols"
_CONFIG_KEY_BACKTEST = "backtest"
_CONFIG_KEY_INITIAL_CASH = "initial_cash"
_CONFIG_KEY_FULL_START = "full_start_date"
_CONFIG_KEY_FULL_END = "full_end_date"


# ============================================
# Common Utility Functions (P1 - Code Duplication)
# ============================================

def _run_portfolio_experiment(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
    experiment_name: str,
    experiment_label: str,
    use_execute_fusion: bool = False,
) -> pd.DataFrame:
    """
    通用组合实验执行函数（E2/E3 复用）。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录
        experiment_name: 实验名称（如 "E2"）
        experiment_label: 实验标签（如 "E2_等权融合"）
        use_execute_fusion: 是否使用 execute 融合模式

    Returns:
        汇总指标 DataFrame
    """
    symbols: List[str] = config.get(_CONFIG_KEY_SYMBOLS, [])
    strategy_names = get_strategy_names(config)
    bt_cfg = config[_CONFIG_KEY_BACKTEST]
    all_results: List[ExperimentResult] = []

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        try:
            runner = get_pybroker_runner(data_source, config, strategies=strategy_names)
            result = safe_run_backtest(
                runner,
                bt_cfg[_CONFIG_KEY_FULL_START],
                bt_cfg[_CONFIG_KEY_FULL_END],
                f"{experiment_name}_{sym}",
                use_execute_fusion=use_execute_fusion,
            )

            if result is None:
                all_results.append({
                    "symbol": sym,
                    "strategy": None,
                    "experiment": experiment_label,
                    "error": "回测失败",
                })
                continue

            m = format_metrics(result.metrics)
            result_row: ExperimentResult = {
                "symbol": sym,
                "strategy": None,
                "experiment": experiment_label,
                "error": None,
            }
            result_row.update(m)  # type: ignore
            all_results.append(result_row)

            logger.info(
                f"  portfolio: return={m.get('total_return_pct', 'N/A')} "
                f"sharpe={m.get('sharpe', 'N/A')}"
            )

            eq = result.equity_curve
            if eq is not None and not eq.empty:
                save_csv(
                    eq.assign(symbol=sym),
                    output_dir / f"{experiment_name.lower()}_equity_{sanitize_filename(sym.replace('.', '_'))}.csv",
                )

            if result.switch_log is not None and not result.switch_log.empty:
                save_csv(
                    result.switch_log.assign(symbol=sym),
                    output_dir / f"{experiment_name.lower()}_switch_log_{sanitize_filename(sym.replace('.', '_'))}.csv",
                )
                logger.info(f"  {sym}: 已保存 {len(result.switch_log)} 条调仓决策记录")

            if use_execute_fusion and result.regime_history is not None and not result.regime_history.empty:
                save_csv(
                    result.regime_history,
                    output_dir / f"{experiment_name.lower()}_regime_{sanitize_filename(sym.replace('.', '_'))}.csv",
                )

        except KeyboardInterrupt:
            logger.warning("用户中断执行")
            raise
        except SystemExit:
            logger.warning("系统退出")
            raise
        except Exception as e:
            logger.error(f"  {sym}: 失败 - {e}")
            all_results.append({
                "symbol": sym,
                "strategy": None,
                "experiment": experiment_label,
                "error": str(e),
            })

    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    save_csv(
        df,
        output_dir / f"{experiment_name.lower()}_{'dynamic_weight' if use_execute_fusion else 'equal_weight'}_metrics.csv",
    )
    return df


# ============================================
# Experiment Functions
# ============================================

def run_e1_single_strategy_baselines(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E1: 单策略基线回测。
    对每个品种×每个策略单独运行回测，汇总绩效指标。

    Returns:
        汇总指标 DataFrame
    """
    logger.info("E1: 单策略基线回测")
    symbols: List[str] = config.get(_CONFIG_KEY_SYMBOLS, [])
    strategy_names = get_strategy_names(config)
    bt_cfg = config[_CONFIG_KEY_BACKTEST]
    all_results: List[ExperimentResult] = []

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        for sname in strategy_names:
            try:
                runner = get_pybroker_runner(data_source, config, strategies=[sname])
                result = safe_run_backtest(
                    runner,
                    bt_cfg[_CONFIG_KEY_FULL_START],
                    bt_cfg[_CONFIG_KEY_FULL_END],
                    f"E1_{sym}_{sname}",
                )

                if result is not None:
                    m = format_metrics(result.metrics)
                    result_row: ExperimentResult = {
                        "symbol": sym,
                        "strategy": sname,
                        "experiment": "E1_单策略基线",
                        "error": None,
                    }
                    result_row.update(m)  # type: ignore
                    all_results.append(result_row)
                    logger.info(
                        f"  {sname}: return={m.get('total_return_pct', 'N/A')} "
                        f"sharpe={m.get('sharpe', 'N/A')} "
                        f"max_dd={m.get('max_drawdown_pct', 'N/A')}"
                    )
                else:
                    all_results.append({
                        "symbol": sym,
                        "strategy": sname,
                        "experiment": "E1_单策略基线",
                        "error": "回测失败",
                    })

            except KeyboardInterrupt:
                logger.warning("用户中断执行")
                raise
            except SystemExit:
                logger.warning("系统退出")
                raise
            except Exception as e:
                logger.error(f"  {sym}/{sname}: 失败 - {e}")
                all_results.append({
                    "symbol": sym,
                    "strategy": sname,
                    "experiment": "E1_单策略基线",
                    "error": str(e),
                })

    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    save_csv(df, output_dir / "e1_baseline_metrics.csv")
    return df


def run_e2_equal_weight(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E2: 等权信号融合回测。
    所有策略信号等权融合，单一品种组合。

    Returns:
        汇总指标 DataFrame
    """
    logger.info("E2: 等权信号融合回测")
    return _run_portfolio_experiment(
        data_source, config, output_dir, "E2", "E2_等权融合", use_execute_fusion=False,
    )


def run_e3_dynamic_weight(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E3: 环境动态加权回测（execute 融合模式）。
    根据市场环境动态分配各策略权重。

    Returns:
        汇总指标 DataFrame
    """
    logger.info("E3: 环境动态加权回测（execute 融合模式）")
    return _run_portfolio_experiment(
        data_source, config, output_dir, "E3", "E3_动态权重", use_execute_fusion=True,
    )


def run_e4_placeholder(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    E4: [占位实现] 待定义实验。

    预留位置用于未来实验扩展，如：
    - 因子增强策略
    - 自适应参数优化
    - 多时间框架融合

    Returns:
        占位返回值，当前返回 None
    """
    logger.info("E4: [占位] 待定义实验 - 跳过")
    # TODO: 实现 E4 实验逻辑
    return None


def run_e5_multi_symbol(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> Optional[PortfolioResult]:
    """
    E5: 多品种分散回测。

    对每个品种独立运行回测，提取日收益率序列，
    日期对齐后等权合并为组合，计算组合绩效指标。

    Returns:
        组合指标与净值，失败返回 None
    """
    logger.info("E5: 多品种分散回测")
    symbols: List[str] = config.get(_CONFIG_KEY_SYMBOLS, [])
    strategy_names = get_strategy_names(config)
    bt_cfg = config[_CONFIG_KEY_BACKTEST]
    initial_cash = float(bt_cfg.get(_CONFIG_KEY_INITIAL_CASH, 1_000_000))
    strategy_returns_by_symbol: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        try:
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

        except KeyboardInterrupt:
            logger.warning("用户中断执行")
            raise
        except SystemExit:
            logger.warning("系统退出")
            raise
        except Exception as e:
            logger.error(f"  {sym}: 失败 - {e}")

    if len(strategy_returns_by_symbol) == 0:
        logger.warning("E5: 无有效品种收益数据，跳过多品种组合计算")
        return None

    if len(strategy_returns_by_symbol) == 1:
        sym, rets_df = next(iter(strategy_returns_by_symbol.items()))
        single_ret = rets_df.reset_index()
        single_ret = single_ret[single_ret["daily_return"].apply(is_valid_number)]
        if single_ret.empty:
            return None
        portfolio_equity = (1.0 + single_ret["daily_return"]).cumprod() * initial_cash
        multi_eq = pd.DataFrame({"date": single_ret["date"], "equity": portfolio_equity.values})
        multi_metrics = PerformanceEvaluator.compute_metrics(portfolio_equity)
        m = format_metrics(multi_metrics)
        logger.info(f"  单品种({sym})组合: return={m.get('total_return_pct','N/A')} sharpe={m.get('sharpe','N/A')}")
        save_csv(multi_eq, output_dir / "e5_multi_symbol_equity.csv")
        return {"metrics": m, "equity": multi_eq}

    logger.info("  计算多品种等权组合...")
    try:
        combined_rets: Optional[pd.DataFrame] = None
        for sym, rets_df in strategy_returns_by_symbol.items():
            renamed = rets_df.rename(columns={"daily_return": sym})
            if combined_rets is None:
                combined_rets = renamed
            else:
                combined_rets = combined_rets.join(renamed, how="outer")

        if combined_rets is None or combined_rets.empty:
            logger.error("E5: 合并收益率失败")
            return None

        combined_rets = combined_rets.fillna(0.0)
        portfolio_ret = combined_rets.mean(axis=1)
        portfolio_equity = (1.0 + portfolio_ret).cumprod() * initial_cash
        multi_eq = pd.DataFrame({"date": portfolio_equity.index, "equity": portfolio_equity.values})

        multi_metrics = PerformanceEvaluator.compute_metrics(portfolio_equity)
        m = format_metrics(multi_metrics)
        logger.info(
            f"  多品种组合: return={m.get('total_return_pct', 'N/A')} "
            f"sharpe={m.get('sharpe', 'N/A')} "
            f"max_dd={m.get('max_drawdown_pct', 'N/A')}"
        )

        save_csv(multi_eq, output_dir / "e5_multi_symbol_equity.csv")
        corr_matrix = combined_rets.corr()
        save_csv(corr_matrix, output_dir / "e5_correlation_matrix.csv")

        return {"metrics": m, "equity": multi_eq}

    except KeyboardInterrupt:
        logger.warning("用户中断执行")
        raise
    except SystemExit:
        logger.warning("系统退出")
        raise
    except Exception as e:
        logger.error(f"E5 多品种组合计算失败: {e}")
        return None
