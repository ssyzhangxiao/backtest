"""
实验 E6 / E7：样本内/外验证（WalkForward 滚动 + 样本外 OOS）。

E6：对每个策略执行滚动窗口回测，评估参数稳定性；
    从配置读取 walkforward.window 和 walkforward.step。

E7：将数据分为样本内（in_sample）和样本外（out_sample）两段，
    分别回测并比较 Sharpe 衰减率（30% 阈值）。
    若未配置 in_sample_end_date / out_sample_start_date，
    按 7:3 比例用工作日（BDay）自动划分。

委托 runner/backtest/runner.py 执行回测，runner/common/config_utils.py 提供
walkforward 配置读取。
"""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from core.engine.pybroker_data_source import PyBrokerDataSource
from runner.backtest.runner import get_pybroker_runner, safe_run_backtest
from runner.common.config_utils import get_walkforward_config
from runner.common.utils import (
    format_metrics,
    handle_backtest_errors,
    safe_float,
    save_csv,
    save_equity_curve,
)
from runner.strategy.selector import get_strategy_names


# ============================================
# 模块常量
# ============================================


_EPSILON = 1e-10
_SAFE_DECAY_THRESHOLD = 0.3


# ============================================
# E6：WalkForward 滚动验证
# ============================================


@handle_backtest_errors(return_value=pd.DataFrame())
def run_e6_walkforward(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E6：WalkForward 滚动验证。

    对每个策略执行滚动窗口回测，评估参数稳定性。
    从配置读取 walkforward.window 和 walkforward.step。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        各窗口汇总指标 DataFrame
    """
    logger.info("E6：WalkForward 滚动验证")
    bt_cfg = config["backtest"]
    wf_cfg = get_walkforward_config(config)
    strategy_names = get_strategy_names(config)
    symbols: List[str] = config.get("symbols", [])

    all_wf_metrics: List[Dict[str, Any]] = []
    for sname in strategy_names:
        try:
            # 修复 per-symbol 隔离 bug：传 target_symbols=全部品种（walkforward 内部不分品种）
            runner = get_pybroker_runner(
                data_source, config, strategies=[sname], target_symbols=symbols
            )
            wf_result = runner.walkforward(
                start_date=bt_cfg["full_start_date"],
                end_date=bt_cfg["full_end_date"],
            )
            # 修复 2026-06-10：兼容 _WindowRunner._compute_simple_metrics
            # 返回的 metrics 字段（既有 total_return 也有 total_return_pct，
            # 但 DataFrame 构造时嵌套 dict 会触发 KeyError）。
            # 显式平铺 metrics 到外层 dict，避免 pandas 内部 dict 转换失败。
            for w in wf_result.windows:
                flat_w = dict(w)  # 浅拷贝
                metrics_nested = flat_w.pop("metrics", {}) or {}
                for mk, mv in metrics_nested.items():
                    flat_w[f"metric_{mk}"] = mv
                flat_w["strategy"] = sname
                all_wf_metrics.append(flat_w)
            logger.info(
                f"  {sname}: {len(wf_result.windows)} 窗口, "
                f"avg_sharpe={wf_result.overall_metrics.get('sharpe', 'N/A')}"
            )
        except Exception as e:
            logger.error(f"  {sname} WalkForward 失败: {e}")
            logger.exception(f"  {sname} WalkForward 异常详情")

    df = pd.DataFrame(all_wf_metrics) if all_wf_metrics else pd.DataFrame()
    if not df.empty:
        save_csv(df, output_dir / "e6_walkforward_metrics.csv")
    return df


# ============================================
# E7：样本外验证
# ============================================


@handle_backtest_errors(return_value=pd.DataFrame())
def run_e7_out_of_sample(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E7：样本外验证。

    将数据分为样本内和样本外两段，分别回测并比较 Sharpe 衰减率。
    强制要求 in_sample_end_date 和 out_sample_start_date，
    如果未配置则自动按 7:3 比例划分。

    Args:
        data_source: 数据源
        config: 配置字典
        output_dir: 输出目录

    Returns:
        汇总指标 DataFrame
    """
    logger.info("E7：样本外验证")
    symbols: List[str] = config.get("symbols", [])
    strategy_names = get_strategy_names(config)
    bt_cfg = config["backtest"]

    # 确定样本划分日期
    full_start = pd.to_datetime(bt_cfg["full_start_date"])
    full_end = pd.to_datetime(bt_cfg["full_end_date"])

    if "in_sample_end_date" in bt_cfg and bt_cfg["in_sample_end_date"]:
        in_sample_end = pd.to_datetime(bt_cfg["in_sample_end_date"])
    else:
        # P1 整改：使用交易日计数（pd.date_range + len(df['date'].unique())）
        # 比日历天数更准确：避免节假日差异导致样本划分偏移
        # 由于此阶段数据源尚未加载，我们以"工作日频率"近似估计交易日数
        # 实际确切的交易日计数由回测侧 date_range 提供；此处用作粗估
        trading_days_total = len(
            pd.date_range(full_start, full_end, freq="B")  # B = 工作日
        )
        split_day_idx = int(trading_days_total * 0.7)
        # 将 idx 反推为日期：取 start + 70% 个工作日
        in_sample_end = full_start + pd.tseries.offsets.BDay(split_day_idx)
        logger.info(
            f"  自动划分样本内结束日期: {in_sample_end.date()} "
            f"（{trading_days_total} 个工作日中前 {split_day_idx} 天）"
        )

    if "out_sample_start_date" in bt_cfg and bt_cfg["out_sample_start_date"]:
        out_sample_start = pd.to_datetime(bt_cfg["out_sample_start_date"])
    else:
        # 从样本内结束日开始
        out_sample_start = in_sample_end
        logger.info(f"  自动设置样本外开始日期: {out_sample_start.date()}")

    all_results: List[Dict[str, Any]] = []
    primary_symbol = symbols[0] if symbols else None

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        try:
            # 样本内回测（修复 per-symbol 隔离 bug：仅对当前品种做回测）
            runner_in = get_pybroker_runner(
                data_source, config, strategies=strategy_names, target_symbols=[sym]
            )
            result_in = safe_run_backtest(
                runner_in,
                str(full_start.date()),
                str(in_sample_end.date()),
                f"E7_in_{sym}",
            )
            if result_in is not None:
                m_in = format_metrics(result_in.metrics)
                m_in["symbol"] = sym
                m_in["split"] = "in_sample"
                all_results.append(m_in)
                if sym == primary_symbol:
                    eq_in = result_in.equity_curve
                    if eq_in is not None and not eq_in.empty:
                        save_equity_curve(eq_in, output_dir, "e7_equity_in_sample")

            # 样本外回测（修复 per-symbol 隔离 bug：仅对当前品种做回测）
            runner_out = get_pybroker_runner(
                data_source, config, strategies=strategy_names, target_symbols=[sym]
            )
            result_out = safe_run_backtest(
                runner_out,
                str(out_sample_start.date()),
                str(full_end.date()),
                f"E7_out_{sym}",
            )
            if result_out is not None:
                m_out = format_metrics(result_out.metrics)
                m_out["symbol"] = sym
                m_out["split"] = "out_sample"
                all_results.append(m_out)
                if sym == primary_symbol:
                    eq_out = result_out.equity_curve
                    if eq_out is not None and not eq_out.empty:
                        save_equity_curve(eq_out, output_dir, "e7_equity_out_sample")

            # Sharpe 衰减率
            if result_in is not None and result_out is not None:
                sharpe_in = safe_float(result_in.metrics.get("sharpe", 0))
                sharpe_out = safe_float(result_out.metrics.get("sharpe", 0))
                if abs(sharpe_in) > _EPSILON:
                    decay = (sharpe_in - sharpe_out) / abs(sharpe_in)
                    is_qualified = decay < _SAFE_DECAY_THRESHOLD
                    logger.info(
                        f"  Sharpe衰减率: {decay:.1%} {'合格' if is_qualified else '不合格'}"
                    )
        except Exception as e:
            logger.error(f"  {sym}: 失败 - {e}")

    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    save_csv(df, output_dir / "e7_out_of_sample_metrics.csv")
    return df
