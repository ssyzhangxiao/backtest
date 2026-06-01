#!/usr/bin/env python3
"""
多策略量化回测系统 — 完整回测执行脚本（整合版）

整合自 run_full_backtest.py 与 run_pybroker_full_backtest_v2.py，
保留两者核心业务功能，消除代码冗余，统一使用系统标准模块。

所有功能实现严格调用现有系统模块：
  1. 数据加载：core.engine.pybroker_data_source.create_hybrid_data_source
2. 回测引擎：core.engine.backtest_runner.PyBrokerBacktestRunner
  3. 市场环境：core.market_regime.MarketRegimeDetector
  4. 策略库：core.strategy_registry.StrategyLibrary
  5. 绩效指标：utils.metrics.MetricsCalculator
  6. 因子打分：core.engine.switch_engine.FactorScoringEngine
  7. 配置管理：config.yaml

实验阶段：
  E1: 单策略基线回测
  E2: 等权信号融合
  E3: 环境动态加权
  E5: 多品种分散
  E6: WalkForward 滚动验证
  E7: 样本外验证
  E8: Bootstrap 置信区间
  E9: 蒙特卡洛模拟
  E10: HTML 报告生成
  E11: 滚动IC加权与因子衰减分析

注：E4（策略切换）已废弃，系统已切换为因子打分调仓模式。
"""

import os
import sys
import yaml
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")

from loguru import logger

from core.engine.backtest_runner import (
    PyBrokerBacktestRunner,
    PyBrokerResult,
)
from core.engine.pybroker_data_source import (
    PyBrokerDataSource,
    create_hybrid_data_source,
)
from core.engine.rolling_ic import RollingICWeightEngine, RollingICConfig
from core.engine.factor_decay import (
    FactorDecayMonitor, FactorDecayConfig, DecayStatus, DecayAlert,
)
from core.config import BacktestConfig, DEFAULT_FACTOR_WEIGHTS
from core.performance import PerformanceEvaluator
from utils.metrics import MetricsCalculator

# ══════════════════════════════════════════════════════════════════════════════
# 全局常量
# ══════════════════════════════════════════════════════════════════════════════

_EPSILON = 1e-10
_DEFAULT_CHART_DPI = 150
_DEFAULT_FIGSIZE_WIDE = (14, 6)
_DEFAULT_FIGSIZE_FULL = (12, 8)
_SAFE_DECAY_THRESHOLD = 0.3  # Sharpe衰减率合格阈值


def _safe_float(val: Any) -> float:
    """安全转float，异常时返回0.0。"""
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _is_valid_number(val: Any) -> bool:
    """检查值是否为有效的有限数值。"""
    try:
        v = float(val)
        return not (np.isnan(v) or np.isinf(v))
    except (ValueError, TypeError):
        return False


def _safe_div(a: float, b: float) -> float:
    """安全除法，除零返回0.0。"""
    return a / b if abs(b) > _EPSILON else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数：配置与数据
# ══════════════════════════════════════════════════════════════════════════════


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载YAML配置文件。

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config


def get_tqsdk_credentials() -> Tuple[Optional[str], Optional[str]]:
    """获取天勤SDK凭证，优先环境变量，回退config.yaml。"""
    phone: Optional[str] = os.getenv("TQSDK_PHONE")
    password: Optional[str] = os.getenv("TQSDK_PASSWORD")
    if not phone or not password:
        try:
            cfg = load_config()
            data_cfg = cfg.get("data", {})
            phone = phone or data_cfg.get("tqsdk_phone")
            password = password or data_cfg.get("tqsdk_password")
        except Exception:
            pass
    if not phone or not password:
        logger.warning("TqSdk凭证未设置，将仅使用CSV数据")
    return phone, password


def format_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    """
    格式化绩效指标：四舍五入float，NaN/Inf改为'N/A'。

    Args:
        m: 原始指标字典

    Returns:
        格式化后的指标字典
    """
    result: Dict[str, Any] = {}
    for k, v in m.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            result[k] = "N/A"
        elif isinstance(v, float):
            result[k] = round(v, 4)
        else:
            result[k] = v
    return result


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """安全保存DataFrame到CSV。"""
    try:
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            logger.debug(f"跳过空数据保存: {path}")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"已保存: {path}")
    except Exception as e:
        logger.error(f"保存CSV失败 {path}: {e}")


def _get_strategy_names(config: Dict[str, Any]) -> List[str]:
    """从配置中提取策略名称列表。"""
    strategies = config.get("strategies", [])
    return [s["name"] for s in strategies if isinstance(s, dict) and "name" in s]


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数：回测运行器
# ══════════════════════════════════════════════════════════════════════════════

# 回测配置缓存，避免重复构建
_bt_config_cache: Dict[str, BacktestConfig] = {}


def _build_backtest_config(config: Dict[str, Any]) -> BacktestConfig:
    """
    构建BacktestConfig对象（带缓存）。

    Args:
        config: 完整配置字典

    Returns:
        BacktestConfig实例
    """
    cache_key = f"{id(config)}"
    if cache_key in _bt_config_cache:
        return _bt_config_cache[cache_key]

    bt_cfg = config["backtest"]
    risk_cfg: Dict[str, Any] = config.get("risk_management", {})
    bt_config = BacktestConfig(
        initial_cash=bt_cfg.get("initial_cash", 1_000_000),
        commission_rate=bt_cfg.get("commission_rate", bt_cfg.get("commission", 0.0001)),
        slippage_rate=bt_cfg.get("slippage_rate", bt_cfg.get("slippage", 0.0001)),
        stop_loss_pct=risk_cfg.get("stop_loss_pct", bt_cfg.get("stop_loss_pct", 0.03)),
        max_position_pct=risk_cfg.get("position_limit_pct", bt_cfg.get("max_position_pct", 0.15)),
        max_total_position_pct=risk_cfg.get("total_position_limit", bt_cfg.get("max_total_position_pct", 0.6)),
        in_sample_end=bt_cfg.get("in_sample_end_date"),
        factor_weights=config.get("factor_weights", {}),
        entry_threshold=float(bt_cfg.get("entry_threshold", 0.05)),
        min_position_pct=float(bt_cfg.get("min_position_pct", 0.0)),
        use_cross_section=bool(bt_cfg.get("use_cross_section", True)),
        use_rank_score=bool(bt_cfg.get("use_rank_score", True)),
        use_rolling_ic=bool(bt_cfg.get("use_rolling_ic", True)),
        top_n_symbols=int(bt_cfg.get("top_n_symbols", 5)),
    )
    bt_config.rebalance_days = bt_cfg.get("rebalance_freq", 3)
    _bt_config_cache[cache_key] = bt_config
    return bt_config


def _extract_optimized_weights(e11_result: Dict[str, Any]) -> Dict[str, float]:
    """
    从E11因子分析结果中提取优化后的权重。

    取所有品种权重的均值作为全局优化权重。
    """
    if not e11_result:
        return {}

    all_weights: Dict[str, List[float]] = {}
    for symbol, data in e11_result.items():
        if isinstance(data, dict) and "final_weights" in data:
            for factor, weight in data["final_weights"].items():
                if factor not in all_weights:
                    all_weights[factor] = []
                all_weights[factor].append(float(weight))

    if not all_weights:
        return {}

    optimized = {}
    for factor, weights in all_weights.items():
        optimized[factor] = round(sum(weights) / len(weights), 4)

    total = sum(optimized.values())
    if total > 0:
        optimized = {k: round(v / total, 4) for k, v in optimized.items()}

    return optimized


def _apply_ic_weights_to_config(
    config: Dict[str, Any],
    ic_weights: Dict[str, float],
) -> Dict[str, Any]:
    """将IC优化权重应用到配置字典。"""
    import copy
    enhanced = copy.deepcopy(config)
    enhanced["factor_weights"] = ic_weights
    return enhanced


def get_pybroker_runner(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    strategies: Optional[List[str]] = None,
) -> PyBrokerBacktestRunner:
    """
    创建PyBroker回测运行器。

    Args:
        data_source: 数据源
        config: 配置字典
        strategies: 策略名称列表

    Returns:
        PyBrokerBacktestRunner实例
    """
    bt_config = _build_backtest_config(config)
    symbols = config.get("symbols", [])
    runner = PyBrokerBacktestRunner(data_source, bt_config, target_symbols=symbols)
    if strategies:
        runner.register_strategies(strategies)
    return runner


def _safe_run_backtest(
    runner: PyBrokerBacktestRunner,
    start_date: str,
    end_date: str,
    experiment_name: str,
    **kwargs,
) -> Optional[PyBrokerResult]:
    """
    安全执行回测，捕获异常并记录日志。

    Args:
        runner: 回测运行器
        start_date: 开始日期
        end_date: 结束日期
        experiment_name: 实验名称（用于日志）
        **kwargs: 传给 runner.run 的额外参数

    Returns:
        回测结果，失败返回None
    """
    try:
        return runner.run(start_date=start_date, end_date=end_date, **kwargs)
    except Exception as e:
        logger.error(f"{experiment_name} 回测执行失败: {e}", exc_info=True)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数：绘图
# ══════════════════════════════════════════════════════════════════════════════


def _plot_equity_curve(
    eq: pd.DataFrame,
    title: str,
    label: str,
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
) -> None:
    """
    绘制净值曲线与回撤双图。

    Args:
        eq: 净值DataFrame，需包含date和equity列
        title: 图表标题
        label: 曲线图例标签
        path: 保存路径
        dpi: 图片DPI
    """
    if eq is None or eq.empty:
        logger.debug(f"跳过空净值曲线绘图: {title}")
        return

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=_DEFAULT_FIGSIZE_FULL,
            sharex=True, gridspec_kw={"height_ratios": [3, 1]},
        )
        dates = pd.to_datetime(eq["date"])
        equity = eq["equity"].values

        ax1.plot(dates, equity, linewidth=1, label=label)
        ax1.set_title(f"{title} — 净值曲线", fontsize=14)
        ax1.set_ylabel("净值")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / np.where(peak > 0, peak, 1.0) * 100
        ax2.fill_between(dates, 0, dd, color="red", alpha=0.3)
        ax2.plot(dates, dd, color="red", linewidth=0.8)
        ax2.set_ylabel("回撤 %")
        ax2.set_xlabel("日期")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"净值曲线已保存: {path}")
    except Exception as e:
        logger.error(f"净值曲线绘图失败 {title}: {e}")
        plt.close("all")


def _plot_monte_carlo(
    median: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
) -> None:
    """
    绘制蒙特卡洛模拟净值分布图。

    Args:
        median: 中位数序列
        lower: 5%分位序列
        upper: 95%分位序列
        path: 保存路径
        dpi: 图片DPI
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE_WIDE)
        days = np.arange(len(median))
        ax.fill_between(days, lower, upper, alpha=0.3, color="blue", label="90% CI")
        ax.plot(days, median, color="blue", linewidth=1.5, label="Median")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="初始值")
        ax.set_title("蒙特卡洛模拟 — 净值曲线分布 (1000次)", fontsize=14)
        ax.set_xlabel("交易日")
        ax.set_ylabel("净值")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"蒙特卡洛图表已保存: {path}")
    except Exception as e:
        logger.error(f"蒙特卡洛绘图失败: {e}")
        plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
# E1: 单策略基线回测
# ══════════════════════════════════════════════════════════════════════════════


def run_e1_single_strategy_baselines(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E1: 单策略基线回测。
    对每个品种×每个策略单独运行回测，汇总绩效指标。

    Returns:
        汇总指标DataFrame
    """
    logger.info("E1: 单策略基线回测")
    symbols: List[str] = config.get("symbols", [])
    strategy_names = _get_strategy_names(config)
    bt_cfg = config["backtest"]

    all_results: List[Dict[str, Any]] = []
    for sym in symbols:
        logger.info(f"  品种: {sym}")
        for sname in strategy_names:
            try:
                runner = get_pybroker_runner(data_source, config, strategies=[sname])
                result = _safe_run_backtest(
                    runner, bt_cfg["full_start_date"], bt_cfg["full_end_date"],
                    f"E1_{sym}_{sname}",
                )
                if result is not None:
                    m = format_metrics(result.metrics)
                    m["symbol"] = sym
                    m["strategy"] = sname
                    all_results.append(m)
                    logger.info(
                        f"  {sname}: return={m.get('total_return_pct', 'N/A')} "
                        f"sharpe={m.get('sharpe', 'N/A')} "
                        f"max_dd={m.get('max_drawdown_pct', 'N/A')}"
                    )
                else:
                    all_results.append({"symbol": sym, "strategy": sname, "error": "回测失败"})
            except Exception as e:
                logger.error(f"  {sym}/{sname}: 失败 - {e}")
                all_results.append({"symbol": sym, "strategy": sname, "error": str(e)})

    df = pd.DataFrame(all_results)
    save_csv(df, output_dir / "e1_baseline_metrics.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# E2: 等权信号融合
# ══════════════════════════════════════════════════════════════════════════════


def run_e2_equal_weight(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E2: 等权信号融合回测。
    所有策略信号等权融合，单一品种组合。

    Returns:
        汇总指标DataFrame
    """
    logger.info("E2: 等权信号融合回测")
    symbols: List[str] = config.get("symbols", [])
    strategy_names = _get_strategy_names(config)
    bt_cfg = config["backtest"]
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    for sym in symbols:
        logger.info(f"  品种: {sym}")
        try:
            runner = get_pybroker_runner(
                data_source, config, strategies=strategy_names,
            )
            result = _safe_run_backtest(
                runner, bt_cfg["full_start_date"], bt_cfg["full_end_date"],
                f"E2_{sym}",
            )
            if result is None:
                continue

            m = format_metrics(result.metrics)
            m["symbol"] = sym
            m["experiment"] = "E2_等权融合"
            all_results.append(m)
            logger.info(
                f"  portfolio: sharpe={m.get('sharpe', 'N/A')} "
                f"return={m.get('total_return_pct', 'N/A')}"
            )

            eq = result.equity_curve
            if eq is not None and not eq.empty:
                save_csv(
                    eq.assign(symbol=sym),
                    output_dir / f"e2_equity_{sym.replace('.', '_')}.csv",
                )
                _plot_equity_curve(
                    eq, sym, "E2_等权融合",
                    charts_dir / f"e2_equity_{sym.replace('.', '_')}.png",
                )
            
            # 保存调仓决策日志
            if result.switch_log is not None and not result.switch_log.empty:
                save_csv(
                    result.switch_log.assign(symbol=sym),
                    output_dir / f"e2_switch_log_{sym.replace('.', '_')}.csv",
                )
                logger.info(f"  {sym}: 已保存 {len(result.switch_log)} 条调仓决策记录")
        except Exception as e:
            logger.error(f"  {sym}: 失败 - {e}")

    df = pd.DataFrame(all_results)
    save_csv(df, output_dir / "e2_equal_weight_metrics.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# E3: 环境动态加权
# ══════════════════════════════════════════════════════════════════════════════


def run_e3_dynamic_weight(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E3: 环境动态加权回测（execute融合模式）。
    根据市场环境动态分配各策略权重。

    Returns:
        汇总指标DataFrame
    """
    logger.info("E3: 环境动态加权回测（execute 融合模式）")
    symbols: List[str] = config.get("symbols", [])
    strategy_names = _get_strategy_names(config)
    bt_cfg = config["backtest"]

    all_results: List[Dict[str, Any]] = []
    for sym in symbols:
        logger.info(f"  品种: {sym}")
        try:
            runner = get_pybroker_runner(
                data_source, config, strategies=strategy_names,
            )
            result = _safe_run_backtest(
                runner, bt_cfg["full_start_date"], bt_cfg["full_end_date"],
                f"E3_{sym}", use_execute_fusion=True,
            )
            if result is None:
                continue

            m = format_metrics(result.metrics)
            m["symbol"] = sym
            m["experiment"] = "E3_动态权重"
            all_results.append(m)
            logger.info(
                f"  portfolio: return={m.get('total_return_pct', 'N/A')} "
                f"sharpe={m.get('sharpe', 'N/A')}"
            )

            eq = result.equity_curve
            if eq is not None and not eq.empty:
                save_csv(
                    eq.assign(symbol=sym),
                    output_dir / f"e3_equity_{sym.replace('.', '_')}.csv",
                )
            if result.regime_history is not None and not result.regime_history.empty:
                save_csv(
                    result.regime_history,
                    output_dir / f"e3_regime_{sym.replace('.', '_')}.csv",
                )
            
            # 保存调仓决策日志
            if result.switch_log is not None and not result.switch_log.empty:
                save_csv(
                    result.switch_log.assign(symbol=sym),
                    output_dir / f"e3_switch_log_{sym.replace('.', '_')}.csv",
                )
                logger.info(f"  {sym}: 已保存 {len(result.switch_log)} 条调仓决策记录")
        except Exception as e:
            logger.error(f"  {sym}: 失败 - {e}")

    df = pd.DataFrame(all_results)
    save_csv(df, output_dir / "e3_dynamic_weight_metrics.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# E5: 多品种分散
# ══════════════════════════════════════════════════════════════════════════════


def run_e5_multi_symbol(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    E5: 多品种分散回测。

    逻辑：
    1. 对每个品种独立运行回测
    2. 提取各品种日收益率序列
    3. 日期对齐后等权合并为组合
    4. 计算组合绩效指标

    Returns:
        组合指标与净值，失败返回None
    """
    logger.info("E5: 多品种分散回测")
    symbols: List[str] = config.get("symbols", [])
    strategy_names = _get_strategy_names(config)
    bt_cfg = config["backtest"]
    initial_cash = float(bt_cfg.get("initial_cash", 1_000_000))
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    # 各品种日收益率字典
    strategy_returns_by_symbol: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        try:
            runner = get_pybroker_runner(
                data_source, config, strategies=strategy_names,
            )
            result = _safe_run_backtest(
                runner, bt_cfg["full_start_date"], bt_cfg["full_end_date"],
                f"E5_{sym}",
            )
            if result is None:
                continue

            eq = result.equity_curve
            if eq is None or eq.empty:
                logger.warning(f"  {sym}: 无净值数据，跳过")
                continue

            # 计算日收益率，确保日期索引标准化
            eq_sorted = eq.sort_values("date").copy()
            eq_sorted["date"] = pd.to_datetime(eq_sorted["date"])
            eq_sorted["daily_return"] = eq_sorted["equity"].pct_change()
            # 过滤掉空收益率和正负无穷
            rets = eq_sorted[["date", "daily_return"]].dropna()
            rets = rets[rets["daily_return"].apply(_is_valid_number)]
            strategy_returns_by_symbol[sym] = rets.set_index("date")
        except Exception as e:
            logger.error(f"  {sym}: 失败 - {e}")

    if len(strategy_returns_by_symbol) == 0:
        logger.warning("E5: 无有效品种收益数据，跳过多品种组合计算")
        return None

    if len(strategy_returns_by_symbol) == 1:
        logger.warning("E5: 仅1个有效品种，直接使用该品种作为组合")
        sym, rets_df = next(iter(strategy_returns_by_symbol.items()))
        single_ret = rets_df.reset_index()
        single_ret = single_ret[single_ret["daily_return"].apply(_is_valid_number)]
        if single_ret.empty:
            return None
        portfolio_equity = (1.0 + single_ret["daily_return"]).cumprod() * initial_cash
        multi_eq = pd.DataFrame({"date": single_ret["date"], "equity": portfolio_equity.values})
        multi_metrics = PerformanceEvaluator.compute_metrics(portfolio_equity)
        m = format_metrics(multi_metrics)
        logger.info(f"  单品种({sym})组合: return={m.get('total_return_pct','N/A')} sharpe={m.get('sharpe','N/A')}")
        save_csv(multi_eq, output_dir / "e5_multi_symbol_equity.csv")
        _plot_equity_curve(multi_eq, f"单品种 {sym}", "E5_多品种分散", charts_dir / "e5_multi_symbol_equity.png")
        return {"metrics": m, "equity": multi_eq}

    logger.info("  计算多品种等权组合...")
    try:
        # 日期对齐合并所有品种收益率
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

        # 缺失值填0（某品种某天无交易则收益为0）
        combined_rets = combined_rets.fillna(0.0)
        # 等权平均
        portfolio_ret = combined_rets.mean(axis=1)
        # 组合净值
        portfolio_equity = (1.0 + portfolio_ret).cumprod() * initial_cash
        multi_eq = pd.DataFrame({
            "date": portfolio_equity.index,
            "equity": portfolio_equity.values,
        })

        multi_metrics = PerformanceEvaluator.compute_metrics(portfolio_equity)
        m = format_metrics(multi_metrics)
        logger.info(
            f"  多品种组合: return={m.get('total_return_pct', 'N/A')} "
            f"sharpe={m.get('sharpe', 'N/A')} "
            f"max_dd={m.get('max_drawdown_pct', 'N/A')}"
        )

        save_csv(multi_eq, output_dir / "e5_multi_symbol_equity.csv")

        # 相关性矩阵
        corr_matrix = combined_rets.corr()
        save_csv(corr_matrix, output_dir / "e5_correlation_matrix.csv")

        _plot_equity_curve(
            multi_eq, "多品种等权组合", "E5_多品种分散",
            charts_dir / "e5_multi_symbol_equity.png",
        )

        return {"metrics": m, "equity": multi_eq}
    except Exception as e:
        logger.error(f"E5 多品种组合计算失败: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# E6: WalkForward 滚动验证
# ══════════════════════════════════════════════════════════════════════════════


def run_e6_walkforward(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E6: WalkForward滚动验证。
    对每个策略执行滚动窗口回测，评估参数稳定性。

    Returns:
        各窗口汇总指标DataFrame
    """
    logger.info("E6: WalkForward 滚动验证")
    bt_cfg = config["backtest"]
    strategy_names = _get_strategy_names(config)

    all_wf_metrics: List[Dict[str, Any]] = []
    for sname in strategy_names:
        try:
            runner = get_pybroker_runner(data_source, config, strategies=[sname])
            wf_result = runner.walkforward(
                start_date=bt_cfg["full_start_date"],
                end_date=bt_cfg["full_end_date"],
            )
            for w in wf_result.windows:
                w["strategy"] = sname
                all_wf_metrics.append(w)
            logger.info(
                f"  {sname}: {len(wf_result.windows)} 窗口, "
                f"avg_sharpe={wf_result.overall_metrics.get('sharpe', 'N/A')}"
            )
        except Exception as e:
            logger.error(f"  {sname} WalkForward 失败: {e}")

    df = pd.DataFrame(all_wf_metrics) if all_wf_metrics else pd.DataFrame()
    if not df.empty:
        save_csv(df, output_dir / "e6_walkforward_metrics.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# E7: 样本外验证
# ══════════════════════════════════════════════════════════════════════════════


def run_e7_out_of_sample(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    E7: 样本外验证。
    将数据分为样本内和样本外两段，分别回测并比较Sharpe衰减率。

    Returns:
        汇总指标DataFrame
    """
    logger.info("E7: 样本外验证")
    symbols: List[str] = config.get("symbols", [])
    strategy_names = _get_strategy_names(config)
    bt_cfg = config["backtest"]
    in_sample_end = str(bt_cfg.get("in_sample_end_date", bt_cfg["full_end_date"]))
    out_sample_start = str(bt_cfg.get("out_sample_start_date", in_sample_end))

    all_results: List[Dict[str, Any]] = []
    primary_symbol = symbols[0] if symbols else None

    for sym in symbols:
        logger.info(f"  品种: {sym}")
        try:
            # 样本内回测
            runner_in = get_pybroker_runner(
                data_source, config, strategies=strategy_names,
            )
            result_in = _safe_run_backtest(
                runner_in, bt_cfg["full_start_date"], in_sample_end,
                f"E7_in_{sym}",
            )
            if result_in is not None:
                m_in = format_metrics(result_in.metrics)
                m_in["symbol"] = sym
                m_in["split"] = "in_sample"
                all_results.append(m_in)

                # 保存主品种的样本内净值曲线
                if sym == primary_symbol:
                    eq_in = result_in.equity_curve
                    if eq_in is not None and not eq_in.empty:
                        save_csv(eq_in, output_dir / "e7_equity_in_sample.csv")

            # 样本外回测
            runner_out = get_pybroker_runner(
                data_source, config, strategies=strategy_names,
            )
            result_out = _safe_run_backtest(
                runner_out, out_sample_start, bt_cfg["full_end_date"],
                f"E7_out_{sym}",
            )
            if result_out is not None:
                m_out = format_metrics(result_out.metrics)
                m_out["symbol"] = sym
                m_out["split"] = "out_sample"
                all_results.append(m_out)

                # 保存主品种的样本外净值曲线
                if sym == primary_symbol:
                    eq_out = result_out.equity_curve
                    if eq_out is not None and not eq_out.empty:
                        save_csv(eq_out, output_dir / "e7_equity_out_sample.csv")

            # 计算Sharpe衰减率
            if result_in is not None and result_out is not None:
                sharpe_in = _safe_float(result_in.metrics.get("sharpe", 0))
                sharpe_out = _safe_float(result_out.metrics.get("sharpe", 0))
                if abs(sharpe_in) > _EPSILON:
                    decay = (sharpe_in - sharpe_out) / abs(sharpe_in)
                    is_qualified = decay < _SAFE_DECAY_THRESHOLD
                    logger.info(
                        f"  Sharpe衰减率: {decay:.1%} "
                        f"{'合格' if is_qualified else '不合格'}"
                    )
        except Exception as e:
            logger.error(f"  {sym}: 失败 - {e}")

    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    save_csv(df, output_dir / "e7_out_of_sample_metrics.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# E8: Bootstrap 置信区间
# ══════════════════════════════════════════════════════════════════════════════


def run_e8_bootstrap(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> Tuple[List[float], pd.DataFrame]:
    """
    E8: Bootstrap置信区间。

    对回测收益序列进行Bootstrap重采样，估计Sharpe等指标的置信区间。
    优先使用系统Bootstrap，失败则回退到MetricsCalculator。

    Returns:
        (sharpe_samples, 置信区间DataFrame)
    """
    logger.info("E8: Bootstrap 置信区间")
    bt_cfg = config["backtest"]
    bs_config: Dict[str, Any] = config.get("bootstrap", {})
    n_samples = int(bs_config.get("n_samples", 5000))
    strategy_names = _get_strategy_names(config)
    default_strategy = strategy_names[:1] if strategy_names else ["ts_momentum"]
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    runner = get_pybroker_runner(data_source, config, strategies=default_strategy)
    result = _safe_run_backtest(
        runner, bt_cfg["full_start_date"], bt_cfg["full_end_date"],
        "E8_base",
    )

    if result is None or result.equity_curve is None or result.equity_curve.empty:
        logger.warning("E8: 无净值数据，跳过Bootstrap")
        return [], pd.DataFrame()

    bootstrap_result: Any = None
    try:
        bootstrap_result = runner.bootstrap_metrics(n_samples=n_samples)
        logger.info(f"  系统Bootstrap完成: {n_samples} 样本")
    except Exception as e:
        logger.warning(f"  系统Bootstrap失败: {e}, 回退到MetricsCalculator")
        try:
            equity = result.equity_curve["equity"].values
            bootstrap_result = MetricsCalculator.bootstrap_confidence_interval(
                equity, n_samples=n_samples,
            )
            logger.info(f"  MetricsCalculator Bootstrap完成: {n_samples} 样本")
        except Exception as e2:
            logger.error(f"  MetricsCalculator Bootstrap也失败: {e2}")
            return [], pd.DataFrame()

    if bootstrap_result is None:
        return [], pd.DataFrame()

    # 统一处理结果格式
    if isinstance(bootstrap_result, dict):
        # 检查是否为结构化结果（{metric: {mean, ci_lower, ci_upper}}）
        first_val = next(iter(bootstrap_result.values()), None)
        if isinstance(first_val, dict) and "mean" in first_val:
            rows: List[Dict[str, Any]] = []
            for metric_name, vals in bootstrap_result.items():
                if isinstance(vals, dict):
                    rows.append({"metric": metric_name, **vals})
            df_ci = pd.DataFrame(rows)
            save_csv(df_ci, output_dir / "e8_bootstrap_confidence_intervals.csv")
            sharpe_ci = bootstrap_result.get("sharpe", {})
            logger.info(f"  Bootstrap置信区间: {sharpe_ci}")
            return [], df_ci

        # 兼容老格式：直接取第一个列表值
        sharpe_samples: List[float] = []
        for val in bootstrap_result.values():
            if isinstance(val, list) and len(val) > 0:
                sharpe_samples = val
                break

        if sharpe_samples:
            df_samples = pd.DataFrame({"sharpe": sharpe_samples})
            save_csv(df_samples, output_dir / "e8_bootstrap_samples.csv")

            # 绘制分布直方图
            try:
                fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE_WIDE)
                ax.hist(sharpe_samples, bins=50, alpha=0.7, color="#1f77b4", edgecolor="black")
                ax.axvline(np.percentile(sharpe_samples, 5), color="#ff7f0e", linestyle="--", label="5% CI")
                ax.axvline(np.percentile(sharpe_samples, 95), color="#ff7f0e", linestyle="--", label="95% CI")
                ax.axvline(np.mean(sharpe_samples), color="#d62728", linestyle="-", label="Mean")
                ax.set_xlabel("Sharpe Ratio", fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.set_title(f"Bootstrap Sharpe Ratio Distribution (n={n_samples})", fontsize=14)
                ax.legend(fontsize=11)
                ax.grid(alpha=0.3)
                fig.savefig(
                    charts_dir / "bootstrap_sharpe_distribution.png",
                    dpi=config.get("output", {}).get("chart_dpi", _DEFAULT_CHART_DPI),
                    bbox_inches="tight",
                )
                plt.close(fig)
                logger.info("  Bootstrap图表已保存")
            except Exception as e:
                logger.error(f"  Bootstrap绘图失败: {e}")
                plt.close("all")

            return sharpe_samples, df_samples

    return [], pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# E9: 蒙特卡洛模拟
# ══════════════════════════════════════════════════════════════════════════════


def run_e9_monte_carlo(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    E9: 蒙特卡洛模拟。

    基于历史收益率序列，通过有放回重采样模拟未来净值路径分布。
    计算终值分布、破产概率和最大回撤分布。

    Returns:
        模拟结果DataFrame，失败返回None
    """
    logger.info("E9: 蒙特卡洛模拟")
    bt_cfg = config["backtest"]
    strategy_names = _get_strategy_names(config)
    mc_config: Dict[str, Any] = config.get("monte_carlo", {})
    n_simulations = int(mc_config.get("n_simulations", 1000))
    random_seed = int(mc_config.get("random_seed", 42))
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    try:
        runner = get_pybroker_runner(
            data_source, config, strategies=strategy_names,
        )
        result = _safe_run_backtest(
            runner, bt_cfg["full_start_date"], bt_cfg["full_end_date"],
            "E9_base",
        )
        if result is None:
            return None

        eq = result.equity_curve
        if eq is None or eq.empty:
            logger.warning("E9: 无净值数据，跳过蒙特卡洛模拟")
            return None

        eq_sorted = eq.sort_values("date")
        returns = eq_sorted["equity"].pct_change().dropna()
        # 过滤异常值
        returns = returns[returns.apply(_is_valid_number)]

        if len(returns) == 0:
            logger.warning("E9: 无有效收益率数据")
            return None

        n_days = len(returns)
        rng = np.random.default_rng(random_seed)
        ret_array = returns.values

        sim_equities = np.zeros((n_simulations, n_days + 1))
        sim_equities[:, 0] = 1.0

        for i in range(n_simulations):
            sampled = rng.choice(ret_array, size=n_days, replace=True)
            sim_equities[i, 1:] = np.cumprod(1.0 + sampled)

        final_values = sim_equities[:, -1]
        # 计算最大回撤时防止除零
        peak_equities = np.maximum.accumulate(sim_equities, axis=1)
        peak_equities_safe = np.where(peak_equities > 0, peak_equities, 1.0)
        drawdowns = sim_equities / peak_equities_safe - 1.0
        max_drawdowns = np.min(drawdowns, axis=1)

        bankruptcy_prob = float(np.mean(final_values < 0.8))
        logger.info(f"  模拟次数: {n_simulations}")
        logger.info(f"  终值均值: {np.mean(final_values):.4f}, 中位数: {np.median(final_values):.4f}")
        logger.info(f"  破产概率(终值<0.8): {bankruptcy_prob:.2%}")

        mc_results = pd.DataFrame({
            "sim_id": range(n_simulations),
            "final_value": final_values,
            "max_drawdown": max_drawdowns,
        })
        save_csv(mc_results, output_dir / "e9_monte_carlo_results.csv")

        lower = np.percentile(sim_equities, 5, axis=0)
        upper = np.percentile(sim_equities, 95, axis=0)
        median = np.percentile(sim_equities, 50, axis=0)

        _plot_monte_carlo(median, lower, upper, charts_dir / "e9_monte_carlo.png")
        return mc_results
    except Exception as e:
        logger.error(f"  蒙特卡洛模拟失败: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# E10: HTML 报告生成
# ══════════════════════════════════════════════════════════════════════════════


def run_e10_html_report(
    config: Dict[str, Any],
    results: Dict[str, PyBrokerResult],
    output_dir: Path,
    optimization_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    E10: 生成完整的量化回测分析 HTML 报告。

    使用 core.report_builder 模块生成包含 KPI 卡片、策略对比表、
    Chart.js 可视化图表、风险分析的专业报告。
    """
    from core.report_builder import generate_report as build_report

    logger.info("E10: 生成完整 HTML 分析报告")

    # 将 PyBrokerResult 转换为 report_builder 所需格式
    strategies_data = {}
    for name, res in results.items():
        sd = {"metrics": dict(res.metrics) if hasattr(res, "metrics") and res.metrics else {}}
        if hasattr(res, "equity_curve") and res.equity_curve is not None and not res.equity_curve.empty:
            df = res.equity_curve
            sd["dates"] = df["date"].astype(str).tolist()
            sd["equity"] = df["equity"].astype(float).tolist()
        strategies_data[name] = sd

    if not strategies_data:
        logger.warning("E10: 无策略数据，跳过报告生成")
        return

    try:
        build_report(
            output_dir=str(output_dir),
            strategies_data=strategies_data,
            title="量化回测分析报告",
            subtitle=f"PyBroker 多策略回测 · {datetime.now().strftime('%Y-%m-%d')}",
            report_name="backtest_report_full.html",
        )
        logger.info(f"E10: 报告已保存至 {output_dir / 'backtest_report_full.html'}")
    except Exception as e:
        logger.error(f"E10 报告生成失败: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# E11: 滚动IC加权与因子衰减分析
# ══════════════════════════════════════════════════════════════════════════════


def _compute_factor_scores_from_ohlcv(
    ohlcv: pd.DataFrame,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    从OHLCV数据计算各因子得分（不依赖PyBroker）。

    因子：
      - ts_momentum: 20日收益率，归一化到 [-1, 1]
      - roll_yield: 价格偏离20日均线的百分比，归一化到 [-1, 1]
      - alpha019: 简化版短期反转 × 长期动量排名
      - alpha032: 简化版 收盘价与VWAP的相关系数

    Args:
        ohlcv: 含 close, high, low, volume 列的DataFrame，按日期排序
        atr_period: ATR计算周期

    Returns:
        含各因子得分和前瞻收益的DataFrame
    """
    df = ohlcv.copy().sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)

    # ── ts_momentum: 20日收益率归一化 ──
    ret_20 = close.pct_change(20)
    atr = _compute_atr(df["high"], df["low"], close, atr_period)
    mom_norm = np.clip(ret_20 / (atr / close + 1e-8) * 0.1, -1.0, 1.0)
    df["ts_momentum"] = mom_norm.fillna(0.0)

    # ── roll_yield: 价格偏离20日均线 ──
    sma_20 = close.rolling(20, min_periods=1).mean()
    spread_pct = (close - sma_20) / (sma_20 + 1e-8) * 100
    df["roll_yield"] = np.clip(spread_pct / 5.0, -1.0, 1.0).fillna(0.0)

    # ── alpha019: 简化版 ──
    short_term = close - close.shift(7) + (close - close.shift(7)).shift(7)
    sign_component = -np.sign(short_term.fillna(0.0))
    returns = close.pct_change()
    cum_ret_250 = returns.rolling(250, min_periods=1).apply(
        lambda x: np.prod(1 + x) - 1, raw=False
    )
    cum_rank = cum_ret_250.rank(pct=True).fillna(0.5)
    df["alpha019"] = np.clip(sign_component * (1 + cum_rank) * 0.3, -1.0, 1.0).fillna(0.0)

    # ── alpha032: 简化版 收盘价与VWAP的相关系数 ──
    typical_price = (df["high"] + df["low"] + close) / 3
    vwap = (typical_price * df["volume"]).rolling(10, min_periods=1).sum() / \
           df["volume"].rolling(10, min_periods=1).sum().replace(0, 1)
    corr_vwap = close.rolling(10, min_periods=1).corr(vwap)
    df["alpha032"] = np.clip(corr_vwap.fillna(0.0), -1.0, 1.0)

    # ── 前瞻收益（5日） ──
    df["forward_return"] = close.shift(-5) / close - 1.0

    return df


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算ATR（平均真实波幅）。"""
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float).shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - c).abs(),
        (l - c).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _plot_ic_analysis(
    ic_df: pd.DataFrame,
    title: str,
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
) -> None:
    """绘制滚动IC时间序列和动态权重图。"""
    if ic_df is None or ic_df.empty:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        factor_cols = [c for c in ic_df.columns if c.startswith("ic_")]
        if not factor_cols:
            return

        fig, axes = plt.subplots(2, 1, figsize=_DEFAULT_FIGSIZE_FULL, sharex=True)
        dates = pd.to_datetime(ic_df["date"]) if "date" in ic_df.columns else range(len(ic_df))

        # 上图：各因子滚动IC
        for col in factor_cols:
            label = col.replace("ic_", "")
            axes[0].plot(dates, ic_df[col].values, linewidth=1, alpha=0.8, label=label)
        axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_title(f"{title} — 滚动IC", fontsize=14)
        axes[0].set_ylabel("IC")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # 下图：动态权重
        weight_cols = [c for c in ic_df.columns if c.startswith("w_")]
        if weight_cols:
            for col in weight_cols:
                label = col.replace("w_", "")
                axes[1].plot(dates, ic_df[col].values, linewidth=1, alpha=0.8, label=label)
            axes[1].set_ylabel("权重")
            axes[1].legend(fontsize=9)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].set_visible(False)

        if "date" in ic_df.columns:
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[-1].set_xlabel("日期")

        plt.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"IC分析图表已保存: {path}")
    except Exception as e:
        logger.error(f"IC分析绘图失败: {e}")
        plt.close("all")


def _plot_decay_analysis(
    decay_df: pd.DataFrame,
    title: str,
    path: Path,
    dpi: int = _DEFAULT_CHART_DPI,
) -> None:
    """绘制因子衰减状态图。"""
    if decay_df is None or decay_df.empty:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        status_cols = [c for c in decay_df.columns if c.startswith("status_")]
        if not status_cols:
            return

        status_map = {"healthy": 0, "warning": 1, "decaying": 2, "dead": 3}
        dates = pd.to_datetime(decay_df["date"]) if "date" in decay_df.columns else range(len(decay_df))

        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE_WIDE)
        for col in status_cols:
            label = col.replace("status_", "")
            numeric = decay_df[col].map(status_map).fillna(0).astype(int)
            ax.plot(dates, numeric.values, linewidth=1.5, alpha=0.8, label=label, marker=".", markersize=2)

        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["健康", "警告", "衰减", "失效"])
        ax.set_title(f"{title} — 因子衰减状态", fontsize=14)
        ax.set_ylabel("状态")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if "date" in decay_df.columns:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.set_xlabel("日期")

        plt.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"衰减分析图表已保存: {path}")
    except Exception as e:
        logger.error(f"衰减分析绘图失败: {e}")
        plt.close("all")


def run_e11_factor_analysis(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    E11: 滚动IC加权与因子衰减分析。

    对每个品种独立计算因子得分、滚动IC、动态权重和衰减状态，
    输出CSV和图表。

    分析维度：
      1. 各因子滚动IC时间序列
      2. 动态权重变化（基于|IC|的加权）
      3. 因子衰减状态检测
      4. IC统计摘要（均值、标准差、IR）

    Returns:
        {symbol: {ic_df, decay_df, summary}} 字典
    """
    logger.info("E11: 滚动IC加权与因子衰减分析")
    symbols: List[str] = config.get("symbols", [])
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

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
            sym_df = data_source.query(
                data_source.date_range[0], data_source.date_range[1], symbols=[symbol]
            )
            if sym_df is None or len(sym_df) < 60:
                logger.warning(f"    {symbol}: 数据不足，跳过")
                continue

            # 计算因子得分
            scored = _compute_factor_scores_from_ohlcv(sym_df)
            factor_names = ["ts_momentum", "roll_yield", "alpha019", "alpha032"]

            # 初始化引擎
            ic_engine = RollingICWeightEngine(ic_config)
            decay_monitor = FactorDecayMonitor(decay_config)

            ic_rows: List[Dict[str, Any]] = []

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

                ic_engine.update(factor_scores, forward_ret, symbol)

                current_ic = ic_engine.current_ic
                for name, ic_val in current_ic.items():
                    decay_monitor.update(name, ic_val, str(row["date"])[:10])

                # 每10条记录一次
                if i % 10 == 0:
                    current_weights = ic_engine.get_dynamic_weights()
                    ic_row = {"date": str(row["date"])[:10]}
                    for name, ic_val in current_ic.items():
                        ic_row[f"ic_{name}"] = round(ic_val, 6)
                    for name, w in current_weights.items():
                        ic_row[f"w_{name}"] = round(w, 4)
                    ic_rows.append(ic_row)

            ic_df = pd.DataFrame(ic_rows)
            if not ic_df.empty:
                save_csv(ic_df, output_dir / f"e11_ic_{symbol.replace('.', '_')}.csv")

            # 衰减检测
            alerts = decay_monitor.check_decay()
            decay_rows = []
            for name in factor_names:
                if name in decay_monitor._ic_history:
                    ic_series = decay_monitor._ic_history[name]
                    decay_rows.append({
                        "date": str(scored["date"].iloc[-1])[:10],
                        "factor": name,
                        "current_ic": round(ic_series[-1], 6) if ic_series else 0.0,
                        "mean_ic": round(np.mean(ic_series), 6) if ic_series else 0.0,
                        "status": decay_monitor.current_status.get(name, DecayStatus.HEALTHY).value,
                    })
            decay_df = pd.DataFrame(decay_rows)
            if not decay_df.empty:
                save_csv(decay_df, output_dir / f"e11_decay_{symbol.replace('.', '_')}.csv")

            # IC统计摘要
            ic_summary = ic_engine.get_ic_summary()
            for name, stats in ic_summary.items():
                summary_rows.append({
                    "symbol": symbol,
                    "factor": name,
                    "mean_ic": round(stats.get("mean", 0.0), 6),
                    "std_ic": round(stats.get("std", 0.0), 6),
                    "ir": round(stats.get("ir", 0.0), 4),
                    "current_ic": round(stats.get("current", 0.0), 6),
                    "current_weight": round(ic_engine.get_dynamic_weights().get(name, 0.0), 4),
                })

            # 图表
            if not ic_df.empty:
                _plot_ic_analysis(
                    ic_df, f"{symbol} 滚动IC与动态权重",
                    charts_dir / f"e11_ic_{symbol.replace('.', '_')}.png",
                )

            all_results[symbol] = {
                "ic_df": ic_df,
                "decay_df": decay_df,
                "alerts": alerts,
                "final_weights": ic_engine.get_dynamic_weights(),
            }

            final_weights = ic_engine.get_dynamic_weights()
            logger.info(
                f"    {symbol}: 最终权重={({k: round(v, 4) for k, v in final_weights.items()})}"
            )

        except Exception as e:
            logger.error(f"    {symbol} 因子分析失败: {e}")

    # 汇总
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        save_csv(summary_df, output_dir / "e11_ic_summary.csv")

        logger.info("\n  因子IC汇总:")
        for _, row in summary_df.iterrows():
            logger.info(
                f"    {row['symbol']}/{row['factor']}: "
                f"mean_IC={row['mean_ic']:.4f}, IR={row['ir']:.2f}, "
                f"weight={row['current_weight']:.4f}"
            )

    return all_results

def _setup_logging(config: Dict[str, Any]) -> None:
    """配置日志系统，失败时降级为控制台输出。"""
    log_config: Dict[str, Any] = config.get("logging", {})
    if not log_config:
        logger.add(sys.stdout, level="INFO")
        return

    try:
        log_dir = Path(log_config.get("log_dir", "logs"))
        log_dir.mkdir(exist_ok=True)
        logger.remove()
        logger.add(
            log_dir / log_config.get("log_file", "backtest.log"),
            rotation=log_config.get("rotation", "100 MB"),
            retention=log_config.get("retention", "30 days"),
            level=log_config.get("log_level", "INFO"),
        )
        logger.add(log_dir / log_config.get("error_file", "error.log"), level="ERROR")
        logger.add(sys.stdout, level=log_config.get("log_level", "INFO"))
    except Exception as e:
        logger.add(sys.stdout, level="INFO")
        logger.warning(f"日志文件配置失败，仅使用控制台输出: {e}")


def _run_experiment_safely(
    name: str,
    func: Callable,
    *args,
) -> Any:
    """
    安全执行单个实验，捕获异常不中断整体流程。

    Args:
        name: 实验名称
        func: 实验函数
        *args: 传递给实验函数的参数

    Returns:
        实验结果，失败返回None
    """
    logger.info("=" * 60)
    logger.info(f"{name}")
    try:
        return func(*args)
    except Exception as e:
        logger.error(f"{name} 执行失败: {e}", exc_info=True)
        return None


def _collect_results_for_report(
    data_source: PyBrokerDataSource,
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, PyBrokerResult]:
    """
    收集全部策略回测结果用于报告生成。

    运行所有配置策略、融合策略和切换策略，并将结果写入 CSV 供后续复用。
    """
    bt_cfg: Dict[str, Any] = config.get("backtest", {})
    strategy_names = _get_strategy_names(config)
    results: Dict[str, PyBrokerResult] = {}

    if not strategy_names:
        logger.warning("报告生成: 无可用策略，跳过")
        return results

    start_date = str(bt_cfg.get("full_start_date", "2016-01-01"))
    end_date = str(bt_cfg.get("full_end_date", "2026-05-01"))

    # 收集所有单策略结果
    all_metrics_rows: List[Dict[str, Any]] = []
    for sname in strategy_names:
        try:
            runner = get_pybroker_runner(data_source, config, strategies=[sname])
            res = _safe_run_backtest(
                runner, start_date, end_date, f"Report_{sname}",
            )
            if res is not None:
                exp_name = f"E1_{sname}"
                results[exp_name] = res
                # 保存净值曲线
                eq = res.equity_curve
                if eq is not None and not eq.empty:
                    save_csv(eq, output_dir / f"e1_equity_{sname}.csv")
                # 收集指标
                m = {"experiment": exp_name}
                m.update(res.metrics)
                all_metrics_rows.append(m)
                logger.info(f"  收集策略: {exp_name}")
        except Exception as e:
            logger.error(f"报告数据 - 策略 {sname}: {e}")

    # 融合策略结果
    try:
        runner = get_pybroker_runner(
            data_source, config, strategies=strategy_names,
        )
        fusion_res = _safe_run_backtest(
            runner, start_date, end_date, "Report_Fusion",
        )
        if fusion_res is not None:
            results["E2_Fusion"] = fusion_res
            eq = fusion_res.equity_curve
            if eq is not None and not eq.empty:
                save_csv(eq, output_dir / "e2_equity_fusion.csv")
            m = {"experiment": "E2_Fusion"}
            m.update(fusion_res.metrics)
            all_metrics_rows.append(m)
            logger.info("  收集策略: E2_Fusion")
    except Exception as e:
        logger.error(f"报告数据 - Fusion: {e}")

    # 保存汇总指标 CSV
    if all_metrics_rows:
        save_csv(pd.DataFrame(all_metrics_rows), output_dir / "all_metrics.csv")

    return results


def run_backtest_with_config(
    config: Dict[str, Any],
    optimization_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    运行完整回测流程（内部使用）

    Args:
        config: 配置字典
        optimization_info: 可选，参数优化信息
    """
    _setup_logging(config)

    output_dir = Path(config.get("output", {}).get("output_dir", "results"))
    output_dir.mkdir(exist_ok=True)

    # ── 数据加载 ──
    try:
        logger.info("加载数据...")
        phone, password = get_tqsdk_credentials()
        data_source = create_hybrid_data_source(
            phone=phone,
            password=password,
            symbols=config.get("symbols"),
            data_dir=config.get("data", {}).get("csv_data_dir", "data"),
            data_length=config.get("data", {}).get("tqsdk_data_length", 4000),
        )
        pybroker_df = data_source.to_pybroker_df()
        if pybroker_df is not None:
            save_csv(pybroker_df, output_dir / "data_summary.csv")
    except Exception as e:
        logger.exception(f"数据加载致命错误: {e}")
        sys.exit(1)

    # ── 实验执行 ──
    _run_experiment_safely("E1: 单策略基线回测", run_e1_single_strategy_baselines, data_source, config, output_dir)
    _run_experiment_safely("E2: 等权信号融合", run_e2_equal_weight, data_source, config, output_dir)
    _run_experiment_safely("E3: 环境动态加权", run_e3_dynamic_weight, data_source, config, output_dir)
    _run_experiment_safely("E5: 多品种分散", run_e5_multi_symbol, data_source, config, output_dir)
    _run_experiment_safely("E6: WalkForward滚动验证", run_e6_walkforward, data_source, config, output_dir)
    _run_experiment_safely("E7: 样本外验证", run_e7_out_of_sample, data_source, config, output_dir)
    _run_experiment_safely("E8: Bootstrap置信区间", run_e8_bootstrap, data_source, config, output_dir)
    _run_experiment_safely("E9: 蒙特卡洛模拟", run_e9_monte_carlo, data_source, config, output_dir)

    # ── E10: 报告生成 ──
    logger.info("=" * 60)
    logger.info("E10: HTML 报告生成")
    try:
        results = _collect_results_for_report(data_source, config, output_dir)
        run_e10_html_report(config, results, output_dir, optimization_info)
    except Exception as e:
        logger.error(f"E10 报告生成失败: {e}")

    # ── E11: 滚动IC与因子衰减分析（先运行，权重已通过引擎实时集成） ──
    e11_result = _run_experiment_safely("E11: 滚动IC加权与因子衰减分析", run_e11_factor_analysis, data_source, config, output_dir)

    # ── E12: 使用E11优化权重重新回测 ──
    if e11_result and config.get("backtest", {}).get("use_rolling_ic", True):
        logger.info("=" * 60)
        logger.info("E12: 基于滚动IC优化权重的增强回测")
        try:
            optimized_weights = _extract_optimized_weights(e11_result)
            if optimized_weights:
                logger.info(f"  应用E11优化权重: {optimized_weights}")
                enhanced_config = _apply_ic_weights_to_config(config, optimized_weights)
                enhanced_output = output_dir / "e12_ic_weighted"
                enhanced_output.mkdir(exist_ok=True)
                _run_experiment_safely(
                    "E12: IC加权增强回测",
                    run_e2_equal_weight, data_source, enhanced_config, enhanced_output,
                )
        except Exception as e:
            logger.error(f"E12 IC加权增强回测失败: {e}")


def main() -> None:
    """主执行入口：依次运行E1-E10实验并生成报告。"""
    import argparse
    parser = argparse.ArgumentParser(description="多策略量化回测系统")
    parser.add_argument("--optimize", action="store_true",
                        help="先运行参数优化，再执行回测")
    args = parser.parse_args()

    print("=" * 80)
    print("  多策略量化回测系统 — 完整回测执行（整合版）")
    print(f"  开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        config = load_config()
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        sys.exit(1)

    # 记录参数优化信息
    optimization_info: Optional[Dict[str, Any]] = None

    if args.optimize:
        logger.info("\n" + "=" * 80)
        logger.info("  完整工作流：参数优化 → 回测执行")
        logger.info("=" * 80)

        # 导入并运行优化
        from run_parameter_optimization import main as run_optimization
        try:
            opt_result = run_optimization()

            # 使用优化后的参数回测
            from core.strategy_registry import StrategyLibrary
            lib = StrategyLibrary()

            # 更新策略默认参数
            best_params = opt_result.get("best_params", {})
            applied = False
            if best_params:
                logger.info("\n" + "=" * 60)
                logger.info("  应用优化后的参数")
                logger.info("=" * 60)
                for sname, params in best_params.items():
                    logger.info(f"  {sname}: {params}")
                    lib.update_default_params(sname, params)
                applied = True
            else:
                logger.warning("无有效优化结果，使用默认参数")

            # 记录优化信息
            optimization_info = {
                "executed": True,
                "applied": applied,
                "best_params": best_params,
                "summary_path": Path(opt_result.get("summary_path", "")),
            }

            # 运行回测
            logger.info("\n" + "=" * 80)
            logger.info("  开始回测（使用优化后的参数）")
            logger.info("=" * 80)

        except Exception as e:
            logger.exception(f"优化流程失败: {e}")
            logger.warning("跳过优化，使用默认参数运行回测")
            optimization_info = {"executed": False, "applied": False, "best_params": {}}

    # 运行回测
    run_backtest_with_config(config, optimization_info)

    # 完成提示
    output_dir = Path(config.get("output", {}).get("output_dir", "results"))
    logger.success("=" * 80)
    logger.success("回测完成")
    logger.success(f"输出目录: {output_dir.resolve()}")
    logger.success("=" * 80)


if __name__ == "__main__":
    main()