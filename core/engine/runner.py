"""
组合交易策略系统 - 核心回测引擎。

整合市场环境分类、策略库、策略切换、绩效评估等模块，
提供完整的回测流程。

⚠️ 自研回测引擎标记为 validate_only 模式。
   推荐使用 PyBroker 主引擎（core/engine/broker_adapter.py），
   自研引擎主要用于交叉验证和边缘场景测试。

组合净值计算假设：
  - rebalance_frequency="none"（默认）：无再平衡，各策略权重随净值变化自然漂移，
    组合收益率为各策略日收益率的加权平均。
  - rebalance_frequency="daily"/"weekly"/"monthly"：在指定频率的再平衡日，
    将各策略权重重置为目标权重，并扣除调仓成本（commission_rate + slippage_rate）。

使用方式:
    from core.engine.runner import BacktestRunner

    runner = BacktestRunner(data_dir="./data")
    results = runner.run(
        strategies=["dual_ma", "rsi", "vol_breakout"],
        start_date="2023-01-01",
        end_date="2024-12-31",
    )
    runner.generate_report(results, output_dir="./output")
"""

import os
import logging
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

import pandas as pd
import numpy as np

from core.config import BacktestConfig
from core.data_loader import DataLoader
from core.market_regime import MarketRegimeDetector, MarketRegime
from core.strategy_library import StrategyLibrary
from core.engine.switch_engine import StrategySwitchEngine, SwitchConfig
from core.performance import PerformanceEvaluator, PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """单策略回测结果。"""

    strategy_name: str
    equity_curve: pd.DataFrame  # date, equity
    trades: pd.DataFrame
    metrics: Dict[str, float]
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class PortfolioResult:
    """组合回测结果。"""

    strategy_results: Dict[str, StrategyResult]
    portfolio_equity: pd.DataFrame  # date, equity
    portfolio_metrics: Dict[str, float]
    regime_history: pd.DataFrame
    switch_log: pd.DataFrame
    alerts: List = field(default_factory=list)


class BacktestRunner:
    """
    核心回测运行器。

    整合所有模块，执行完整的回测流程：
    1. 加载数据
    2. 识别市场环境
    3. 运行各策略（含环境自适应信号过滤）
    4. 策略切换决策（含滚动Sharpe计算）
    5. 计算组合净值（支持多种再平衡频率）
    6. 绩效评估与预警
    7. 交叉验证（vs PyBroker）
    """

    def __init__(self, data_dir: str, config: Optional[BacktestConfig] = None):
        self.data_dir = data_dir
        self.config = config or BacktestConfig()

        # 初始化各模块
        self.data_loader = DataLoader(data_dir=data_dir, data_source="csv")
        self.regime_detector = MarketRegimeDetector()
        self.strategy_library = StrategyLibrary()
        self.switch_engine = StrategySwitchEngine(self.strategy_library)
        self.evaluator = PerformanceEvaluator()
        self.monitor = PerformanceMonitor()

        self._data: Optional[pd.DataFrame] = None
        self._regime_data: Optional[pd.DataFrame] = None
        self._last_portfolio_result: Optional[PortfolioResult] = None

    def load_data(self, file_pattern: str = "*.csv") -> pd.DataFrame:
        """加载并预处理数据。"""
        self.data_loader.load_csv_files(file_pattern)
        self.data_loader.build_continuous_series()
        self._data = self.data_loader.get_pybroker_df()
        logger.info(
            f"数据加载完成: {len(self._data)} 行, "
            f"{self._data['symbol'].nunique()} 个合约"
        )
        return self._data

    def detect_regimes(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """识别市场环境。"""
        if df is None:
            df = self._data
        if df is None:
            raise RuntimeError("请先调用 load_data()")

        # 对主力合约数据识别环境
        dominant = (
            df[df["is_dominant"]].copy() if "is_dominant" in df.columns else df.copy()
        )
        dominant = dominant.sort_values("date")

        self._regime_data = self.regime_detector.detect(dominant)
        logger.info(f"环境识别完成: {len(self._regime_data)} 行")
        return self._regime_data

    # ------------------------------------------------------------------
    # 环境自适应交易辅助方法
    # ------------------------------------------------------------------

    def _should_trade(self, regime: Optional[str], strategy_name: str) -> bool:
        """
        判断给定策略在当前市场环境下是否应该交易。

        环境自适应规则：
          - dual_ma: RANGE_BOUND（震荡市）时趋势策略容易反复穿越均线，强制不交易。
          - rsi: RANGE_BOUND 时允许交易（超买超卖在震荡市更有效）。
          - vol_breakout: HIGH_VOLATILITY（高波动）时允许，LOW_VOLATILITY 时限制。
          - term_structure: 对环境不敏感，始终允许。

        Args:
            regime: 市场环境标签字符串（如 "RANGE_BOUND"），None 时默认允许。
            strategy_name: 策略名称。

        Returns:
            是否应生成交易信号。
        """
        if regime is None:
            return True  # 无环境信息时不做过滤

        regime_upper = (
            regime.upper() if isinstance(regime, str) else str(regime).upper()
        )

        if strategy_name == "dual_ma":
            # 震荡市趋势策略效果差 → 禁止交易
            if regime_upper in ("RANGE_BOUND",):
                return False
            return True

        elif strategy_name == "vol_breakout":
            # 极低波动时突破信号不可靠 → 禁止交易
            if regime_upper in ("LOW_VOLATILITY",):
                return False
            return True

        # 其他策略（rsi, term_structure, spread）不做环境过滤
        return True

    # ------------------------------------------------------------------
    # 策略信号生成
    # ------------------------------------------------------------------

    def run_strategy(
        self, strategy_name: str, df: pd.DataFrame, params: Optional[Dict] = None
    ) -> StrategyResult:
        """
        运行单个策略的回测。

        使用简化的回测引擎（不依赖PyBroker的实时执行），
        直接在DataFrame上模拟交易。
        """
        profile = self.strategy_library.get_profile(strategy_name)
        if profile is None:
            raise ValueError(f"未知策略: {strategy_name}")

        # 合并默认参数和自定义参数
        strategy_params = {**profile.default_params, **(params or {})}

        # 获取环境数据
        regime_df = self._regime_data
        if regime_df is None:
            regime_df = self.detect_regimes(df)

        # 简化回测：基于信号生成交易
        data = df.copy()
        if "date" in data.columns:
            data = data.sort_values("date").reset_index(drop=True)

        # 生成交易信号（含环境自适应过滤）
        signals = self._generate_signals(
            strategy_name, data, strategy_params, regime_df
        )

        # 模拟交易执行
        trades, equity = self._simulate_trading(signals, data, strategy_params)

        # 计算指标
        metrics = self.evaluator.compute_metrics(equity, trades)

        # 计算各环境下的表现
        regime_perf = self._compute_regime_performance(
            equity, trades, regime_df, strategy_name
        )

        # 更新策略库性能数据
        for regime_val, perf in regime_perf.items():
            try:
                regime_enum = MarketRegime(regime_val)
                self.strategy_library.update_performance(
                    strategy_name, regime_enum, perf
                )
            except ValueError:
                pass

        # 绩效预警
        last_date = str(data["date"].iloc[-1]) if "date" in data.columns else ""
        alerts = self.monitor.evaluate(strategy_name, metrics, last_date)

        return StrategyResult(
            strategy_name=strategy_name,
            equity_curve=pd.DataFrame(
                {"date": data["date"].values, "equity": equity.values}
            ),
            trades=trades,
            metrics=metrics,
            regime_performance=regime_perf,
        )

    def _generate_signals(
        self,
        strategy_name: str,
        data: pd.DataFrame,
        params: Dict,
        regime_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        生成交易信号。

        包含环境自适应逻辑：
          - 先计算原始信号
          - 再通过 _should_trade() 过滤：在不适合交易的环境中将信号清零

        Args:
            strategy_name: 策略名称。
            data: 行情数据（含 date, close, high, low）。
            params: 策略参数。
            regime_df: 市场环境数据（含 date, regime 列），用于环境自适应过滤。

        Returns:
            含 signal 列（0/1/-1）的 DataFrame。
        """
        result = data[["date", "close", "high", "low"]].copy()
        result["signal"] = 0  # 0=无信号, 1=做多, -1=做空

        close = data["close"]
        high = data["high"]
        low = data["low"]

        # ── 步骤1：计算原始信号 ──
        if strategy_name == "dual_ma":
            short_ma = close.rolling(window=params.get("short_ma", 5)).mean()
            long_ma = close.rolling(window=params.get("long_ma", 20)).mean()
            result["signal"] = np.where(
                short_ma > long_ma, 1, np.where(short_ma < long_ma, -1, 0)
            )

        elif strategy_name == "rsi":
            delta = close.diff()
            rsi_period = params.get("rsi_period", 14)
            gain = (
                delta.where(delta > 0, 0.0)
                .ewm(alpha=1 / rsi_period, adjust=False)
                .mean()
            )
            loss = (
                (-delta.where(delta < 0, 0.0))
                .ewm(alpha=1 / rsi_period, adjust=False)
                .mean()
            )
            rs = np.where(loss > 0, gain / loss, 100.0)
            rsi = 100 - 100 / (1 + rs)
            oversold = params.get("oversold", 30.0)
            overbought = params.get("overbought", 70.0)
            result["signal"] = np.where(
                rsi < oversold, 1, np.where(rsi > overbought, -1, 0)
            )

        elif strategy_name == "term_structure":
            lookback = params.get("lookback", 20)
            entry_th = params.get("entry_threshold", 8.0)
            exit_th = params.get("exit_threshold", 0.5)
            sma = close.rolling(window=lookback).mean()
            spread = (close - sma) / sma * 100
            result["signal"] = np.where(
                spread < -entry_th,
                1,
                np.where(
                    spread > entry_th, -1, np.where(spread.abs() < exit_th, 0, None)
                ),
            )
            result["signal"] = (
                result["signal"].ffill().fillna(0).infer_objects(copy=False)
            )

        elif strategy_name == "vol_breakout":
            atr_period = params.get("atr_period", 26)
            band_period = params.get("band_period", 30)
            mult = params.get("atr_multiplier", 2.0)
            tr = pd.concat(
                [
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = tr.rolling(window=atr_period).mean()
            center = close.rolling(window=band_period).mean()
            upper = center + mult * atr
            lower = center - mult * atr
            result["signal"] = np.where(
                close > upper,
                1,
                np.where(
                    close < lower,
                    -1,
                    np.where(
                        (close > lower) & (close < upper) & (close > center), 0, None
                    ),
                ),
            )
            result["signal"] = (
                result["signal"].ffill().fillna(0).infer_objects(copy=False)
            )

        elif strategy_name == "spread":
            result["signal"] = 0  # 跨期套利需要多合约数据，简化处理

        # ── 步骤2：环境自适应过滤 ──
        # 利用 regime_df 中的环境标签，对不适合交易的环境将信号清零
        if (
            regime_df is not None
            and not regime_df.empty
            and "regime" in regime_df.columns
            and "date" in regime_df.columns
        ):
            # 构建 date → regime 映射
            regime_map = dict(zip(regime_df["date"], regime_df["regime"]))
            # 按行遍历（信号行数通常不多，可接受）
            for idx in result.index:
                date_val = result.loc[idx, "date"]
                if date_val in regime_map:
                    if not self._should_trade(regime_map[date_val], strategy_name):
                        result.loc[idx, "signal"] = 0

        return result

    def _simulate_trading(
        self, signals: pd.DataFrame, data: pd.DataFrame, params: Dict
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        模拟交易执行（v3 修复版）。

        修复项：
          - 空头/多头止损成本统一：P&L 不含成本，成本独立通过现金变动扣除。
          - 集成 StrategySwitchEngine：每日调用 decide()，传入实时滚动 Sharpe。
          - 添加滑点和佣金扣除。

        盈亏计算原则：
          - 基础 P&L = shares × (exit_price - entry_price)，多/空方向由符号处理
          - 交易成本独立通过 cash 变动体现
          - 记录中的 pnl 为净盈亏（基础P&L - 平仓成本）
        """
        position_size = params.get("position_size", 0.2)
        stop_loss_pct = params.get("trailing_stop_pct", 0.03) or 0.03
        cfg = self.config
        commission = cfg.commission_rate
        slippage = cfg.slippage_rate
        cost_rate = commission + slippage

        cash = cfg.initial_cash
        position = 0  # 1=多头, -1=空头, 0=空仓
        entry_price = 0.0
        shares = 0
        equity_list = []
        trade_records = []

        # ── 滚动 Sharpe 跟踪（用于策略切换引擎） ──
        switch_cfg = SwitchConfig()
        lookback = switch_cfg.performance_lookback  # 默认20
        daily_returns: deque = deque(maxlen=lookback)
        prev_equity = cash  # 前一日权益

        # ── 策略切换引擎状态 ──
        active_strategy = None  # 由 switch_engine 管理
        switch_engine = self.switch_engine
        regime_df = self._regime_data
        has_regime = (
            regime_df is not None
            and not regime_df.empty
            and "regime" in regime_df.columns
            and "regime_confidence" in regime_df.columns
        )

        for i in range(len(signals)):
            close = signals["close"].iloc[i]
            date = signals["date"].iloc[i]
            signal = signals["signal"].iloc[i]

            # ── 计算当前权益（修复空头公式） ──
            if position == 1:
                equity = cash + shares * close
            elif position == -1:
                # 空头：equity = cash - shares * close
                # （cash 已包含卖出所得 proceeds，买入需要 sh*close 成本）
                equity = cash - shares * close
            else:
                equity = cash

            # ── 计算滚动 Sharpe（v3 新增） ──
            daily_return = 0.0
            if i > 0:
                daily_return = (equity / prev_equity) - 1
            daily_returns.append(daily_return)
            prev_equity = equity

            # 当队列满时计算年化 Sharpe
            current_sharpe = 0.0
            if len(daily_returns) >= lookback:
                ret_list = list(daily_returns)
                mean_ret = np.mean(ret_list)
                std_ret = np.std(ret_list, ddof=1)
                if std_ret > 1e-10:
                    current_sharpe = (mean_ret / std_ret) * np.sqrt(252)

            # ── 策略切换引擎集成 ──
            # 每日调用 decide()，传入实时滚动 Sharpe
            if has_regime and i < len(regime_df):
                r_row = regime_df.iloc[i]
                regime_str = r_row.get("regime")
                regime_conf = r_row.get("regime_confidence", 0.5)
                if regime_str is not None:
                    try:
                        regime = MarketRegime(regime_str)
                        switch_decision = switch_engine.decide(
                            current_date=str(date),
                            current_regime=regime,
                            regime_confidence=float(regime_conf),
                            current_sharpe=current_sharpe,
                            position_value=(shares * close if position != 0 else 0.0),
                            has_position=(position != 0),
                            sharpe_samples=len(daily_returns),
                            trading_day_index=i,
                        )
                        if switch_decision and switch_decision.approved:
                            active_strategy = switch_decision.to_strategy
                            logger.info(
                                "策略切换: %s → %s (原因: %s)",
                                switch_decision.from_strategy,
                                switch_decision.to_strategy,
                                switch_decision.reason.value,
                            )
                    except Exception:
                        pass  # 切换引擎评估失败不影响主流程

            # ── 止损检查 ──
            # v3 修复：统一 P&L 计算模式
            #   基础 P&L 不含成本，成本通过 cash 变动独立体现
            if position == 1 and close < entry_price * (1 - stop_loss_pct):
                # 多头止损
                base_pnl = shares * (close - entry_price)
                exit_cost = shares * close * cost_rate
                net_pnl = base_pnl - exit_cost
                trade_records.append(
                    {
                        "date": date,
                        "side": "long_stop_loss",
                        "entry": entry_price,
                        "exit": close,
                        "shares": shares,
                        "pnl": net_pnl,
                    }
                )
                # 平仓现金流入（扣除成本）
                cash += shares * close * (1 - cost_rate)
                position = 0
                shares = 0

            elif position == -1 and close > entry_price * (1 + stop_loss_pct):
                # 空头止损
                base_pnl = shares * (entry_price - close)
                exit_cost = shares * close * cost_rate
                net_pnl = base_pnl - exit_cost
                trade_records.append(
                    {
                        "date": date,
                        "side": "short_stop_loss",
                        "entry": entry_price,
                        "exit": close,
                        "shares": shares,
                        "pnl": net_pnl,
                    }
                )
                # 空头平仓：买回股票（扣除成本）
                cash -= shares * close * (1 + cost_rate)
                position = 0
                shares = 0

            # ── 信号执行 ──
            if signal == 1 and position != 1:
                # 先平空头（如有）
                if position == -1:
                    base_pnl = shares * (entry_price - close)
                    exit_cost = shares * close * cost_rate
                    net_pnl = base_pnl - exit_cost
                    trade_records.append(
                        {
                            "date": date,
                            "side": "short_close",
                            "entry": entry_price,
                            "exit": close,
                            "shares": shares,
                            "pnl": net_pnl,
                        }
                    )
                    cash -= shares * close * (1 + cost_rate)
                    position = 0
                    shares = 0

                # 开多头
                alloc = equity * position_size
                shares = int(alloc / close) if close > 0 else 0
                if shares > 0:
                    # 开仓成本：cash -= shares * close * (1 + cost_rate)
                    cash -= shares * close * (1 + cost_rate)
                    entry_price = close
                    position = 1

            elif signal == -1 and position != -1:
                # 先平多头（如有）
                if position == 1:
                    base_pnl = shares * (close - entry_price)
                    exit_cost = shares * close * cost_rate
                    net_pnl = base_pnl - exit_cost
                    trade_records.append(
                        {
                            "date": date,
                            "side": "long_close",
                            "entry": entry_price,
                            "exit": close,
                            "shares": shares,
                            "pnl": net_pnl,
                        }
                    )
                    cash += shares * close * (1 - cost_rate)
                    position = 0
                    shares = 0

                # 开空头
                alloc = equity * position_size
                shares = int(alloc / close) if close > 0 else 0
                if shares > 0:
                    # 空头开仓：cash += shares * close（卖出所得）减成本
                    cash += shares * close * (1 - cost_rate)
                    entry_price = close
                    position = -1

            # ── 计算期末权益 ──
            if position == 1:
                equity = cash + shares * close
            elif position == -1:
                equity = cash - shares * close
            else:
                equity = cash

            equity_list.append(equity)

        equity_series = pd.Series(equity_list, index=signals.index)
        trades_df = (
            pd.DataFrame(trade_records)
            if trade_records
            else pd.DataFrame(
                columns=["date", "side", "entry", "exit", "shares", "pnl"]
            )
        )

        return trades_df, equity_series

    def _compute_regime_performance(
        self,
        equity: pd.Series,
        trades: pd.DataFrame,
        regime_df: pd.DataFrame,
        strategy_name: str,
    ) -> Dict[str, Dict[str, float]]:
        """计算各市场环境下的策略表现（使用 pd.merge 对齐日期）。"""
        if regime_df is None or regime_df.empty:
            return {}

        regime_perf = {}
        if "regime" in regime_df.columns and "date" in regime_df.columns:
            # 将 equity index 转为日期列，通过 merge 对齐
            eq_df = equity.reset_index()
            eq_df.columns = ["row_idx", "equity"]
            # 重建日期映射：用 signals 的日期（run_strategy 中 data["date"]）
            # 这里用 regime_df 自带的日期 + equity 位置
            if len(eq_df) == len(regime_df):
                eq_df["date"] = regime_df["date"].values

            for regime_val in regime_df["regime"].unique():
                regime_dates = set(
                    regime_df[regime_df["regime"] == regime_val]["date"].values
                )
                regime_eq = eq_df[eq_df["date"].isin(regime_dates)]["equity"]
                if len(regime_eq) > 10:
                    perf = self.evaluator.compute_metrics(regime_eq)
                    regime_perf[str(regime_val)] = perf

        return regime_perf

    def run(
        self,
        strategies: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        params: Optional[Dict[str, Dict]] = None,
    ) -> PortfolioResult:
        """
        运行完整的组合回测。

        Args:
            strategies: 策略名称列表，None则使用全部
            start_date: 开始日期
            end_date: 结束日期
            params: 各策略的自定义参数

        Returns:
            组合回测结果（同时保存到 _last_portfolio_result）
        """
        warnings.warn(
            "Using legacy backtest engine (validate_only mode). "
            "PyBroker is recommended as the primary engine. "
            "See core/broker_adapter.py for usage.",
            FutureWarning,
            stacklevel=2,
        )

        # 加载数据
        if self._data is None:
            self.load_data()

        df = self._data.copy()

        # 日期过滤
        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]

        # 识别市场环境
        regime_df = self.detect_regimes(df)

        # 确定策略列表
        if strategies is None:
            strategies = [s.name for s in self.strategy_library.list_all()]

        # 运行各策略
        strategy_results = {}
        for name in strategies:
            logger.info(f"运行策略: {name}")
            result = self.run_strategy(name, df, (params or {}).get(name))
            strategy_results[name] = result

        # 计算组合净值
        weights = self.config.strategy_weights
        if not weights:
            weights = {name: 1.0 / len(strategies) for name in strategies}

        portfolio_equity = self._compute_portfolio_equity(strategy_results, weights)

        # 计算组合指标
        portfolio_metrics = self.evaluator.compute_metrics(portfolio_equity["equity"])

        # 策略切换日志
        switch_log = self.switch_engine.get_decision_summary()

        # 预警
        all_alerts = []
        for name, result in strategy_results.items():
            alerts = self.monitor.evaluate(
                name, result.metrics, str(df["date"].iloc[-1]) if len(df) > 0 else ""
            )
            all_alerts.extend(alerts)

        result = PortfolioResult(
            strategy_results=strategy_results,
            portfolio_equity=portfolio_equity,
            portfolio_metrics=portfolio_metrics,
            regime_history=regime_df,
            switch_log=switch_log,
            alerts=all_alerts,
        )

        # 保存为最后一次结果，供 cross_validate_with_pybroker 使用
        self._last_portfolio_result = result

        return result

    def _get_rebalance_dates(self, dates: List, frequency: str) -> set:
        """
        根据再平衡频率确定需要再平衡的日期集合。

        Args:
            dates: 全部交易日列表（已排序）。
            frequency: "daily", "weekly", "monthly"。

        Returns:
            需要再平衡的日期集合。
        """
        if frequency == "daily":
            return set(dates)
        elif frequency == "weekly":
            # 每周最后一个交易日
            result = set()
            for i, d in enumerate(dates):
                d_ts = pd.Timestamp(d)
                next_d = dates[i + 1] if i + 1 < len(dates) else None
                if next_d is None:
                    result.add(d)
                else:
                    next_ts = pd.Timestamp(next_d)
                    if d_ts.isocalendar()[1] != next_ts.isocalendar()[1]:
                        result.add(d)
            return result
        elif frequency == "monthly":
            # 每月最后一个交易日
            result = set()
            for i, d in enumerate(dates):
                d_ts = pd.Timestamp(d)
                next_d = dates[i + 1] if i + 1 < len(dates) else None
                if next_d is None:
                    result.add(d)
                else:
                    next_ts = pd.Timestamp(next_d)
                    if d_ts.month != next_ts.month:
                        result.add(d)
            return result
        else:
            return set()

    def _compute_portfolio_equity(
        self, strategy_results: Dict[str, StrategyResult], weights: Dict[str, float]
    ) -> pd.DataFrame:
        """
        计算组合净值。

        再平衡假设：
          - rebalance_frequency="none"（默认）：收益率加权平均，
            各策略权重随净值变化自然漂移，组合净值 = 各策略净值按初始权重加权累加。
          - rebalance_frequency="daily"/"weekly"/"monthly"：
            在再平衡日将各策略现金重置为目标权重，扣除调仓成本。

        Args:
            strategy_results: 各策略的结果。
            weights: 目标权重字典 {strategy_name: weight}。

        Returns:
            含 date + equity 列的 DataFrame。
        """
        freq = self.config.rebalance_frequency

        # 收集所有日期和策略净值
        eq_curves: Dict[str, pd.Series] = {}
        all_dates = set()
        for name, result in strategy_results.items():
            if result.equity_curve.empty:
                continue
            eq = result.equity_curve.sort_values("date")
            eq_curves[name] = pd.Series(eq["equity"].values, index=eq["date"])
            all_dates.update(eq["date"].tolist())

        all_dates = sorted(all_dates)
        if not all_dates:
            return pd.DataFrame(columns=["date", "equity"])

        rebalance_dates = set()
        cost_rate = 0.0
        if freq != "none":
            rebalance_dates = self._get_rebalance_dates(all_dates, freq)
            cost_rate = self.config.commission_rate + self.config.slippage_rate

        # 计算各策略日收益率（用于 "none" 模式）
        strategy_returns: Dict[str, Dict] = {}
        if freq == "none":
            for name, eq_ser in eq_curves.items():
                daily_ret = eq_ser.pct_change().fillna(0)
                strategy_returns[name] = dict(zip(eq_ser.index, daily_ret))

        # ── 模式 "none"：收益率加权平均（无再平衡） ──
        if freq == "none":
            portfolio_data = []
            current_equity = self.config.initial_cash
            for i, date in enumerate(all_dates):
                daily_return = 0.0
                for name, weight in weights.items():
                    if name in strategy_returns and date in strategy_returns[name]:
                        daily_return += strategy_returns[name][date] * weight
                if i > 0:
                    current_equity = current_equity * (1 + daily_return)
                portfolio_data.append({"date": date, "equity": current_equity})
            return pd.DataFrame(portfolio_data)

        # ── 有再平衡模式 ──
        # 初始化：每个策略分配 initial_cash * weight
        strategy_cash: Dict[str, float] = {}
        strategy_shares_pct: Dict[str, float] = {}
        total_cash = self.config.initial_cash
        for name in eq_curves:
            alloc = self.config.initial_cash * weights.get(name, 0)
            strategy_cash[name] = alloc
            # 用第一天净值反推初始"份额比例"（虚拟份额）
            first_eq = eq_curves[name].iloc[0]
            strategy_shares_pct[name] = alloc / first_eq if first_eq > 0 else 0

        portfolio_data = []
        prev_date_idx = -1

        for date_idx, date in enumerate(all_dates):
            # 计算组合总权益
            total_equity = sum(
                strategy_shares_pct.get(name, 0) * eq_curves[name].get(date, 0)
                for name in eq_curves
            )

            # 再平衡日：调仓
            if date in rebalance_dates and date_idx > 0:
                total_equity_before = total_equity
                turnover = 0.0  # 累计调仓金额
                for name in eq_curves:
                    target_alloc = total_equity * weights.get(name, 0)
                    current_alloc = strategy_shares_pct.get(name, 0) * eq_curves[
                        name
                    ].get(date, 0)
                    diff = abs(target_alloc - current_alloc)
                    turnover += diff
                    # 重置份额
                    eq_val = eq_curves[name].get(date, 0)
                    strategy_shares_pct[name] = (
                        target_alloc / eq_val if eq_val > 0 else 0
                    )

                # 扣除再平衡成本
                total_equity = total_equity_before - turnover * cost_rate

            portfolio_data.append({"date": date, "equity": total_equity})

        return pd.DataFrame(portfolio_data)

    def generate_report(
        self, result: PortfolioResult, output_dir: str = "./output"
    ) -> str:
        """生成回测报告。"""
        os.makedirs(output_dir, exist_ok=True)

        lines = [
            "# 组合交易策略系统 - 回测报告",
            f"",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## 组合绩效",
            f"",
            f"| 指标 | 值 |",
            f"|------|-----|",
        ]

        for key, val in result.portfolio_metrics.items():
            if isinstance(val, float):
                lines.append(f"| {key} | {val:.4f} |")
            else:
                lines.append(f"| {key} | {val} |")

        # 各策略表现
        lines.extend(["", "## 各策略表现", ""])
        for name, sr in result.strategy_results.items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            for key, val in sr.metrics.items():
                if isinstance(val, float):
                    lines.append(f"| {key} | {val:.4f} |")
                else:
                    lines.append(f"| {key} | {val} |")
            lines.append("")

        # 环境分析
        if (
            not result.regime_history.empty
            and "regime" in result.regime_history.columns
        ):
            lines.extend(["", "## 市场环境分析", ""])
            regime_counts = result.regime_history["regime"].value_counts()
            lines.append("| 环境类型 | 出现次数 | 占比 |")
            lines.append("|----------|----------|------|")
            total = len(result.regime_history)
            for regime, count in regime_counts.items():
                lines.append(f"| {regime} | {count} | {count / total * 100:.1f}% |")

        # 预警信息
        if result.alerts:
            lines.extend(["", "## 预警信息", ""])
            for alert in result.alerts[-10:]:
                lines.append(f"- [{alert.level.value}] {alert.message}")

        # 策略切换日志
        if not result.switch_log.empty:
            lines.extend(["", "## 策略切换记录", ""])
            lines.append(result.switch_log.to_markdown(index=False))

        report_text = "\n".join(lines)

        # 保存报告
        report_path = os.path.join(output_dir, "report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # 保存组合净值
        equity_path = os.path.join(output_dir, "portfolio_equity.csv")
        result.portfolio_equity.to_csv(equity_path, index=False)

        # 保存各策略净值
        for name, sr in result.strategy_results.items():
            if not sr.equity_curve.empty:
                path = os.path.join(output_dir, f"equity_{name}.csv")
                sr.equity_curve.to_csv(path, index=False)

        logger.info(f"报告已保存到: {output_dir}")
        return report_path

    # ----------------------------------------------------------------
    # 交叉验证（自研引擎 vs PyBroker）
    # ----------------------------------------------------------------

    def cross_validate_with_pybroker(
        self,
        pybroker_result,
        own_result=None,
    ) -> Dict:
        """
        交叉验证：比较自研引擎与 PyBroker 的净值曲线差异。

        自动从最近一次 run() 结果中提取组合净值进行对比。

        Args:
            pybroker_result: PyBroker 回测结果对象，需包含 equity_curve (DataFrame,
                含 date + equity 列) 或直接传入 pd.DataFrame。
            own_result: 自研引擎结果（PortfolioResult），None 则自动使用
                _last_portfolio_result。

        Returns:
            包含以下键的字典：
              - correlation: 归一化净值相关系数
              - max_abs_diff: 最大绝对差异
              - mean_abs_diff: 平均绝对差异
              - max_diff_pct: 最大百分比差异
              - max_diff_date: 最大偏离日期
              - returns_correlation: 日收益率相关系数
              - final_pybroker_eq: PyBroker 归一化终值
              - final_legacy_eq: 自研引擎归一化终值
              - n_samples: 对齐后样本数
              - dates_range: 日期范围
        """
        import pandas as pd

        # ── 提取 PyBroker 净值 ──
        if isinstance(pybroker_result, pd.DataFrame):
            pybroker_eq = pybroker_result.copy()
        elif hasattr(pybroker_result, "equity_curve"):
            pybroker_eq = pybroker_result.equity_curve.copy()
        else:
            raise ValueError(
                "pybroker_result 必须为 pd.DataFrame 或包含 equity_curve 属性的对象"
            )

        # ── 提取自研引擎净值 ──
        if own_result is not None:
            portfolio_equity = own_result.portfolio_equity
        elif self._last_portfolio_result is not None:
            portfolio_equity = self._last_portfolio_result.portfolio_equity
        else:
            raise ValueError(
                "无可用自研引擎结果。请先调用 run() 或传入 own_result 参数。"
            )

        legacy_eq = portfolio_equity[["date", "equity"]].copy()

        # ── 统一列名 ──
        pybroker_eq = pybroker_eq.rename(
            columns={"date": "date", "equity": "pybroker_equity"}
        )
        legacy_eq = legacy_eq.rename(
            columns={"date": "date", "equity": "legacy_equity"}
        )

        # ── 通过 date 合并对齐 ──
        merged = pd.merge(
            pybroker_eq[["date", "pybroker_equity"]],
            legacy_eq[["date", "legacy_equity"]],
            on="date",
            how="inner",
        )
        if len(merged) < 10:
            return {"error": "样本太少，无法交叉验证", "n_samples": len(merged)}

        # ── 归一化到同一初始值 ──
        merged["pybroker_eq"] = (
            merged["pybroker_equity"] / merged["pybroker_equity"].iloc[0]
        )
        merged["legacy_eq"] = merged["legacy_equity"] / merged["legacy_equity"].iloc[0]

        # ── 差异统计 ──
        diff = (merged["pybroker_eq"] - merged["legacy_eq"]).abs()
        correlation = merged["pybroker_eq"].corr(merged["legacy_eq"])

        # 日收益率相关系数
        merged["pybroker_ret"] = merged["pybroker_eq"].pct_change().fillna(0)
        merged["legacy_ret"] = merged["legacy_eq"].pct_change().fillna(0)
        returns_corr = merged["pybroker_ret"].corr(merged["legacy_ret"])

        # 最大偏离日期
        max_diff_idx = diff.idxmax()
        max_diff_date = (
            merged.loc[max_diff_idx, "date"] if max_diff_idx in merged.index else None
        )

        return {
            "correlation": round(correlation, 4),
            "max_abs_diff": round(diff.max(), 6),
            "mean_abs_diff": round(diff.mean(), 6),
            "max_diff_pct": round((diff / merged["pybroker_eq"]).max() * 100, 4),
            "max_diff_date": str(max_diff_date) if max_diff_date is not None else None,
            "returns_correlation": round(returns_corr, 4),
            "final_pybroker_eq": round(merged["pybroker_eq"].iloc[-1], 4),
            "final_legacy_eq": round(merged["legacy_eq"].iloc[-1], 4),
            "n_samples": len(merged),
            "dates_range": f"{merged['date'].iloc[0]} ~ {merged['date'].iloc[-1]}",
        }
