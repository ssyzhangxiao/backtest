"""
组合交易策略系统 - 核心回测引擎。

整合市场环境分类、策略库、策略切换、绩效评估等模块，
提供完整的回测流程。

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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import numpy as np

from core.data_loader import DataLoader
from core.market_regime import MarketRegimeDetector, MarketRegime
from core.strategy_library import StrategyLibrary
from core.engine.switch_engine import StrategySwitchEngine
from core.performance import PerformanceEvaluator, PerformanceMonitor

logger = logging.getLogger(__name__)

INITIAL_CASH = 1_000_000


@dataclass
class BacktestConfig:
    """回测配置。"""
    initial_cash: float = INITIAL_CASH
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0002

    # 样本内/外分割
    in_sample_end: Optional[str] = None

    # 策略权重
    strategy_weights: Dict[str, float] = field(default_factory=dict)

    # 风控
    stop_loss_pct: float = 0.05
    max_position_pct: float = 0.2
    max_total_position_pct: float = 0.6


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
    3. 运行各策略
    4. 策略切换决策
    5. 计算组合净值
    6. 绩效评估与预警
    """

    def __init__(self, data_dir: str, config: Optional[BacktestConfig] = None):
        self.data_dir = data_dir
        self.config = config or BacktestConfig()

        # 初始化各模块
        self.data_loader = DataLoader(data_dir)
        self.regime_detector = MarketRegimeDetector()
        self.strategy_library = StrategyLibrary()
        self.switch_engine = StrategySwitchEngine(self.strategy_library)
        self.evaluator = PerformanceEvaluator()
        self.monitor = PerformanceMonitor()

        self._data: Optional[pd.DataFrame] = None
        self._regime_data: Optional[pd.DataFrame] = None

    def load_data(self, file_pattern: str = "*.csv") -> pd.DataFrame:
        """加载并预处理数据。"""
        self.data_loader.load_csv_files(file_pattern)
        self.data_loader.build_continuous_series()
        self._data = self.data_loader.get_pybroker_df()
        logger.info(f"数据加载完成: {len(self._data)} 行, "
                     f"{self._data['symbol'].nunique()} 个合约")
        return self._data

    def detect_regimes(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """识别市场环境。"""
        if df is None:
            df = self._data
        if df is None:
            raise RuntimeError("请先调用 load_data()")

        # 对主力合约数据识别环境
        dominant = df[df["is_dominant"]].copy() if "is_dominant" in df.columns else df.copy()
        dominant = dominant.sort_values("date")

        self._regime_data = self.regime_detector.detect(dominant)
        logger.info(f"环境识别完成: {len(self._regime_data)} 行")
        return self._regime_data

    def run_strategy(self, strategy_name: str, df: pd.DataFrame,
                     params: Optional[Dict] = None) -> StrategyResult:
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

        # 生成交易信号
        signals = self._generate_signals(strategy_name, data, strategy_params, regime_df)

        # 模拟交易执行
        trades, equity = self._simulate_trading(
            signals, data, strategy_params
        )

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
                self.strategy_library.update_performance(strategy_name, regime_enum, perf)
            except ValueError:
                pass

        # 绩效预警
        last_date = str(data["date"].iloc[-1]) if "date" in data.columns else ""
        alerts = self.monitor.evaluate(strategy_name, metrics, last_date)

        return StrategyResult(
            strategy_name=strategy_name,
            equity_curve=pd.DataFrame({"date": data["date"].values, "equity": equity.values}),
            trades=trades,
            metrics=metrics,
            regime_performance=regime_perf,
        )

    def _generate_signals(self, strategy_name: str, data: pd.DataFrame,
                          params: Dict, regime_df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号。"""
        result = data[["date", "close", "high", "low"]].copy()
        result["signal"] = 0  # 0=无信号, 1=做多, -1=做空

        close = data["close"]
        high = data["high"]
        low = data["low"]

        if strategy_name == "dual_ma":
            short_ma = close.rolling(window=params.get("short_ma", 5)).mean()
            long_ma = close.rolling(window=params.get("long_ma", 20)).mean()
            result["signal"] = np.where(short_ma > long_ma, 1,
                                        np.where(short_ma < long_ma, -1, 0))

        elif strategy_name == "rsi":
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=params.get("rsi_period", 14)).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=params.get("rsi_period", 14)).mean()
            rs = np.where(loss > 0, gain / loss, 100.0)
            rsi = 100 - 100 / (1 + rs)
            oversold = params.get("oversold", 30.0)
            overbought = params.get("overbought", 70.0)
            result["signal"] = np.where(rsi < oversold, 1,
                                        np.where(rsi > overbought, -1, 0))

        elif strategy_name == "term_structure":
            lookback = params.get("lookback", 20)
            entry_th = params.get("entry_threshold", 8.0)
            exit_th = params.get("exit_threshold", 0.5)
            sma = close.rolling(window=lookback).mean()
            spread = (close - sma) / sma * 100
            result["signal"] = np.where(spread < -entry_th, 1,
                                        np.where(spread > entry_th, -1,
                                                 np.where(spread.abs() < exit_th, 0, None)))
            result["signal"] = result["signal"].ffill().fillna(0).infer_objects(copy=False)

        elif strategy_name == "vol_breakout":
            atr_period = params.get("atr_period", 26)
            band_period = params.get("band_period", 30)
            mult = params.get("atr_multiplier", 2.0)
            tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                            (low - close.shift(1)).abs()], axis=1).max(axis=1)
            atr = tr.rolling(window=atr_period).mean()
            center = close.rolling(window=band_period).mean()
            upper = center + mult * atr
            lower = center - mult * atr
            result["signal"] = np.where(close > upper, 1,
                                        np.where(close < lower, -1,
                                                 np.where((close > lower) & (close < upper) &
                                                          (close > center), 0, None)))
            result["signal"] = result["signal"].ffill().fillna(0).infer_objects(copy=False)

        elif strategy_name == "spread":
            result["signal"] = 0  # 跨期套利需要多合约数据，简化处理

        return result

    def _simulate_trading(self, signals: pd.DataFrame, data: pd.DataFrame,
                          params: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        """模拟交易执行。"""
        position_size = params.get("position_size", 0.2)
        stop_loss_pct = params.get("trailing_stop_pct", 0.03) or 0.03

        cash = self.config.initial_cash
        position = 0  # 1=多头, -1=空头, 0=空仓
        entry_price = 0.0
        shares = 0
        equity_list = []
        trade_records = []

        for i in range(len(signals)):
            close = signals["close"].iloc[i]
            date = signals["date"].iloc[i]
            signal = signals["signal"].iloc[i]

            # 计算当前权益
            if position == 1:
                market_value = shares * close
            elif position == -1:
                market_value = shares * (2 * entry_price - close)
            else:
                market_value = cash

            current_equity = cash + (market_value - cash if position != 0 else 0)

            # 止损检查
            if position == 1 and close < entry_price * (1 - stop_loss_pct):
                pnl = shares * (close - entry_price)
                trade_records.append({
                    "date": date, "side": "long", "entry": entry_price,
                    "exit": close, "shares": shares, "pnl": pnl,
                })
                cash += shares * close
                position = 0
                shares = 0
            elif position == -1 and close > entry_price * (1 + stop_loss_pct):
                pnl = shares * (entry_price - close)
                trade_records.append({
                    "date": date, "side": "short", "entry": entry_price,
                    "exit": close, "shares": shares, "pnl": pnl,
                })
                cash += shares * (2 * entry_price - close)
                position = 0
                shares = 0

            # 信号执行
            if signal == 1 and position != 1:
                if position == -1:
                    pnl = shares * (entry_price - close)
                    trade_records.append({
                        "date": date, "side": "short_close", "entry": entry_price,
                        "exit": close, "shares": shares, "pnl": pnl,
                    })
                    cash += shares * (2 * entry_price - close)
                alloc = current_equity * position_size
                shares = int(alloc / close) if close > 0 else 0
                if shares > 0:
                    cash -= shares * close
                    entry_price = close
                    position = 1

            elif signal == -1 and position != -1:
                if position == 1:
                    pnl = shares * (close - entry_price)
                    trade_records.append({
                        "date": date, "side": "long_close", "entry": entry_price,
                        "exit": close, "shares": shares, "pnl": pnl,
                    })
                    cash += shares * close
                alloc = current_equity * position_size
                shares = int(alloc / close) if close > 0 else 0
                if shares > 0:
                    cash += shares * close
                    entry_price = close
                    position = -1

            # 计算期末权益
            if position == 1:
                equity = cash + shares * close
            elif position == -1:
                equity = cash + shares * (2 * entry_price - close)
            else:
                equity = cash

            equity_list.append(equity)

        equity_series = pd.Series(equity_list, index=signals.index)
        trades_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame(
            columns=["date", "side", "entry", "exit", "shares", "pnl"]
        )

        return trades_df, equity_series

    def _compute_regime_performance(self, equity: pd.Series, trades: pd.DataFrame,
                                     regime_df: pd.DataFrame,
                                     strategy_name: str) -> Dict[str, Dict[str, float]]:
        """计算各市场环境下的策略表现。"""
        if regime_df is None or regime_df.empty:
            return {}

        regime_perf = {}
        if "regime" in regime_df.columns and "date" in regime_df.columns:
            for regime_val in regime_df["regime"].unique():
                regime_dates = set(
                    regime_df[regime_df["regime"] == regime_val]["date"].tolist()
                )
                regime_indices = [i for i, d in enumerate(equity.index)
                                  if regime_df["date"].iloc[i] in regime_dates
                                  if i < len(regime_df)]
                if len(regime_indices) > 10:
                    regime_equity = equity.iloc[regime_indices]
                    perf = self.evaluator.compute_metrics(regime_equity)
                    regime_perf[regime_val] = perf

        return regime_perf

    def run(self, strategies: Optional[List[str]] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            params: Optional[Dict[str, Dict]] = None) -> PortfolioResult:
        """
        运行完整的组合回测。

        Args:
            strategies: 策略名称列表，None则使用全部
            start_date: 开始日期
            end_date: 结束日期
            params: 各策略的自定义参数

        Returns:
            组合回测结果
        """
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
            alerts = self.monitor.evaluate(name, result.metrics,
                                            str(df["date"].iloc[-1]) if len(df) > 0 else "")
            all_alerts.extend(alerts)

        return PortfolioResult(
            strategy_results=strategy_results,
            portfolio_equity=portfolio_equity,
            portfolio_metrics=portfolio_metrics,
            regime_history=regime_df,
            switch_log=switch_log,
            alerts=all_alerts,
        )

    def _compute_portfolio_equity(self, strategy_results: Dict[str, StrategyResult],
                                   weights: Dict[str, float]) -> pd.DataFrame:
        """计算组合净值（收益率加权平均法）。"""
        all_dates = set()
        for result in strategy_results.values():
            if not result.equity_curve.empty:
                all_dates.update(result.equity_curve["date"].tolist())

        all_dates = sorted(all_dates)

        # 计算各策略日收益率
        strategy_returns = {}
        for name, result in strategy_results.items():
            if result.equity_curve.empty:
                continue
            eq = result.equity_curve.sort_values("date")
            daily_ret = eq["equity"].pct_change().fillna(0)
            strategy_returns[name] = dict(zip(eq["date"], daily_ret))

        # 计算组合日收益率
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

    def generate_report(self, result: PortfolioResult, output_dir: str = "./output") -> str:
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
        if not result.regime_history.empty and "regime" in result.regime_history.columns:
            lines.extend(["", "## 市场环境分析", ""])
            regime_counts = result.regime_history["regime"].value_counts()
            lines.append("| 环境类型 | 出现次数 | 占比 |")
            lines.append("|----------|----------|------|")
            total = len(result.regime_history)
            for regime, count in regime_counts.items():
                lines.append(f"| {regime} | {count} | {count/total*100:.1f}% |")

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
