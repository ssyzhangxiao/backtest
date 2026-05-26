"""
PyBroker 适配器 — 以 PyBroker 为主回测引擎，自研引擎为验证层的混合架构。

位置: core/engine/broker_adapter.py

提供五个核心组件：
  - PyBrokerDataSource：将 DataFrame 转为 PyBroker 数据源
  - create_hybrid_data_source：TqSdk 在线数据优先，本地 CSV 为 fallback
  - RegimeIndicator：将 MarketRegimeDetector 注册为 PyBroker 自定义指标
  - StrategyExecutorFactory：根据策略名生成 PyBroker 策略执行函数
  - PyBrokerBacktestRunner：主回测运行器，含 walkforward、bootstrap

设计决策：
  - PyBroker 是可选依赖。若未安装，导入时打印提示，自动回退到 _run_simplified。
  - 所有共享模块（MarketRegimeDetector, StrategyLibrary, StrategySwitchEngine）
    通过适配器间接调用，不做重大改动。
  - 策略执行函数内部通过 ctx.indicator("regime") 获取环境，
    通过 switch_engine.decide() 评估是否切换。
  - 指标注册使用 @pybroker.indicator 装饰器，向量化计算。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque

import pandas as pd
import numpy as np

from core.config import BacktestConfig
from core.market_regime import MarketRegimeDetector, MarketRegime
from core.strategy_library import StrategyLibrary
from core.engine.switch_engine import StrategySwitchEngine, SwitchConfig

logger = logging.getLogger(__name__)

# ── PyBroker 可选依赖 ──
# 若未安装则打印提示，相关功能在调用时自动回退到自研简化引擎
try:
    import pybroker
    from pybroker import ExecContext

    PYBROKER_AVAILABLE = True
except ImportError:
    PYBROKER_AVAILABLE = False
    logger.warning("PyBroker 未安装。请运行: pip install pybroker>=1.0.0")
    ExecContext = Any  # type: ignore


# ═══════════════════════════════════════════════════════════════
# 结果封装
# ═══════════════════════════════════════════════════════════════


@dataclass
class PyBrokerResult:
    """PyBroker 回测结果封装。"""

    metrics: Dict[str, float]
    equity_curve: pd.DataFrame  # date, equity
    trades: pd.DataFrame  # 交易记录
    regime_history: pd.DataFrame  # 环境识别历史
    switch_log: pd.DataFrame  # 策略切换日志
    bootstrap_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class WalkforwardResult:
    """Walkforward 向前滚动分析结果。"""

    windows: List[Dict[str, Any]]  # 每轮窗口的详细结果
    overall_metrics: Dict[str, float]  # 合并后的整体指标
    equity_curves: List[pd.DataFrame]  # 每轮窗口的净值曲线

    def plot_equity_curves(self):
        """绘制各窗口净值曲线（需 plotly）。"""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            for i, eq in enumerate(self.equity_curves):
                fig.add_trace(
                    go.Scatter(
                        x=eq["date"],
                        y=eq["equity"],
                        mode="lines",
                        name=f"Window {i + 1}",
                    )
                )
            fig.update_layout(
                title="Walkforward Equity Curves",
                xaxis_title="Date",
                yaxis_title="Equity",
            )
            fig.show()
        except ImportError:
            logger.warning("plotly 未安装，无法绘图。请运行: pip install plotly")
            for i, eq in enumerate(self.equity_curves):
                logger.info(
                    "Window %d: final equity = %.2f",
                    i + 1,
                    eq["equity"].iloc[-1],
                )


# ═══════════════════════════════════════════════════════════════
# 1. PyBroker 数据源
# ═══════════════════════════════════════════════════════════════


class PyBrokerDataSource:
    """
    PyBroker 兼容数据源。

    接受 pd.DataFrame（格式同 DataLoader.get_pybroker_df() 输出），
    提供 query 方法返回按日期/合约筛选的数据。

    使用方式：
        ds = PyBrokerDataSource(df)
        df_pybroker = ds.to_pybroker_df()
    """

    def __init__(self, df: pd.DataFrame):
        required_cols = {"date", "symbol", "open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"数据缺少必要列: {missing}")

        self._df = df.copy()
        self._df["date"] = pd.to_datetime(self._df["date"])
        self._df = self._df.sort_values(["symbol", "date"]).reset_index(drop=True)
        self._symbols = sorted(self._df["symbol"].unique())

    @property
    def symbols(self) -> List[str]:
        return self._symbols

    @property
    def date_range(self) -> Tuple[str, str]:
        return (
            str(self._df["date"].min().date()),
            str(self._df["date"].max().date()),
        )

    def query(
        self, start_date: str, end_date: str, symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """按日期和合约查询数据。"""
        mask = (self._df["date"] >= pd.Timestamp(start_date)) & (
            self._df["date"] <= pd.Timestamp(end_date)
        )
        result = self._df[mask].copy()
        if symbols:
            result = result[result["symbol"].isin(symbols)]
        return result

    def to_pybroker_df(self) -> pd.DataFrame:
        """返回 PyBroker 可直接使用的完整 DataFrame。"""
        return self._df.copy()

    def __len__(self) -> int:
        return len(self._df)


# ═══════════════════════════════════════════════════════════════
# 1.1 混合数据源工厂 — TqSdk 优先，本地 CSV 为 fallback
# ═══════════════════════════════════════════════════════════════


def create_hybrid_data_source(
    phone: Optional[str] = None,
    password: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
    data_length: int = 2000,
) -> PyBrokerDataSource:
    """
    混合数据源工厂：TqSdk 在线数据优先，本地 CSV 为 fallback。

    加载策略：
      1. 若提供 phone + password + symbols → 尝试从 TqSdk 加载实时数据
         → 成功则转为 PyBrokerDataSource 返回
      2. TqSdk 加载失败（未提供凭证/网络错误/账号过期等）→ 回退到 DataLoader
         → 从 data_dir 加载 CSV 数据
      3. 两者均失败 → 抛出 RuntimeError

    Args:
        phone: 快期账号手机号（可选，从环境变量 TQSDK_PHONE 读取）
        password: 快期账号密码（可选，从环境变量 TQSDK_PASSWORD 读取）
        symbols: 品种代码列表（如 ["SHFE.RB", "DCE.M"]），TqSdk 模式必需
        data_dir: 本地 CSV 数据目录（默认 "./data"）
        data_length: TqSdk 每个合约下载的 K 线数量

    Returns:
        PyBrokerDataSource 实例

    Raises:
        RuntimeError: 两种数据源均加载失败
    """
    import os

    # ── 1. 尝试 TqSdk ──
    phone = phone or os.environ.get("TQSDK_PHONE")
    password = password or os.environ.get("TQSDK_PASSWORD")

    if phone and password and symbols:
        try:
            from core.data_loader_tqsdk import TqSdkDataSource

            logger.info("尝试从 TqSdk 加载数据 (%d 个品种)...", len(symbols))
            tqsdk_loader = TqSdkDataSource(
                phone=phone,
                password=password,
                symbols=symbols,
                data_length=data_length,
            )
            tqsdk_loader.load_from_tqsdk(show_progress=True)

            if tqsdk_loader.load_errors:
                logger.warning(
                    "TqSdk 部分品种加载失败 (%d 个)：%s",
                    len(tqsdk_loader.load_errors),
                    [e.get("symbol", "?") for e in tqsdk_loader.load_errors],
                )

            tqsdk_loader.identify_dominant_contracts()
            tqsdk_loader.build_continuous_series()
            df = tqsdk_loader.get_pybroker_df()

            if df.empty:
                raise RuntimeError("TqSdk 返回空数据")

            logger.info(
                "TqSdk 数据加载成功: %d 行, %d 品种",
                len(df),
                df["product"].nunique()
                if "product" in df.columns
                else df["symbol"].nunique(),
            )
            return PyBrokerDataSource(df)

        except Exception as e:
            logger.warning("TqSdk 数据加载失败 (%s)，回退到本地 CSV 数据源。", e)

    # ── 2. 回退到 DataLoader ──
    data_dir = data_dir or os.environ.get("DATA_DIR", "./data")
    logger.info("从本地 CSV 加载数据 (%s)...", data_dir)

    try:
        from core.data_loader import DataLoader

        loader = DataLoader(data_dir)
        loader.load_csv_files("*.csv")
        loader.build_continuous_series()
        df = loader.get_pybroker_df()

        if df.empty:
            raise RuntimeError("DataLoader 返回空数据")

        logger.info(
            "本地 CSV 数据加载成功: %d 行",
            len(df),
        )
        return PyBrokerDataSource(df)

    except Exception as e:
        raise RuntimeError(
            f"两种数据源均加载失败。TqSdk 和本地 CSV ({data_dir}) 均不可用。"
            f"\n  最后错误: {e}"
        ) from e


# ═══════════════════════════════════════════════════════════════
# 2. 市场环境指标
# ═══════════════════════════════════════════════════════════════


class RegimeIndicator:
    """
    将 MarketRegimeDetector 包装为可在 PyBroker 中使用的指标。

    提供 fit() / detect() 用于非 PyBroker 路径，
    以及 create_pybroker_fn() 返回可用于 @pybroker.indicator 的函数。

    注意：PyBroker 每个 bar 调用一次 indicator，缓存意义不大，
    因此不再使用实例级缓存，而是利用 PyBroker 的内置缓存机制。
    """

    def __init__(self, detector: Optional[MarketRegimeDetector] = None):
        self._detector = detector or MarketRegimeDetector()
        self._is_fitted = False

    def fit(self, df: pd.DataFrame):
        """在样本内数据上拟合探测器。"""
        dominant = df.copy()
        if "is_dominant" in dominant.columns:
            dominant = dominant[dominant["is_dominant"]]
        dominant = dominant.sort_values("date")
        self._detector.fit(dominant)
        self._is_fitted = True
        logger.info("RegimeIndicator 已拟合，样本内 %d 行", len(dominant))

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对整个 DataFrame 执行环境检测（非 PyBroker 路径使用）。

        Args:
            df: 行情数据 DataFrame。

        Returns:
            含 regime, regime_confidence 列的 DataFrame。
        """
        dominant = df.copy()
        if "is_dominant" in dominant.columns:
            dominant = dominant[dominant["is_dominant"]]
        dominant = dominant.sort_values("date")

        if self._is_fitted:
            result = self._detector.transform(dominant)
        else:
            result = self._detector.detect(dominant)
        return result

    def create_pybroker_regime_fn(self):
        """
        创建可用于 @pybroker.indicator('regime') 的函数。

        该函数接受 bar_data 并返回 Series：
          - regime: 环境标签字符串
          - regime_confidence: 置信度浮点数

        Returns:
            (fn_regime, fn_confidence) 两个可注册的函数。
        """
        detector = self._detector  # 捕获引用

        def regime_fn(bar_data):
            """返回环境标签序列。"""
            df = pd.DataFrame(
                {
                    "open": bar_data.open,
                    "high": bar_data.high,
                    "low": bar_data.low,
                    "close": bar_data.close,
                    "volume": bar_data.volume,
                }
            )
            if len(df) < 20:
                return pd.Series(["UNKNOWN"] * len(df), index=bar_data.date)
            try:
                result = detector.detect(df)
                if "regime" in result.columns:
                    # 对齐到 bar_data 的日期
                    regime_series = result.set_index("date")["regime"]
                    return regime_series.reindex(bar_data.index, fill_value="UNKNOWN")
            except Exception:
                pass
            return pd.Series(["UNKNOWN"] * len(bar_data), index=bar_data.index)

        def regime_conf_fn(bar_data):
            """返回环境置信度序列。"""
            df = pd.DataFrame(
                {
                    "open": bar_data.open,
                    "high": bar_data.high,
                    "low": bar_data.low,
                    "close": bar_data.close,
                    "volume": bar_data.volume,
                }
            )
            if len(df) < 20:
                return pd.Series([0.5] * len(df), index=bar_data.index)
            try:
                result = detector.detect(df)
                if "regime_confidence" in result.columns:
                    conf_series = result.set_index("date")["regime_confidence"]
                    return conf_series.reindex(bar_data.index, fill_value=0.5)
            except Exception:
                pass
            return pd.Series([0.5] * len(bar_data), index=bar_data.index)

        return regime_fn, regime_conf_fn


# ═══════════════════════════════════════════════════════════════
# 3. 策略执行器工厂
# ═══════════════════════════════════════════════════════════════


class StrategyExecutorFactory:
    """
    根据策略名称生成 PyBroker 策略执行函数（fn(ctx: ExecContext)）。

    每个策略执行函数：
      1. 从 ctx.indicator("regime") 获取当前市场环境
      2. 从 ctx.indicator("regime_confidence") 获取置信度
      3. 调用 switch_engine.decide() 评估是否切换策略
      4. 根据当前激活策略计算买卖信号
      5. 调用 ctx.buy_shares / ctx.sell_shares 下单
    """

    def __init__(
        self,
        library: Optional[StrategyLibrary] = None,
        switch_engine: Optional[StrategySwitchEngine] = None,
        config: Optional[BacktestConfig] = None,
    ):
        self.library = library or StrategyLibrary()
        self.switch_engine = switch_engine or StrategySwitchEngine(self.library)
        self.config = config or BacktestConfig()
        self._position_size = self.config.max_position_pct

    def create_executor(self, strategy_name: str):
        """
        创建单个策略的 PyBroker 执行函数。

        Args:
            strategy_name: 策略名称（如 "dual_ma", "rsi"）

        Returns:
            可传入 pybroker.Strategy.add_execution() 的执行函数。
        """
        profile = self.library.get_profile(strategy_name)
        if profile is None:
            raise ValueError(f"未知策略: {strategy_name}")

        params = dict(profile.default_params)
        position_size = self._position_size
        switch_engine = self.switch_engine

        # ── 滚动 Sharpe 状态（闭包内维护） ──
        switch_cfg = SwitchConfig()
        lookback = switch_cfg.performance_lookback  # 默认20
        daily_returns: deque = deque(maxlen=lookback)
        prev_equity: float = self.config.initial_cash

        def executor_fn(ctx: ExecContext):
            """PyBroker 策略执行函数。"""
            nonlocal prev_equity, params

            # ── 1. 获取当前市场环境 ──
            regime_str = None
            regime_confidence = 0.5
            try:
                regime_str = ctx.indicator("regime")
                regime_confidence_val = ctx.indicator("regime_confidence")
                if regime_str is not None:
                    # PyBroker indicator 返回 Series，取最新值
                    if hasattr(regime_str, "iloc"):
                        regime_str = (
                            str(regime_str.iloc[-1]) if len(regime_str) > 0 else None
                        )
                    else:
                        regime_str = str(regime_str)
                if regime_confidence_val is not None:
                    if hasattr(regime_confidence_val, "iloc"):
                        regime_confidence = (
                            float(regime_confidence_val.iloc[-1])
                            if len(regime_confidence_val) > 0
                            else 0.5
                        )
                    else:
                        regime_confidence = float(regime_confidence_val)
            except Exception:
                pass  # 指标未注册时忽略

            # ── 2. 计算滚动 Sharpe ──
            try:
                current_equity = ctx.total_equity
            except Exception:
                current_equity = prev_equity
            daily_ret = (current_equity / prev_equity) - 1 if prev_equity > 0 else 0.0
            daily_returns.append(daily_ret)
            prev_equity = current_equity

            current_sharpe = 0.0
            if len(daily_returns) >= lookback:
                ret_list = list(daily_returns)
                mean_ret = np.mean(ret_list)
                std_ret = np.std(ret_list, ddof=1)
                if std_ret > 1e-10:
                    current_sharpe = (mean_ret / std_ret) * np.sqrt(252)

            # ── 3. 策略切换评估 ──
            # 从 switch_engine 获取当前激活策略（所有 executor 共享同一状态）
            active_strategy = switch_engine.get_current_strategy() or strategy_name
            current_regime = (
                MarketRegime(regime_str) if regime_str else MarketRegime.TREND_UP
            )
            try:
                bar_date = str(ctx.dt.date()) if hasattr(ctx, "dt") else ""
                decision = switch_engine.decide(
                    current_date=bar_date,
                    current_regime=current_regime,
                    regime_confidence=regime_confidence,
                    current_sharpe=current_sharpe,
                    position_value=ctx.total_market_value,
                    has_position=(ctx.long_shares + ctx.short_shares > 0),
                    sharpe_samples=len(daily_returns),
                )
                if decision and decision.approved:
                    active_strategy = decision.to_strategy
                    # 更新参数为新策略的默认参数
                    new_profile = self.library.get_profile(active_strategy)
                    if new_profile:
                        params.update(new_profile.default_params)
                    logger.debug(
                        "PyBroker executor 策略切换: → %s (原因: %s)",
                        active_strategy,
                        decision.reason.value,
                    )
            except Exception:
                pass  # 切换失败不影响执行

            # ── 4. 信号计算 ──
            close = ctx.close
            current_close = (
                close[-1] if hasattr(close, "__getitem__") and len(close) > 0 else close
            )

            # 获取指标值（从 PyBroker registered indicators）
            sma_5_val = None
            sma_20_val = None
            rsi_val = None
            bb_upper_val = None
            bb_lower_val = None

            try:
                raw = ctx.indicator("sma_5")
                sma_5_val = raw[-1] if hasattr(raw, "iloc") and len(raw) > 0 else raw
            except Exception:
                pass
            try:
                raw = ctx.indicator("sma_20")
                sma_20_val = raw[-1] if hasattr(raw, "iloc") and len(raw) > 0 else raw
            except Exception:
                pass
            try:
                raw = ctx.indicator("rsi_14")
                rsi_val = raw[-1] if hasattr(raw, "iloc") and len(raw) > 0 else raw
            except Exception:
                pass
            try:
                raw = ctx.indicator("bb_upper")
                bb_upper_val = raw[-1] if hasattr(raw, "iloc") and len(raw) > 0 else raw
            except Exception:
                pass
            try:
                raw = ctx.indicator("bb_lower")
                bb_lower_val = raw[-1] if hasattr(raw, "iloc") and len(raw) > 0 else raw
            except Exception:
                pass

            # ── 5. 根据激活策略执行交易 ──
            signal = 0  # 0=none, 1=buy, -1=sell

            if active_strategy == "dual_ma":
                if sma_5_val is not None and sma_20_val is not None:
                    if sma_5_val > sma_20_val:
                        signal = 1
                    elif sma_5_val < sma_20_val:
                        signal = -1
                else:
                    # 回退到 ctx 已有的指标方法
                    try:
                        sma_5 = ctx.sma(5)
                        sma_20 = ctx.sma(20)
                        if sma_5[-1] > sma_20[-1]:
                            signal = 1
                        elif sma_5[-1] < sma_20[-1]:
                            signal = -1
                    except Exception:
                        pass

            elif active_strategy == "rsi":
                oversold = params.get("oversold", 30.0)
                overbought = params.get("overbought", 70.0)
                if rsi_val is not None:
                    if rsi_val < oversold:
                        signal = 1
                    elif rsi_val > overbought:
                        signal = -1
                else:
                    try:
                        rsi = ctx.rsi(14)
                        if rsi[-1] < oversold:
                            signal = 1
                        elif rsi[-1] > overbought:
                            signal = -1
                    except Exception:
                        pass

            elif active_strategy == "vol_breakout":
                if bb_upper_val is not None and bb_lower_val is not None:
                    if current_close > bb_upper_val:
                        signal = 1
                    elif current_close < bb_lower_val:
                        signal = -1
                else:
                    try:
                        bb = ctx.bb(30)
                        if current_close > bb.upper[-1]:
                            signal = 1
                        elif current_close < bb.lower[-1]:
                            signal = -1
                    except Exception:
                        pass

            # term_structure, spread 等在此扩展

            # ── 6. 环境自适应过滤 ──
            if signal != 0 and regime_str is not None:
                if not self._should_trade(regime_str, active_strategy):
                    signal = 0

            # ── 7. 执行下单 ──
            if signal == 1:
                ctx.buy_shares = ctx.calc_target_shares(position_size)
            elif signal == -1:
                ctx.sell_shares = ctx.calc_target_shares(position_size)

        executor_fn.__name__ = f"executor_{strategy_name}"
        return executor_fn

    @staticmethod
    def _should_trade(regime: Optional[str], strategy_name: str) -> bool:
        """
        判断给定策略在当前市场环境下是否应该交易。

        规则：
          - dual_ma: RANGE_BOUND 时不交易
          - vol_breakout: LOW_VOLATILITY 时不交易
          - 其他策略：始终允许
        """
        if regime is None:
            return True
        regime_upper = (
            regime.upper() if isinstance(regime, str) else str(regime).upper()
        )

        if strategy_name == "dual_ma":
            if regime_upper in ("RANGE_BOUND",):
                return False
            return True
        elif strategy_name == "vol_breakout":
            if regime_upper in ("LOW_VOLATILITY",):
                return False
            return True
        return True


# ═══════════════════════════════════════════════════════════════
# 4. PyBroker 主回测运行器
# ═══════════════════════════════════════════════════════════════


class PyBrokerBacktestRunner:
    """
    PyBroker 主回测运行器。

    功能：
      - run: PyBroker 主回测，不可用时回退到 _run_simplified
      - walkforward: 向前滚动分析（PyBroker 优先 / 自定义实现）
      - bootstrap_metrics: 绩效指标置信区间（PyBroker 优先 / numpy 自实现）

    使用方式：
        ds = PyBrokerDataSource(df)
        runner = PyBrokerBacktestRunner(ds, config)
        runner.register_strategies(["dual_ma", "rsi"])
        result = runner.run("2023-01-01", "2024-12-31")
    """

    def __init__(
        self, data_source: PyBrokerDataSource, config: Optional[BacktestConfig] = None
    ):
        self.data_source = data_source
        self.config = config or BacktestConfig()
        self.library = StrategyLibrary()
        self.switch_engine = StrategySwitchEngine(self.library)
        self.regime_indicator = RegimeIndicator()
        self.executor_factory = StrategyExecutorFactory(
            self.library, self.switch_engine, self.config
        )

        self._registered_strategies: List[str] = []
        self._last_result: Optional[PyBrokerResult] = None

    def register_strategies(self, strategy_names: List[str]):
        """注册策略名称列表。"""
        self._registered_strategies = list(strategy_names)
        logger.info("已注册策略: %s", strategy_names)

    # ------------------------------------------------------------
    # run — 主回测
    # ------------------------------------------------------------

    def run(
        self,
        start_date: str,
        end_date: str,
        initial_cash: Optional[float] = None,
        use_fallback: bool = False,
    ) -> PyBrokerResult:
        """
        执行回测（PyBroker 主引擎优先）。

        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
            initial_cash: 初始资金，默认 config.initial_cash
            use_fallback: 强制使用自研简化引擎

        Returns:
            PyBrokerResult
        """
        if not self._registered_strategies:
            raise RuntimeError("请先调用 register_strategies() 注册策略")

        cash = initial_cash or self.config.initial_cash

        if PYBROKER_AVAILABLE and not use_fallback:
            try:
                result = self._run_pybroker(start_date, end_date, cash)
            except Exception as e:
                logger.warning("PyBroker 执行失败 (%s)，回退到自研简化引擎。", e)
                result = self._run_fallback(start_date, end_date, cash)
        else:
            if not PYBROKER_AVAILABLE:
                logger.warning("PyBroker 未安装，使用自研简化引擎。")
            result = self._run_fallback(start_date, end_date, cash)

        self._last_result = result
        return result

    def _run_pybroker(
        self, start_date: str, end_date: str, initial_cash: float
    ) -> PyBrokerResult:
        """
        使用 PyBroker 原生 API 执行回测。

        流程：
          1. 准备数据和指标
          2. 创建 pybroker.StrategyConfig 和 pybroker.Strategy
          3. 注册向量化指标 + 环境指标
          4. 添加策略执行函数
          5. 执行 strategy.backtest()
          6. 提取结果
        """
        if not PYBROKER_AVAILABLE:
            raise RuntimeError("PyBroker 不可用")

        strategies = self._registered_strategies
        df = self.data_source.to_pybroker_df()

        # 日期过滤
        df = df[
            (df["date"] >= pd.Timestamp(start_date))
            & (df["date"] <= pd.Timestamp(end_date))
        ]

        # ── 拟合环境检测器 ──
        self.regime_indicator.fit(df)
        regime_fn, regime_conf_fn = self.regime_indicator.create_pybroker_regime_fn()

        # ── 创建 PyBroker StrategyConfig + Strategy ──
        pb_config = pybroker.StrategyConfig(
            initial_cash=initial_cash,
            buy_delay=self.config.pybroker_buy_delay,
            sell_delay=self.config.pybroker_sell_delay,
            bootstrap_samples=self.config.pybroker_bootstrap_samples,
        )
        strategy = pybroker.Strategy(df, start_date, end_date, config=pb_config)

        # ── 注册向量化指标（pybroker.indicator 通过 StaticScope 全局注册） ──
        def _sma_5(bar_data):
            return bar_data.close.rolling(5).mean()

        def _sma_20(bar_data):
            return bar_data.close.rolling(20).mean()

        def _rsi_14(bar_data):
            delta = bar_data.close.diff()
            gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / 14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
            rs = np.where(loss > 0, gain / loss, 100.0)
            return 100 - 100 / (1 + rs)

        def _bb_upper(bar_data):
            center = bar_data.close.rolling(30).mean()
            std = bar_data.close.rolling(30).std()
            return center + 2 * std

        def _bb_lower(bar_data):
            center = bar_data.close.rolling(30).mean()
            std = bar_data.close.rolling(30).std()
            return center - 2 * std

        def _regime(bar_data):
            return regime_fn(bar_data)

        def _regime_conf(bar_data):
            return regime_conf_fn(bar_data)

        _indicators = [
            pybroker.indicator("sma_5", _sma_5),
            pybroker.indicator("sma_20", _sma_20),
            pybroker.indicator("rsi_14", _rsi_14),
            pybroker.indicator("bb_upper", _bb_upper),
            pybroker.indicator("bb_lower", _bb_lower),
            pybroker.indicator("regime", _regime),
            pybroker.indicator("regime_confidence", _regime_conf),
        ]
        # 抑制未使用警告
        _ = _indicators

        # ── 添加策略执行函数 ──
        symbols = self.data_source.symbols
        for name in strategies:
            executor = self.executor_factory.create_executor(name)
            strategy.add_execution(executor, symbols=symbols)

        # ── 执行回测（PyBroker v1.2: Strategy.backtest()） ──
        pb_result = strategy.backtest(
            start_date=start_date,
            end_date=end_date,
            lookahead=self.config.pybroker_buy_delay,
            calc_bootstrap=True,
        )
        # 存储原始 TestResult 供 bootstrap 使用
        self._last_pb_result = pb_result

        # ── 提取结果（TestResult 类型） ──
        # 净值曲线 (portfolio: date 是 index, market_value 是 equity)
        if hasattr(pb_result, "portfolio") and isinstance(
            pb_result.portfolio, pd.DataFrame
        ):
            pf = pb_result.portfolio.copy()
            # date 是 index，需 reset_index 转为列
            equity_df = pf.reset_index()
            if "market_value" in equity_df.columns:
                equity_df = equity_df[["date", "market_value"]].rename(
                    columns={"market_value": "equity"}
                )
            elif "equity" in equity_df.columns:
                equity_df = equity_df[["date", "equity"]]
            else:
                equity_df = pd.DataFrame(columns=["date", "equity"])
        else:
            equity_df = pd.DataFrame(columns=["date", "equity"])

        # 交易记录
        if hasattr(pb_result, "trades") and isinstance(pb_result.trades, pd.DataFrame):
            trades = pb_result.trades.copy()
        else:
            trades = pd.DataFrame()

        # 绩效指标 (metrics_df: columns = ["name", "value"])
        if hasattr(pb_result, "metrics_df") and isinstance(
            pb_result.metrics_df, pd.DataFrame
        ):
            mdf = pb_result.metrics_df
            if "name" in mdf.columns and "value" in mdf.columns:
                metrics = dict(zip(mdf["name"], mdf["value"]))
            else:
                metrics = mdf.to_dict(orient="records")[0] if len(mdf) > 0 else {}
        elif hasattr(pb_result, "metrics"):
            m = pb_result.metrics
            metrics = {
                "sharpe": getattr(m, "sharpe", 0.0),
                "total_return_pct": getattr(m, "total_return_pct", 0.0),
                "max_drawdown_pct": getattr(m, "max_drawdown_pct", 0.0),
                "win_rate": getattr(m, "win_rate", 0.0),
                "profit_factor": getattr(m, "profit_factor", 0.0),
                "calmar": getattr(m, "calmar", 0.0),
                "trade_count": getattr(m, "trade_count", 0),
                "total_pnl": getattr(m, "total_pnl", 0.0),
            }
        else:
            metrics = {}

        # 环境历史（从 regime_indicator 重新获取）
        regime_df = self._run_regime_detection(df)

        return PyBrokerResult(
            metrics=metrics,
            equity_curve=equity_df,
            trades=trades,
            regime_history=regime_df,
            switch_log=self.switch_engine.get_decision_summary(),
        )

    def _run_fallback(
        self, start_date: str, end_date: str, initial_cash: float
    ) -> PyBrokerResult:
        """
        自研简化引擎回测（PyBroker 不可用时的 fallback）。

        逻辑与 _run_simplified 相同，确保空头权益计算公式正确。
        """
        return self._run_simplified(start_date, end_date, initial_cash)

    def _run_simplified(
        self, start_date: str, end_date: str, initial_cash: float
    ) -> PyBrokerResult:
        """
        简化回测引擎（fallback / 快速验证用）。

        修复项：
          - 空头权益公式：equity = cash + shares * (entry_price - close)
          - 止损/开平仓的现金变化与空头公式一致
          - 支持多品种（分别回测后等权组合）
        """
        cfg = self.config
        cost_rate = cfg.commission_rate + cfg.slippage_rate
        position_size = cfg.max_position_pct
        strategies = self._registered_strategies or ["dual_ma"]

        df = self.data_source.to_pybroker_df()
        df = df[
            (df["date"] >= pd.Timestamp(start_date))
            & (df["date"] <= pd.Timestamp(end_date))
        ]

        # 环境检测
        regime_result = self._run_regime_detection(df)

        # 多品种：对每个品种分别回测后等权组合
        symbols = self.data_source.symbols
        all_equities = []
        all_trades = []

        for symbol in symbols:
            sym_df = (
                df[df["symbol"] == symbol].sort_values("date").reset_index(drop=True)
            )
            if len(sym_df) < 50:
                continue

            cash = initial_cash / len(symbols)  # 等权分配
            position = 0
            entry_price = 0.0
            shares = 0
            equity_list = []
            trade_records = []

            for i in range(len(sym_df)):
                row = sym_df.iloc[i]
                close = row["close"]
                date = row["date"]

                # ── 权益计算（v3 修复：统一空头公式） ──
                if position == 1:
                    equity = cash + shares * close
                elif position == -1:
                    equity = cash + shares * (entry_price - close)
                else:
                    equity = cash

                # 止损
                stop_pct = cfg.stop_loss_pct
                if position == 1 and close < entry_price * (1 - stop_pct):
                    base_pnl = shares * (close - entry_price)
                    exit_cost = shares * close * cost_rate
                    trade_records.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "side": "stop_loss_long",
                            "price": close,
                            "shares": shares,
                            "pnl": base_pnl - exit_cost,
                        }
                    )
                    cash += shares * close * (1 - cost_rate)
                    position = 0
                    shares = 0

                elif position == -1 and close > entry_price * (1 + stop_pct):
                    base_pnl = shares * (entry_price - close)
                    exit_cost = shares * close * cost_rate
                    trade_records.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "side": "stop_loss_short",
                            "price": close,
                            "shares": shares,
                            "pnl": base_pnl - exit_cost,
                        }
                    )
                    cash -= shares * close * (1 + cost_rate)
                    position = 0
                    shares = 0

                # 信号
                signal = self._generate_simple_signal(sym_df, i, strategies[0])

                if signal == 1 and position != 1:
                    if position == -1:
                        base_pnl = shares * (entry_price - close)
                        exit_cost = shares * close * cost_rate
                        trade_records.append(
                            {
                                "date": date,
                                "symbol": symbol,
                                "side": "short_close",
                                "price": close,
                                "shares": shares,
                                "pnl": base_pnl - exit_cost,
                            }
                        )
                        cash -= shares * close * (1 + cost_rate)
                        position = 0
                        shares = 0
                    alloc = equity * position_size
                    shares = int(alloc / close) if close > 0 else 0
                    if shares > 0:
                        cash -= shares * close * (1 + cost_rate)
                        entry_price = close
                        position = 1

                elif signal == -1 and position != -1:
                    if position == 1:
                        base_pnl = shares * (close - entry_price)
                        exit_cost = shares * close * cost_rate
                        trade_records.append(
                            {
                                "date": date,
                                "symbol": symbol,
                                "side": "long_close",
                                "price": close,
                                "shares": shares,
                                "pnl": base_pnl - exit_cost,
                            }
                        )
                        cash += shares * close * (1 - cost_rate)
                        position = 0
                        shares = 0
                    alloc = equity * position_size
                    shares = int(alloc / close) if close > 0 else 0
                    if shares > 0:
                        cash += shares * close * (1 - cost_rate)
                        entry_price = close
                        position = -1

                # 期末权益
                if position == 1:
                    equity = cash + shares * close
                elif position == -1:
                    equity = cash + shares * (entry_price - close)
                else:
                    equity = cash

                equity_list.append({"date": date, "symbol": symbol, "equity": equity})

            all_equities.append(pd.DataFrame(equity_list))
            all_trades.append(
                pd.DataFrame(trade_records)
                if trade_records
                else pd.DataFrame(
                    columns=["date", "symbol", "side", "price", "shares", "pnl"]
                )
            )

        # ── 合并多品种 ──
        if not all_equities:
            empty_metrics = {"error": "no_data"}
            return PyBrokerResult(
                metrics=empty_metrics,
                equity_curve=pd.DataFrame(columns=["date", "equity"]),
                trades=pd.DataFrame(),
                regime_history=regime_result,
                switch_log=self.switch_engine.get_decision_summary(),
            )

        # 对齐所有品种的净值曲线到相同日期
        combined_eq = pd.DataFrame({"date": pd.NaT, "equity": 0.0}, index=[0])
        if len(all_equities) > 1:
            eq_curves: Dict[str, pd.Series] = {}
            for eq_df in all_equities:
                sym = eq_df["symbol"].iloc[0]
                eq_curves[sym] = pd.Series(eq_df["equity"].values, index=eq_df["date"])

            all_dates = sorted(set().union(*(e["date"] for e in all_equities)))
            portfolio_data = []
            for date in all_dates:
                day_eq = 0.0
                for eq_ser in eq_curves.values():
                    # 取最近的不晚于 date 的净值
                    mask = eq_ser.index <= date
                    if mask.any():
                        day_eq += eq_ser.loc[mask].iloc[-1]
                portfolio_data.append({"date": date, "equity": day_eq})

            combined_eq = pd.DataFrame(portfolio_data)
        else:
            combined_eq = all_equities[0][["date", "equity"]]

        # 绩效指标
        if len(combined_eq) > 1:
            daily_ret = combined_eq["equity"].pct_change().dropna()
            metrics = self._compute_simple_metrics(combined_eq["equity"], daily_ret)
        else:
            metrics = {"error": "insufficient_data"}

        # 交易记录合并
        trades_df = (
            pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        )

        return PyBrokerResult(
            metrics=metrics,
            equity_curve=combined_eq,
            trades=trades_df,
            regime_history=regime_result,
            switch_log=self.switch_engine.get_decision_summary(),
        )

    # ------------------------------------------------------------
    # Walkforward
    # ------------------------------------------------------------

    def walkforward(
        self,
        start_date: str,
        end_date: str,
        train_ratio: Optional[float] = None,
        step_ratio: Optional[float] = None,
    ) -> WalkforwardResult:
        """
        向前滚动分析。

        窗口分割逻辑（训练集和测试集不重叠）：
          - 第 k 轮: train = [start, test_start), test = [test_start, test_end)
          - train_end = test_start（非重叠）
          - 每轮使用独立的 RegimeIndicator 实例，避免状态污染。
          - 默认使用自定义窗口实现（提供 per-window 明细）；
            若指定 use_pybroker_wf=True 则使用 PyBroker 内置 walkforward
            （聚合结果，无 per-window 明细）。

        Args:
            start_date: 总体开始日期
            end_date: 总体结束日期
            train_ratio: 训练集占比（默认 config.wf_train_ratio）
            step_ratio: 每次步进步长（默认 config.wf_step_ratio）

        Returns:
            WalkforwardResult
        """
        train_ratio = train_ratio or self.config.wf_train_ratio
        step_ratio = step_ratio or self.config.wf_step_ratio

        return self._walkforward_custom(start_date, end_date, train_ratio, step_ratio)

    def _walkforward_pybroker(
        self, start_date: str, end_date: str, train_ratio: float
    ) -> WalkforwardResult:
        """使用 PyBroker 内置 walkforward 方法。"""
        df = self.data_source.to_pybroker_df()
        df = df[
            (df["date"] >= pd.Timestamp(start_date))
            & (df["date"] <= pd.Timestamp(end_date))
        ]

        pb_config = pybroker.StrategyConfig(
            initial_cash=self.config.initial_cash,
            buy_delay=self.config.pybroker_buy_delay,
            sell_delay=self.config.pybroker_sell_delay,
        )
        strategy = pybroker.Strategy(df, start_date, end_date, config=pb_config)

        # 注册指标（与 _run_pybroker 相同的指标）
        def _wf_sma_5(bar_data):
            return bar_data.close.rolling(5).mean()

        def _wf_sma_20(bar_data):
            return bar_data.close.rolling(20).mean()

        def _wf_rsi_14(bar_data):
            delta = bar_data.close.diff()
            gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / 14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
            rs = np.where(loss > 0, gain / loss, 100.0)
            return 100 - 100 / (1 + rs)

        def _wf_bb_upper(bar_data):
            center = bar_data.close.rolling(30).mean()
            std = bar_data.close.rolling(30).std()
            return center + 2 * std

        def _wf_bb_lower(bar_data):
            center = bar_data.close.rolling(30).mean()
            std = bar_data.close.rolling(30).std()
            return center - 2 * std

        regime_fn, _ = self.regime_indicator.create_pybroker_regime_fn()
        _, regime_conf_fn = self.regime_indicator.create_pybroker_regime_fn()

        def _wf_regime(bar_data):
            return regime_fn(bar_data)

        def _wf_regime_conf(bar_data):
            return regime_conf_fn(bar_data)

        _wf_indicators = [
            pybroker.indicator("sma_5", _wf_sma_5),
            pybroker.indicator("sma_20", _wf_sma_20),
            pybroker.indicator("rsi_14", _wf_rsi_14),
            pybroker.indicator("bb_upper", _wf_bb_upper),
            pybroker.indicator("bb_lower", _wf_bb_lower),
            pybroker.indicator("regime", _wf_regime),
            pybroker.indicator("regime_confidence", _wf_regime_conf),
        ]
        _ = _wf_indicators

        # 添加执行函数
        symbols = self.data_source.symbols
        for name in self._registered_strategies:
            executor = self.executor_factory.create_executor(name)
            strategy.add_execution(executor, symbols=symbols)

        # Windows 数量估算
        n_windows = max(2, int(1.0 / (1.0 - train_ratio)))

        # 执行 PyBroker walkforward
        wf_result = strategy.walkforward(
            windows=n_windows,
            train_size=train_ratio,
            lookahead=self.config.pybroker_buy_delay,
        )

        # 提取结果
        windows = []
        equity_curves = []

        # walkforward 返回 TestResult（每个窗口结果包含在 metrics_df 中）
        if hasattr(wf_result, "metrics_df") and isinstance(
            wf_result.metrics_df, pd.DataFrame
        ):
            mdf = wf_result.metrics_df
            # 每行是一个窗口
            for _, row in mdf.iterrows():
                window_metrics = {
                    k: v
                    for k, v in row.items()
                    if isinstance(v, (int, float)) and not pd.isna(v)
                }
                windows.append(
                    {
                        "train_start": str(row.get("train_start_date", "")),
                        "train_end": str(row.get("train_end_date", "")),
                        "test_start": str(row.get("test_start_date", "")),
                        "test_end": str(row.get("test_end_date", "")),
                        "metrics": window_metrics,
                    }
                )

        # 净值曲线
        if hasattr(wf_result, "portfolio") and isinstance(
            wf_result.portfolio, pd.DataFrame
        ):
            pf = wf_result.portfolio.copy()
            if "market_value" in pf.columns:
                pf = pf.rename(columns={"market_value": "equity"})
            equity_curves.append(pf)

        # 整体指标
        overall = {}
        if windows:
            metric_keys = [
                k
                for k in windows[0]["metrics"]
                if isinstance(windows[0]["metrics"][k], (int, float))
            ]
            for key in metric_keys:
                vals = [w["metrics"][key] for w in windows]
                overall[key] = round(float(np.mean(vals)), 4)

        return WalkforwardResult(
            windows=windows,
            overall_metrics=overall,
            equity_curves=equity_curves,
        )

    def _walkforward_custom(
        self,
        start_date: str,
        end_date: str,
        train_ratio: float,
        step_ratio: float,
    ) -> WalkforwardResult:
        """
        自定义向前滚动分析（PyBroker 不可用时的 fallback）。

        窗口分割逻辑（训练集和测试集不重叠）：
          - 第 k 轮: train = [start, test_start), test = [test_start, test_end)
          - train_end = test_start（非重叠）
          - 每轮使用独立的 RegimeIndicator 实例，避免状态污染。
        """
        df = self.data_source.to_pybroker_df()
        df = df[
            (df["date"] >= pd.Timestamp(start_date))
            & (df["date"] <= pd.Timestamp(end_date))
        ]
        dates = sorted(df["date"].unique())
        total = len(dates)

        train_size = max(20, int(total * train_ratio))
        step_size = max(5, int(total * step_ratio))

        windows = []
        equity_curves = []

        for test_start_idx in range(train_size, total, step_size):
            test_end_idx = min(test_start_idx + step_size, total)
            if test_end_idx <= test_start_idx:
                continue

            # v3 修正: train_end = test_start（不重叠）
            train_start_idx = test_start_idx - train_size
            train_dates = dates[train_start_idx:test_start_idx]
            test_dates = dates[test_start_idx:test_end_idx]

            if len(train_dates) < 10 or len(test_dates) < 5:
                continue

            train_df = df[df["date"].isin(train_dates)]
            test_df = df[df["date"].isin(test_dates)]

            # 每轮使用独立的 RegimeIndicator，避免状态污染
            window_regime = RegimeIndicator(MarketRegimeDetector())
            window_regime.fit(train_df)
            regime_test = window_regime.detect(test_df)

            # 创建独立的回测运行器执行测试窗口
            window_runner = _WindowRunner(
                symbols=self.data_source.symbols,
                strategies=self._registered_strategies or ["dual_ma"],
                config=self.config,
            )
            test_result = window_runner.run(test_df, regime_test)

            windows.append(
                {
                    "train_start": str(train_dates[0].date()),
                    "train_end": str(train_dates[-1].date()),
                    "test_start": str(test_dates[0].date()),
                    "test_end": str(test_dates[-1].date()),
                    "metrics": test_result.metrics,
                }
            )
            equity_curves.append(test_result.equity_curve)

        # 合并整体指标
        overall = {}
        if windows:
            metric_keys = [
                k
                for k in windows[0]["metrics"]
                if isinstance(windows[0]["metrics"][k], (int, float))
            ]
            for key in metric_keys:
                vals = [w["metrics"][key] for w in windows]
                overall[key] = round(float(np.mean(vals)), 4)

        return WalkforwardResult(
            windows=windows,
            overall_metrics=overall,
            equity_curves=equity_curves,
        )

    # ------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------

    def bootstrap_metrics(self, n_samples: Optional[int] = None) -> Dict:
        """
        绩效指标 bootstrap 重采样。

        PyBroker 可用时优先使用其内置 bootstrap；
        否则使用基于 numpy 的自实现。

        Args:
            n_samples: 重采样次数（默认 config.pybroker_bootstrap_samples）

        Returns:
            {metric_name: {mean, std, ci_lower(2.5%), ci_upper(97.5%)}}
        """
        if self._last_result is None:
            raise RuntimeError("请先调用 run()")

        n_samples = n_samples or self.config.pybroker_bootstrap_samples

        # PyBroker 内置 bootstrap（优先）
        if PYBROKER_AVAILABLE:
            try:
                return self._bootstrap_pybroker(n_samples)
            except Exception as e:
                logger.warning("PyBroker bootstrap 失败 (%s)，回退到 numpy 实现。", e)

        return self._bootstrap_numpy(n_samples)

    def _bootstrap_pybroker(self, _n_samples: int) -> Dict:
        """使用 PyBroker 内置 bootstrap（从 TestResult.bootstrap 提取）。"""
        if not hasattr(self, "_last_pb_result") or self._last_pb_result is None:
            raise RuntimeError("没有可用的 PyBroker 回测结果")

        pb_result = self._last_pb_result
        if not hasattr(pb_result, "bootstrap") or pb_result.bootstrap is None:
            raise RuntimeError(
                "回测结果中无 bootstrap 数据，请使用 calc_bootstrap=True"
            )

        bs = pb_result.bootstrap
        result = {}

        # conf_intervals: MultiIndex DataFrame，列为 ["lower", "upper"]
        if hasattr(bs, "conf_intervals") and isinstance(
            bs.conf_intervals, pd.DataFrame
        ):
            ci = bs.conf_intervals
            # MultiIndex: (name, conf) → 展平为单个 key
            for idx_val in ci.index:
                if isinstance(idx_val, tuple):
                    metric_name, conf_level = idx_val
                else:
                    metric_name, conf_level = str(idx_val), "value"
                row = ci.loc[idx_val]
                key = f"{metric_name} ({conf_level})"
                result[key] = {
                    "ci_lower": round(float(row.get("lower", 0)), 4),
                    "ci_upper": round(float(row.get("upper", 0)), 4),
                }
            return result

        # Fallback: 提取已知指标的数组
        for attr in ("sharpe", "drawdown", "profit_factor"):
            if hasattr(bs, attr):
                arr = getattr(bs, attr)
                if arr is not None and hasattr(arr, "__len__") and len(arr) > 0:
                    arr_np = np.asarray(arr)
                    result[attr] = {
                        "mean": round(float(np.mean(arr_np)), 4),
                        "std": round(float(np.std(arr_np)), 4),
                        "ci_lower": round(float(np.percentile(arr_np, 2.5)), 4),
                        "ci_upper": round(float(np.percentile(arr_np, 97.5)), 4),
                    }
        return result

    def _bootstrap_numpy(self, n_samples: int) -> Dict:
        """numpy 自实现 bootstrap。"""
        equity = self._last_result.equity_curve["equity"]
        daily_returns = equity.pct_change().dropna()

        if len(daily_returns) < 10:
            return {"error": "样本太少，无法 bootstrap"}

        n = len(daily_returns)
        rng = np.random.default_rng(42)

        metrics_samples: Dict[str, List[float]] = {
            "sharpe": [],
            "total_return": [],
            "max_drawdown": [],
            "calmar": [],
            "win_rate": [],
        }

        # v3: 移除硬编码 min(n_samples, 10000)，使用用户指定的 n_samples
        actual_samples = n_samples
        if actual_samples > 50000:
            logger.info(
                "bootstrap n_samples=%d 较大，可能需要较长时间。", actual_samples
            )

        for _ in range(actual_samples):
            idx = rng.integers(0, n, size=n)
            ret_sample = daily_returns.iloc[idx].values
            eq_sample = equity.iloc[0] * np.cumprod(1 + np.insert(ret_sample, 0, 0))

            ann_factor = np.sqrt(252)
            sharpe = (np.mean(ret_sample) / max(np.std(ret_sample), 1e-8)) * ann_factor
            total_ret = (eq_sample[-1] - eq_sample[0]) / eq_sample[0]
            peak = np.maximum.accumulate(eq_sample)
            dd = np.min((eq_sample - peak) / peak)
            calmar = total_ret / abs(dd) if abs(dd) > 1e-10 else 0.0
            win_rate = np.mean(ret_sample > 0)

            metrics_samples["sharpe"].append(float(sharpe))
            metrics_samples["total_return"].append(float(total_ret))
            metrics_samples["max_drawdown"].append(float(dd))
            metrics_samples["calmar"].append(float(calmar))
            metrics_samples["win_rate"].append(float(win_rate))

        result = {}
        for key, vals in metrics_samples.items():
            arr = np.array(vals)
            result[key] = {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "ci_lower": round(float(np.percentile(arr, 2.5)), 4),
                "ci_upper": round(float(np.percentile(arr, 97.5)), 4),
            }

        self._last_result.bootstrap_metrics = result
        return result

    # ------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------

    def _run_regime_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行环境检测并返回结果 DataFrame。"""
        return self.regime_indicator.detect(df)

    @staticmethod
    def _generate_simple_signal(df: pd.DataFrame, idx: int, strategy_name: str) -> int:
        """简化信号生成（fallback 引擎使用）。"""
        close = df["close"]
        i = idx

        if strategy_name == "dual_ma":
            if i < 20:
                return 0
            sma_5 = close.iloc[max(0, i - 5) : i + 1].mean()
            sma_20 = close.iloc[max(0, i - 20) : i + 1].mean()
            return 1 if sma_5 > sma_20 else (-1 if sma_5 < sma_20 else 0)

        elif strategy_name == "rsi":
            if i < 14:
                return 0
            delta = close.iloc[max(0, i - 14) : i + 1].diff().dropna()
            gain = delta[delta > 0].sum() if len(delta[delta > 0]) > 0 else 0
            loss = abs(delta[delta < 0].sum()) if len(delta[delta < 0]) > 0 else 0
            rs = gain / loss if loss > 0 else 100
            rsi = 100 - 100 / (1 + rs)
            if rsi < 30:
                return 1
            elif rsi > 70:
                return -1
            return 0

        return 0

    @staticmethod
    def _compute_simple_metrics(
        equity: pd.Series, daily_returns: pd.Series
    ) -> Dict[str, float]:
        """计算简化绩效指标。"""
        if len(daily_returns) < 2 or len(equity) < 2:
            return {"error": "insufficient_data"}

        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        ann_factor = np.sqrt(252)
        sharpe = (daily_returns.mean() / max(daily_returns.std(), 1e-8)) * ann_factor
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        max_dd = dd.min()
        calmar = total_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0
        win_rate = (daily_returns > 0).mean()

        return {
            "total_return": round(total_return, 4),
            "sharpe": round(float(sharpe), 3),
            "max_drawdown": round(float(max_dd), 4),
            "calmar": round(float(calmar), 3),
            "win_rate": round(float(win_rate), 4),
            "n_days": len(daily_returns),
            "final_equity": round(float(equity.iloc[-1]), 2),
        }

    def get_last_result(self) -> Optional[PyBrokerResult]:
        """获取最近一次回测结果。"""
        return self._last_result


# ═══════════════════════════════════════════════════════════════
# 内部辅助：Walkforward 窗口独立回测运行器
# ═══════════════════════════════════════════════════════════════


class _WindowRunner:
    """
    Walkforward 每轮窗口的独立回测运行器。

    与主 PyBrokerBacktestRunner 隔离，避免状态污染。
    使用简化引擎（避免 PyBroker 全局状态问题）。
    """

    def __init__(
        self, symbols: List[str], strategies: List[str], config: BacktestConfig
    ):
        self.symbols = symbols
        self.strategies = strategies
        self.config = config

    def run(self, df: pd.DataFrame, regime_df: pd.DataFrame) -> PyBrokerResult:
        """
        对单窗口执行简化回测。

        Args:
            df: 窗口内的数据（含 symbol 列）。
            regime_df: 环境的检测结果。

        Returns:
            PyBrokerResult
        """
        cfg = self.config
        cost_rate = cfg.commission_rate + cfg.slippage_rate
        position_size = cfg.max_position_pct

        symbols = (
            sorted(df["symbol"].unique()) if "symbol" in df.columns else self.symbols
        )
        strategy_name = self.strategies[0] if self.strategies else "dual_ma"

        all_equities = []
        all_trades = []

        per_symbol_cash = cfg.initial_cash / max(len(symbols), 1)

        for symbol in symbols:
            sym_df = df[df["symbol"] == symbol] if "symbol" in df.columns else df.copy()
            sym_df = sym_df.sort_values("date").reset_index(drop=True)
            if len(sym_df) < 20:
                continue

            cash = per_symbol_cash
            position = 0
            entry_price = 0.0
            shares = 0
            equity_list = []
            trade_records = []

            for i in range(len(sym_df)):
                row = sym_df.iloc[i]
                close = row["close"]
                date = row["date"]

                if position == 1:
                    equity = cash + shares * close
                elif position == -1:
                    equity = cash + shares * (entry_price - close)
                else:
                    equity = cash

                stop_pct = cfg.stop_loss_pct
                if position == 1 and close < entry_price * (1 - stop_pct):
                    trade_records.append(
                        {
                            "date": date,
                            "side": "stop_loss_long",
                            "price": close,
                            "shares": shares,
                        }
                    )
                    cash += shares * close * (1 - cost_rate)
                    position = 0
                    shares = 0

                elif position == -1 and close > entry_price * (1 + stop_pct):
                    trade_records.append(
                        {
                            "date": date,
                            "side": "stop_loss_short",
                            "price": close,
                            "shares": shares,
                        }
                    )
                    cash -= shares * close * (1 + cost_rate)
                    position = 0
                    shares = 0

                signal = PyBrokerBacktestRunner._generate_simple_signal(
                    sym_df, i, strategy_name
                )

                if signal == 1 and position != 1:
                    if position == -1:
                        cash -= shares * close * (1 + cost_rate)
                        position = 0
                        shares = 0
                    alloc = equity * position_size
                    shares = int(alloc / close) if close > 0 else 0
                    if shares > 0:
                        cash -= shares * close * (1 + cost_rate)
                        entry_price = close
                        position = 1

                elif signal == -1 and position != -1:
                    if position == 1:
                        cash += shares * close * (1 - cost_rate)
                        position = 0
                        shares = 0
                    alloc = equity * position_size
                    shares = int(alloc / close) if close > 0 else 0
                    if shares > 0:
                        cash += shares * close * (1 - cost_rate)
                        entry_price = close
                        position = -1

                if position == 1:
                    equity = cash + shares * close
                elif position == -1:
                    equity = cash + shares * (entry_price - close)
                else:
                    equity = cash

                equity_list.append({"date": date, "equity": equity})

            all_equities.append(pd.DataFrame(equity_list))
            tdf = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()
            if not tdf.empty:
                tdf["symbol"] = symbol
            all_trades.append(tdf)

        if not all_equities:
            return PyBrokerResult(
                metrics={"error": "no_data"},
                equity_curve=pd.DataFrame(columns=["date", "equity"]),
                trades=pd.DataFrame(),
                regime_history=regime_df,
                switch_log=pd.DataFrame(),
            )

        if len(all_equities) > 1:
            # 等权合并
            combined = all_equities[0][["date", "equity"]].copy()
            combined = combined.rename(columns={"equity": "eq_0"})
            for j, eq_df in enumerate(all_equities[1:], 1):
                merged = eq_df[["date", "equity"]].rename(columns={"equity": f"eq_{j}"})
                combined = pd.merge(combined, merged, on="date", how="outer")
            eq_cols = [c for c in combined.columns if c.startswith("eq_")]
            combined["equity"] = combined[eq_cols].fillna(method="ffill").sum(axis=1)
            combined_eq = combined[["date", "equity"]].fillna(method="ffill")
        else:
            combined_eq = all_equities[0][["date", "equity"]]

        trades_df = (
            pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        )

        if len(combined_eq) > 1:
            daily_ret = combined_eq["equity"].pct_change().dropna()
            metrics = PyBrokerBacktestRunner._compute_simple_metrics(
                combined_eq["equity"], daily_ret
            )
        else:
            metrics = {"error": "insufficient_data"}

        return PyBrokerResult(
            metrics=metrics,
            equity_curve=combined_eq,
            trades=trades_df,
            regime_history=regime_df,
            switch_log=pd.DataFrame(),
        )
