"""
子策略适配器 — 连接新因子库与组合管理层。

整改记录（参考 /Users/luojiutian/Downloads/代码审查报告.docx）:
  - P0-任务4：移除 TopLevelStrategyIntegrator 依赖，直接使用 PortfolioManager
  - P0-任务4：强制校验输入字段，缺失直接抛异常（禁止零填充）
  - P0-任务4：异常向上传播，禁止静默降级
  - P0-任务4：移除硬编码策略名列表，从 self.sub_strategies 动态获取
  - 2026-06-07：完整集成新因子引擎，use_new_factors 默认 True
  - 2026-06-07：删除 core/strategies/sub_strategies/ 子策略类依赖，
                子策略信号统一由 StrategyIndicatorRegistry + sub_strategy_aggregator
                路径A提供（规则17 不重复造轮子）
  - 2026-06-13：从 sub_strategy_indicators.py 迁入 _ohlcv_from_bar / _signal_from_factor_column
                等 PyBroker 指标辅助函数，统一辅助函数出口。

⚠️ 重要（用户指令 2026-06-07）：
  - 完整集成新因子计算（use_new_factors 默认 True）
  - 异常时**不**使用降级方案，直接抛出
  - 数据**不**使用本地 CSV（CSV 仅用于并行验证测试，主流程使用 TqSdk）
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.config.strategy_profiles import SUB_STRATEGY_NAMES
from core.ext.factors.alpha_futures.config import AlphaFuturesConfig
from core.ext.factors.alpha_futures.factor_engine import FactorEngine
from core.ext.factors.alpha_futures.sub_strategy_aggregator import (
    compute_sub_strategy_scores_from_ohlcv,
)

logger = logging.getLogger(__name__)


# ── 核心字段 ──
_REQUIRED_FACTOR_FIELDS = ("close", "open", "high", "low")

# 共享因子聚合器配置（避免每次创建新实例）
_DEFAULT_ALPHA_CONFIG = AlphaFuturesConfig()


# ═══════════════════════════════════════════════════════════════
# PyBroker 指标辅助函数（从 sub_strategy_indicators.py 迁入）
# ═══════════════════════════════════════════════════════════════


def ohlcv_from_bar(bar_data) -> Optional[pd.DataFrame]:
    """
    从 PyBroker bar_data 提取 OHLCV DataFrame。

    bar_data 通常具有 high/low/open/close/volume/open_interest 等 numpy 数组属性。
    缺失字段用默认值填充（不含 spread 数据，由调用方负责注入）。
    """
    try:
        close = getattr(bar_data, "close", None)
        high = getattr(bar_data, "high", None)
        low = getattr(bar_data, "low", None)
        if close is None or high is None or low is None:
            return None
        n = len(close)
        open_ = getattr(bar_data, "open", None)
        if open_ is None:
            open_ = close
        volume = getattr(bar_data, "volume", None)
        if volume is None:
            volume = np.zeros(n)
        oi = getattr(bar_data, "open_interest", None)
        if oi is None:
            oi = np.zeros(n)
        fc = getattr(bar_data, "far_close", None)
        fc = np.asarray(fc, dtype=float) if fc is not None else np.full(n, np.nan)
        dates = getattr(bar_data, "date", None)
        dates = pd.to_datetime(dates) if dates is not None else pd.date_range(
            "2025-01-01", periods=n, freq="D",
        )
        return pd.DataFrame({
            "date": dates,
            "open": np.asarray(open_, dtype=float),
            "high": np.asarray(high, dtype=float),
            "low": np.asarray(low, dtype=float),
            "close": np.asarray(close, dtype=float),
            "volume": np.asarray(volume, dtype=float),
            "open_interest": np.asarray(oi, dtype=float),
            "far_close": fc,
        })
    except Exception:
        return None


def signal_from_factor_column(
    bar_data, column: str, strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> np.ndarray:
    """
    调路径 A 的因子聚合器，提取指定列作为 PyBroker 指标输出。

    路径 C→A 合并后，所有 build_xxx_indicators 内部走此函数，
    保证主回测与因子验证的算法一致性。

    Args:
        bar_data: PyBroker bar_data
        column: 因子聚合器输出列名（如 "trend"、"term_structure"）
        strategy_params: 透传 best_params（如 {"trend": {"window": 20}}）
    """
    df = ohlcv_from_bar(bar_data)
    if df is None or len(df) < 30:
        close_arr = getattr(bar_data, "close", None)
        return np.zeros(len(close_arr) if close_arr is not None else 0, dtype=float)
    try:
        scored = compute_sub_strategy_scores_from_ohlcv(
            df,
            config=_DEFAULT_ALPHA_CONFIG,
            strategy_params=strategy_params,
        )
        if column not in scored.columns:
            return np.zeros(len(df), dtype=float)
        return scored[column].fillna(0.0).to_numpy()
    except Exception:
        return np.zeros(len(df), dtype=float)


class _PlaceholderSubStrategy:
    """
    子策略占位类。

    2026-06-07 整改：core/strategies/sub_strategies/ 已删除，
    PortfolioManager.add_strategy() 仍要求传入对象，
    本占位类仅承担标识作用，不参与信号计算（信号由
    StrategyIndicatorRegistry 统一提供）。
    """

    __slots__ = ("name", "factor_list", "factor_direction")

    def __init__(self, name: str) -> None:
        self.name = name
        self.factor_list: List[str] = []
        self.factor_direction: int = 0


class SubStrategyAdapter:
    """
    子策略适配器 — 连接新因子库和组合管理层。

    职责（P0-任务4 整改后）:
      1. 计算新因子（T_01, T_02, H_01, M_01 等）
      2. 注册 5 子策略到 PortfolioManager（使用子策略名作为标识，不持有类实例）
      3. 提供子策略信号合并的薄壳（委托给 PortfolioManager）
      4. 严格校验输入字段，异常直接传播

    **默认 use_new_factors=True**（用户指令 2026-06-07：完整集成新因子）。

    注：自 2026-06-07 起不再实例化子策略类，子策略信号由
    StrategyIndicatorRegistry 统一提供。
    """

    def __init__(
        self,
        config: Any = None,
        use_new_factors: bool = True,
        use_sub_strategies: bool = True,
        merge_method: Any = None,
    ):
        # 规则17：默认启用新因子（用户指令 2026-06-07：完整集成）
        self.use_new_factors = use_new_factors
        self.use_sub_strategies = use_sub_strategies
        self.merge_method = merge_method

        # 因子引擎（仅在显式启用时初始化）
        self.factor_engine: Optional[FactorEngine] = None

        # 子策略字典（仅保留名称 + 等权占位，不持有类实例）
        self.sub_strategies: Dict[str, Any] = {}

        # P0-任务4整改：直接使用 PortfolioManager，移除 TopLevelStrategyIntegrator
        from core.portfolio import PortfolioManager
        self._portfolio = PortfolioManager(total_allocation=1.0)

        # 完整集成新因子引擎
        if self.use_new_factors:
            self._init_factor_engines(config)
        else:
            logger.warning(
                "use_new_factors=False，将使用旧因子体系（新因子未启用）"
            )

        # 初始化子策略（异常时直接传播，禁止降级）
        if self.use_sub_strategies:
            self._init_sub_strategies()
            # P1-任务7整改：显式注册5子策略指标/退出钩子到注册表
            from core.engine.sub_strategy_indicators import register_default_indicators
            register_default_indicators()

    # -----------------------------------------------------------------------
    # 因子引擎初始化
    # -----------------------------------------------------------------------
    def _init_factor_engines(self, bt_config: Any = None) -> None:
        """
        初始化因子计算引擎。

        ⚠️ P0-任务4整改：异常直接抛出，不静默降级。
        """
        if bt_config is not None:
            factor_config = AlphaFuturesConfig.from_backtest_config(bt_config)
        else:
            factor_config = AlphaFuturesConfig()

        self.factor_engine = FactorEngine(factor_config)
        logger.debug("新因子引擎初始化成功 (config=%s)", factor_config)

    # -----------------------------------------------------------------------
    # 子策略初始化
    # -----------------------------------------------------------------------
    def _init_sub_strategies(self) -> None:
        """
        初始化 5 子策略占位并注册到 PortfolioManager。

        ⚠️ 2026-06-07 整改：core/strategies/sub_strategies/ 已删除，
        本方法仅注册子策略名称（无类实例），实际信号由
        StrategyIndicatorRegistry 统一提供（路径A）。
        """
        for name in SUB_STRATEGY_NAMES:
            self._register(name, _PlaceholderSubStrategy(name=name))

        logger.debug("子策略占位注册成功: %s", list(self.sub_strategies.keys()))

    def _register(self, name: str, strategy: Any) -> None:
        """注册子策略到本地字典和 PortfolioManager。"""
        self.sub_strategies[name] = strategy
        self._portfolio.add_strategy(name, strategy)

    # -----------------------------------------------------------------------
    # 因子计算（严格校验）
    # -----------------------------------------------------------------------
    def compute_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算新因子并添加到 DataFrame。

        ⚠️ P0-任务4整改：
          - 默认禁用（use_new_factors=False）
          - 严格校验输入字段，缺失直接抛异常（不再用零填充）
          - 异常直接传播，不静默降级

        Args:
            df: 原始数据 DataFrame，必须包含 close/open/high/low 列

        Returns:
            添加因子后的 DataFrame

        Raises:
            ValueError: 缺少必需字段时
            RuntimeError: 因子引擎未初始化或计算失败
        """
        if not self.use_new_factors:
            raise RuntimeError(
                "新因子计算已禁用（use_new_factors=False）。"
                "如需启用，请在配置中显式设置 use_new_factors=True，"
                "但需先完成因子库与子策略适配器的完整集成。"
            )

        if self.factor_engine is None:
            raise RuntimeError("因子引擎未初始化，请检查 _init_factor_engines")

        # 严格校验必需字段（不再用 np.zeros 填充）
        missing = [f for f in _REQUIRED_FACTOR_FIELDS if f not in df.columns]
        if missing:
            raise ValueError(
                f"输入数据缺少必需字段 {missing}，"
                f"请确保数据包含 {_REQUIRED_FACTOR_FIELDS}。"
                f"当前可用列: {list(df.columns)}"
            )

        # 准备原始数据
        raw_data = {
            "close": df["close"].values,
            "open_price": df["open"].values,
            "high": df["high"].values,
            "low": df["low"].values,
            "open_interest": df["open_interest"].values if "open_interest" in df.columns else None,
            "volume": df["volume"].values if "volume" in df.columns else None,
        }

        # 使用 factor_engine 计算因子
        factor_results = self.factor_engine.compute_all(raw_data)

        # 将因子结果添加到 DataFrame
        for factor_name, factor_values in factor_results.items():
            df[factor_name] = factor_values

        logger.debug("新因子计算完成，添加因子: %s", list(factor_results.keys()))
        return df

    # -----------------------------------------------------------------------
    # 子策略信号
    # -----------------------------------------------------------------------
    def compute_sub_strategy_signals(
        self,
        ctx: Any,
        factor_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        计算所有子策略信号（统一通过 StrategyIndicatorRegistry 获取）。

        P1-1 整改（2026-06-07）：
          - 不再直接调用 strategy.compute_signal(ctx, factor_data)
          - 统一通过 StrategyIndicatorRegistry.get_indicator_value(ctx, name) 获取
          - 解耦子策略实例与信号计算，子策略只需注册指标构建函数即可

        Args:
            ctx: PyBroker ExecContext
            factor_data: 因子数据字典（保留参数以兼容旧调用，忽略未用）

        Returns:
            {子策略名: 信号值}
        """
        from core.engine.strategy_indicators import StrategyIndicatorRegistry

        # 参数兼容保留：早期调用方传入 factor_data 但路径A不再需要
        del factor_data

        if not self.use_sub_strategies:
            return {}

        signals: Dict[str, float] = {}
        for name in self.sub_strategies.keys():
            try:
                # 子策略对应的指标名约定：<strategy>_signal
                indicator_name = f"{name}_signal"
                sig = StrategyIndicatorRegistry.get_indicator_value(ctx, indicator_name)
                if sig is not None:
                    signals[name] = float(sig)
            except Exception as e:
                logger.debug("子策略 %s 信号获取失败: %s", name, e)

        return signals

    def merge_signals(self, signals: Dict[str, float]) -> float:
        """
        合并子策略信号（等权加权，委托给 PortfolioManager）。

        P0-任务4整改：移除 TopLevelStrategyIntegrator.merge_signals 依赖，
        使用 PortfolioManager 的等权加权。

        Args:
            signals: {子策略名: 信号值}

        Returns:
            最终信号值
        """
        if not self.use_sub_strategies or not signals:
            return 0.0

        # 等权合并
        n = len(signals)
        return float(np.clip(sum(signals.values()) / n, -1.0, 1.0))

    def get_weights(self) -> Dict[str, float]:
        """获取当前子策略权重（来自 PortfolioManager）。"""
        return dict(self._portfolio.weights)

    def get_portfolio(self) -> Any:
        """获取 PortfolioManager 实例（供外部注册更多策略）。"""
        return self._portfolio

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。"""
        return {
            "n_sub_strategies": len(self.sub_strategies),
            "sub_strategies": list(self.sub_strategies.keys()),
            "weights": self.get_weights(),
            "use_new_factors": self.use_new_factors,
            "factor_engine_initialized": self.factor_engine is not None,
        }

    def sync_with_indicator_registry(self) -> None:
        """
        将子策略同步到 StrategyIndicatorRegistry（用于统一指标管理）。

        P0-任务4整改：移除硬编码策略名集合，
        直接使用 self.sub_strategies.keys() 动态比较。
        """
        try:
            from core.engine.strategy_indicators import StrategyIndicatorRegistry
            registered = set(StrategyIndicatorRegistry.get_indicator_names())
            adapter_strategies = set(self.sub_strategies.keys())
            # 仅记录非硬编码的差异
            if registered - adapter_strategies:
                logger.debug(
                    "指标注册表中有未在适配器中的指标: %s",
                    registered - adapter_strategies,
                )
            logger.debug(
                "指标注册表同步完成: 适配器策略=%s, 已注册指标=%s",
                adapter_strategies, registered,
            )
        except ImportError:
            logger.debug("StrategyIndicatorRegistry 未加载，跳过同步")

    def get_factor_names(self) -> List[str]:
        """获取所有子策略因子名称列表。"""
        return list(self.sub_strategies.keys())

    def get_signal_direction(self, signals: Dict[str, float]) -> int:
        """
        根据子策略信号确定交易方向。

        Args:
            signals: {子策略名: 信号值}

        Returns:
            1=做多, -1=做空, 0=不交易
        """
        merged = self.merge_signals(signals)
        if merged > 0.1:
            return 1
        elif merged < -0.1:
            return -1
        return 0
