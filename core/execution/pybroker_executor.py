"""
PyBroker 执行器构建器（蓝图实现，P0-1/P0-2/P0-3 整改）。

位置: core/engine/pybroker_executor.py

替代旧的 `core/engine/strategy_executor.py:StrategyExecutorFactory`，
实现用户指定的蓝图模式：
    1. 调仓日收集所有品种的因子得分（每个品种调用一次）
    2. 收齐后 finalize 横截面（最后一个品种触发）
    3. 计算综合得分
    4. 通过 PortfolioManager 分配目标权重
    5. 通过 RiskController 调整（集中度等）
    6. 单品种执行下单

PyBroker 的限制：
    - executor 函数按 (symbol, bar) 顺序调用，逐品种独立
    - 横截面数据需要共享状态：本类持有 SharedState
    - 在每个调仓日，第一个品种开始收集，最后一个品种触发 finalize

P0-3 整改：完全使用 RiskController（而非旧的 RiskManagerAdapter）。

改进（2026-06-13）：
  - 横截面状态管理改用 set 去重，避免重复调用导致计数溢出
  - 统一风险预算逻辑（risk_per_trade + target_vol 波动率平价）
  - 止损使用 state 显式追踪 entry_price / entry_bar

CTA 退出策略注入（2026-06-13）：
  - 通过 cta_exit_policy 参数注入 CTAExitPolicy，启用四层退出 + 风险预算
  - 未注入时保持原有 RiskController 行为
  - 实现单品种 CTA 策略与多品种横截面策略共用同一执行器
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import BacktestConfig
from core.engine.switch_engine import FactorScoringEngine
from core.execution.cta_exit_policy import CTAExitPolicy
from core.execution.signal_abstraction import SignalAbstractionLayer
from core.portfolio import PortfolioManager
from core.risk_controller import RiskController

_logger = logging.getLogger(__name__)

try:
    from pybroker import ExecContext

    PYBROKER_AVAILABLE = True
except ImportError:
    PYBROKER_AVAILABLE = False
    ExecContext = Any  # type: ignore


# ────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────
def _get_indicator(ctx: Any, name: str) -> Optional[float]:
    """从 PyBroker ctx 安全获取指标值。"""
    try:
        val = ctx.indicator(name)
        if val is None:
            return None
        if hasattr(val, "iloc") and len(val) > 0:
            return float(val.iloc[-1])
        if hasattr(val, "__getitem__") and len(val) > 0:
            return float(val[-1])
        return float(val)
    except (ValueError, KeyError):
        return None
    except Exception as e:  # noqa: BLE001
        _logger.debug("获取指标 %s 异常: %s", name, e)
        return None


def _get_close(ctx: Any) -> Optional[float]:
    """安全获取当前收盘价。"""
    try:
        close = ctx.close
        if hasattr(close, "__getitem__") and len(close) > 0:
            return float(close[-1])
        if close is not None:
            return float(close)
    except Exception as e:  # noqa: BLE001
        _logger.debug("获取收盘价异常: %s", e)
    return None


def _get_atr(ctx: Any) -> float:
    """获取 ATR 指标，若不可用返回 0。"""
    atr = _get_indicator(ctx, "atr_14")
    return float(atr) if atr is not None and atr > 0 else 0.0


def _get_pos_shares(ctx: Any, side: str) -> int:
    """获取当前品种的多/空持仓数量。"""
    try:
        pos = ctx.pos(ctx.symbol, side)
        if pos is not None:
            return int(getattr(pos, "shares", 0))
    except Exception:  # noqa: BLE001
        pass
    return 0


def _ohlcv_from_ctx(ctx: Any) -> Optional[pd.DataFrame]:
    """从 PyBroker ExecContext 提取 OHLCV DataFrame。

    用于 UnifiedFactorPool 的信号计算入口。

    Returns:
        DataFrame，含 date, open, high, low, close, volume(可选) 列
    """
    close = getattr(ctx, "close", None)
    if close is None or len(close) < 10:
        return None
    n = len(close)
    try:
        import pandas as pd

        dates = getattr(ctx, "date", None)
        if dates is None:
            # 回退：用索引模拟日期
            dates = pd.date_range(end=pd.Timestamp(ctx.dt), periods=n)
        result = {
            "date": dates
            if hasattr(dates, "__len__") and len(dates) == n
            else pd.date_range(end=pd.Timestamp(ctx.dt), periods=n),
            "open": getattr(ctx, "open", close),
            "high": getattr(ctx, "high", close),
            "low": getattr(ctx, "low", close),
            "close": close,
        }
        vol = getattr(ctx, "volume", None)
        if vol is not None:
            result["volume"] = vol
        df = pd.DataFrame(result)
        if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df["date"]
        ):
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:  # noqa: BLE001
        return None


# ────────────────────────────────────────────────────────────
# 共享状态（蓝图数据 + 预计算数据）
# ────────────────────────────────────────────────────────────
@dataclass
class PyBrokerExecutorSharedState:
    """
    PyBroker 执行器共享状态。

    PyBroker 按 (date, symbol) 顺序调用 executor，
    横截面数据需要在所有品种间共享。

    2026-06-14 改进：移除了 collected_symbols_set依赖，
    改用预计算信号 + 时间驱动触发 finalize。
    """

    total_symbols: int
    # 当前调仓日
    rebalance_date: Any = None
    finalized: bool = False
    # 当前调仓日已收集的品种（蓝图模式使用；预计算模式不依赖）
    collected_symbols_set: set = field(default_factory=set)
    # 当前调仓日计算出的目标权重（finalize 之后才填充）
    target_weights: Dict[str, float] = field(default_factory=dict)
    # 上一bar 数据（用于滚动IC更新）
    prev_close: Dict[str, float] = field(default_factory=dict)
    prev_factor_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # 历史调仓日权重
    last_weights: Dict[str, float] = field(default_factory=dict)
    # 方向二 dynamic 模式下每品种的仓位缩放系数 [0, 1]
    dynamic_pos_scales: Dict[str, float] = field(default_factory=dict)

    # ── 止损追踪（显式 tracking，不用反向推算） ──
    entry_price: Dict[str, float] = field(default_factory=dict)
    entry_bar: Dict[str, int] = field(default_factory=dict)
    entry_direction: Dict[str, str] = field(default_factory=dict)  # "long" or "short"
    highest_since_entry: Dict[str, float] = field(default_factory=dict)
    lowest_since_entry: Dict[str, float] = field(default_factory=dict)

    # ── CTA 退出策略持仓状态 ──
    pos_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────
# 执行器构建器
# ────────────────────────────────────────────────────────────
class PyBrokerExecutorBuilder:
    """
    蓝图模式执行器构建器（P0-1/P0-2/P0-3 整改）。

    替代旧 StrategyExecutorFactory.create_executor()，按蓝图实现：
        for symbol in symbols:
            scores = scoring_engine.extract_factor_scores(ctx, params)
            scoring_engine.update_cross_section(symbol, scores, dt=ctx.dt)
        scoring_engine.finalize_cross_section()
        signals = {sym: scoring_engine.compute_composite_score(sym) for sym in symbols}
        target_weights = portfolio_manager.allocate_weights(signals, method='risk_parity')
        target_weights = risk_controller.adjust(target_weights, ctx)
    """

    def __init__(
        self,
        scoring_engine: FactorScoringEngine,
        portfolio_manager: PortfolioManager,
        risk_controller: RiskController,
        config: BacktestConfig,
        total_symbols: int,
        weight_method: str = "risk_parity",
        risk_estimates_provider: Optional[Callable[[str], Optional[float]]] = None,
        *,
        cta_exit_policy: Optional[CTAExitPolicy] = None,
        cta_signal_provider: Optional[Callable[[Any, str], float]] = None,
        cta_market_state_provider: Optional[Callable[[str], str]] = None,
        cta_sigma_provider: Optional[Callable[[str], Optional[float]]] = None,
        # ── 统一因子池注入（2026-06-13） ──
        signal_abstraction: Optional[SignalAbstractionLayer] = None,
        # ── Per-bar 录制开关（2026-06-16，sweep/分析专用） ──
        record_per_bar: bool = False,
    ):
        self.scoring_engine = scoring_engine
        self.portfolio_manager = portfolio_manager
        self.risk_controller = risk_controller
        self.config = config
        self.weight_method = weight_method
        self.risk_estimates_provider = risk_estimates_provider
        self.state = PyBrokerExecutorSharedState(total_symbols=total_symbols)
        # ── CTA 退出策略注入 ──
        self.cta_exit_policy = cta_exit_policy
        self.cta_signal_provider = cta_signal_provider
        self.cta_market_state_provider = cta_market_state_provider
        self.cta_sigma_provider = cta_sigma_provider
        # ── 统一因子池注入 ──
        self.signal_abstraction = signal_abstraction
        # ── 预计算信号缓存 {symbol: {bar_idx: {name: value}}} ──
        self._signal_cache: Dict[str, Dict[int, Dict[str, float]]] = {}
        # ── Per-bar 录制（sweep/分析专用） ──
        self.record_per_bar = record_per_bar
        self.per_bar_log: List[Dict[str, Any]] = []

    # ────────────────────────────────────────────────────────────
    # 预计算信号（2026-06-14 新增：替代运行时 per-bar 计算）
    # ────────────────────────────────────────────────────────────

    def precompute_signals(
        self,
        full_df: pd.DataFrame,
        strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """预计算所有品种在所有 bar 上的信号，存入 _signal_cache。

        在 PyBroker backtest() 执行前调用，避免 executor 内逐 bar 计算和
        横截面收集竞态。

        Args:
            full_df: PyBroker 格式 DataFrame（含所有品种 OHLCV）
            strategy_params: 策略参数，CTA 模式下 keys 为活跃策略名
        """
        if self.signal_abstraction is None:
            return  # 未注入统一因子池，跳过

        strategy_params = strategy_params or {}
        symbols = sorted(full_df["symbol"].unique().tolist())
        _logger.info(
            "预计算 %d 个品种的全部信号...",
            len(symbols),
        )
        # ── 方向三：配对交易横截面信号预计算（2026-06-17） ──
        if self.signal_abstraction.is_pair_trading_source():
            close_matrix = full_df.pivot_table(
                index="date",
                columns="symbol",
                values="close",
                aggfunc="last",
            ).sort_index()
            # 保持列顺序与 self._signal_cache 一致
            for s in symbols:
                if s not in close_matrix.columns:
                    close_matrix[s] = np.nan
            close_matrix = close_matrix[symbols].ffill()
            self.signal_abstraction.precompute_pair_signals(close_matrix)
        for sym in symbols:
            sym_df = full_df[full_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_df) < 30:
                continue
            signal_df = self.signal_abstraction.pool.compute_all(
                sym_df,
                sym,
                strategy_params,
            )
            cache: Dict[int, Dict[str, float]] = {}
            for bar_idx in range(len(signal_df)):
                row = signal_df.iloc[bar_idx]
                signals = {}
                for col in signal_df.columns:
                    val = row[col]
                    if isinstance(val, (int, float)) and not (val != val):  # NaN check
                        signals[col] = float(val)
                if signals:
                    cache[bar_idx] = signals
            # CTA / HYBRID 模式：预计算 CTA 合成信号
            if self.signal_abstraction.mode in ("cta", "hybrid"):
                from core.execution.factor_pool import CTA_SIGNAL_NAMES
                from core.execution.signal_abstraction import DEFAULT_CTA_WEIGHTS

                active_cta = [
                    k for k in strategy_params if k in CTA_SIGNAL_NAMES
                ] or CTA_SIGNAL_NAMES
                # 方向四 P1：优先使用 signal_layer 自定义权重（2026-06-17）
                custom_w = getattr(self.signal_abstraction, "cta_composite_weights", None)
                # 2026-06-19：四因子 CTA 融合——若 strategy_params 中含 "four_factor"，
                # 强制使用四因子权重（覆盖 custom_w），确保 receipt_change / basis_momentum
                # 真正参与合成。
                if "four_factor" in strategy_params:
                    four_factor_w = (
                        self.signal_abstraction.DEFAULT_FOUR_FACTOR_WEIGHTS
                        if hasattr(self.signal_abstraction, "DEFAULT_FOUR_FACTOR_WEIGHTS")
                        else {
                            "donchian_breakout": 0.30,
                            "carry": 0.25,
                            "basis_momentum": 0.25,
                            "receipt_change": 0.20,
                        }
                    )
                    # 允许从 strategy_params["four_factor"]["weights"] 覆盖
                    ff_param_weights = strategy_params["four_factor"].get("weights", {})
                    four_factor_w = {**four_factor_w, **ff_param_weights}
                    # 限制 active_cta 为四因子相关信号
                    four_factor_keys = [
                        "donchian_breakout", "carry", "basis_momentum", "receipt_change",
                    ]
                    active_cta = [k for k in four_factor_keys if k in CTA_SIGNAL_NAMES]
                    total_w = sum(four_factor_w.get(k, 0.0) for k in active_cta)
                    if total_w > 1e-8:
                        weights = {k: four_factor_w.get(k, 0.0) / total_w for k in active_cta}
                    else:
                        weights = {k: 1.0 / len(active_cta) for k in active_cta}
                elif custom_w is not None:
                    total_w = sum(custom_w.get(k, 0.0) for k in active_cta)
                    if total_w > 1e-8:
                        weights = {k: custom_w.get(k, 0.0) / total_w for k in active_cta}
                    else:
                        weights = {k: 1.0 / len(active_cta) for k in active_cta}
                else:
                    # 默认 DEFAULT_CTA_WEIGHTS（归一化到 active 子集）
                    total_w = sum(DEFAULT_CTA_WEIGHTS.get(k, 0.0) for k in active_cta)
                    if total_w > 1e-8:
                        weights = {
                            k: DEFAULT_CTA_WEIGHTS.get(k, 0.0) / total_w
                            for k in active_cta
                        }
                    else:
                        weights = {k: 1.0 / len(active_cta) for k in active_cta}
                for bar_idx in list(cache.keys()):
                    signals = cache[bar_idx]
                    cta_val = 0.0
                    for cname in active_cta:
                        if cname in signals:
                            cta_val += weights[cname] * signals[cname]
                    if active_cta:
                        cache[bar_idx]["_cta_composite"] = float(
                            np.clip(cta_val, -1.0, 1.0)
                        )
            self._signal_cache[sym] = cache
        _logger.info("预计算完成: %d 个品种", len(self._signal_cache))

    def build(
        self,
        strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Callable[[ExecContext], None]:
        """构建 PyBroker executor 函数。

        支持两种模式：
          1. 蓝图模式（默认）：横截面收集 → finalize → PortfolioManager → 下单
          2. CTA 模式（注入 cta_exit_policy）：每 bar 信号 → 四层退出 → 风险预算 → 下单
        """
        _logger.info(
            "PyBrokerExecutorBuilder.build() called, record_per_bar=%s",
            self.record_per_bar,
        )
        # 2026-06-20：移除冗余的 sys.stderr.write（与上面 _logger.info 重复）
        scoring_engine = self.scoring_engine
        portfolio_manager = self.portfolio_manager
        risk_controller = self.risk_controller
        config = self.config
        state = self.state
        weight_method = self.weight_method
        risk_estimates_provider = self.risk_estimates_provider
        strategy_params = strategy_params or {}
        position_size = config.max_position_pct
        entry_threshold = config.entry_threshold
        exit_policy = self.cta_exit_policy
        signal_provider = self.cta_signal_provider
        market_state_provider = self.cta_market_state_provider
        sigma_provider = self.cta_sigma_provider
        cta_mode = exit_policy is not None
        # ── 统一因子池模式（2026-06-13） ──
        signal_layer = self.signal_abstraction

        def executor_fn(ctx: ExecContext) -> None:
            symbol = ctx.symbol
            current_close = _get_close(ctx)
            current_date = ctx.dt
            current_bar = len(getattr(ctx, "close", []))

            # 0) 更新止损追踪数据（持仓中品种）
            if symbol in state.entry_price:
                if current_close is not None:
                    state.highest_since_entry[symbol] = max(
                        state.highest_since_entry.get(symbol, current_close),
                        current_close,
                    )
                    state.lowest_since_entry[symbol] = min(
                        state.lowest_since_entry.get(symbol, current_close),
                        current_close,
                    )

            # ─────────────────────────────────────────────────────
            # CTA 模式（统一因子池）：合成信号 → 直接执行
            # ─────────────────────────────────────────────────────
            if signal_layer is not None and signal_layer.mode == "cta":
                sym_cache = self._signal_cache.get(symbol, {})
                precomputed = sym_cache.get(current_bar, None)
                cta_signal = (
                    precomputed.get("_cta_composite", 0.0) if precomputed else 0.0
                )

                if abs(cta_signal) >= entry_threshold:
                    target_dir = 1 if cta_signal > 0 else -1
                    effective_size = position_size
                    has_long = _get_pos_shares(ctx, "long") > 0
                    has_short = _get_pos_shares(ctx, "short") > 0
                    self._execute_rebalance(
                        ctx, target_dir, effective_size, has_long, has_short
                    )
                else:
                    self._close_all(ctx)
                    self._clear_entry_state(symbol)
                return

            # ─────────────────────────────────────────────────────
            # 统一因子池模式：预计算信号 + 时间驱动横截面
            # ─────────────────────────────────────────────────────
            if signal_layer is not None:
                # 1) 从预计算缓存获取当前 bar 的信号
                sym_cache = self._signal_cache.get(symbol, {})
                precomputed = sym_cache.get(current_bar, None)
                if precomputed is None:
                    # 缓存未命中（数据不足），尝试实时计算
                    ohlcv = _ohlcv_from_ctx(ctx)
                    if ohlcv is not None and len(ohlcv) >= 10:
                        cs_signals = signal_layer.get_cross_sectional_signals(
                            symbol,
                            ohlcv,
                            len(ohlcv) - 1,
                            strategy_params,
                        )
                    else:
                        cs_signals = {}
                else:
                    # 从预计算缓存提取子策略得分
                    cs_signals = {}
                    for sname in [
                        "trend",
                        "term_structure",
                        "mean_reversion",
                        "vol_breakout",
                        "composite_resonance",
                    ]:
                        if sname in precomputed:
                            cs_signals[sname] = precomputed[sname]

                # 2) 更新横截面引擎
                scoring_engine.update_cross_section(
                    symbol,
                    cs_signals,
                    dt=current_date,
                )

                is_rebalance = scoring_engine.is_rebalance_day(current_date)

                # 3) 时间驱动：调仓日触发 finalize（基于日期变化，而非品种计数）
                if is_rebalance and state.rebalance_date != current_date:
                    state.rebalance_date = current_date
                    state.finalized = False
                    state.target_weights = {}

                    # 立即 finalize：所有品种的信号已预计算，无需等全部收集
                    scoring_engine.finalize_cross_section()
                    state.finalized = True
                    scoring_engine.mark_rebalanced(current_date)

                    all_syms = sorted(self._signal_cache.keys())
                    signals_all: Dict[str, float] = {}

                    # ── 方向三：配对交易横截面信号覆盖（2026-06-17） ──
                    # 若 signal_layer.cross_section_source=="pair_trading"，
                    # 用预计算的配对 z-score 替换默认的 cross-section composite。
                    use_pair = (
                        signal_layer is not None
                        and signal_layer.is_pair_trading_source()
                    )
                    for sym in all_syms:
                        if use_pair:
                            pair_z = signal_layer.get_pair_cross_section_scores(
                                sym,
                                current_bar,
                            )
                            if pair_z is None:
                                pair_z = 0.0
                            # 配对信号是横截面替代品 → 直接用作 cross_section_z
                            signals_all[sym] = float(np.clip(pair_z, -1.0, 1.0))
                        else:
                            signals_all[sym] = scoring_engine.compute_composite_score(
                                sym,
                            )

                    # ── HYBRID 模式：横截面得分 × CTA 时序混合 ──
                    if signal_layer.mode == "hybrid":
                        cw = signal_layer.cta_weight
                        # 方向二 dynamic 模式：收集每品种的仓位缩放系数
                        pos_scales: Dict[str, float] = {}
                        for sym in all_syms:
                            z = signals_all[sym]
                            sym_cache = self._signal_cache.get(sym, {})
                            pre = sym_cache.get(current_bar, None)
                            cta_sig = pre.get("_cta_composite", 0.0) if pre else 0.0
                            if (
                                getattr(signal_layer, "hybrid_blend_method", "linear")
                                == "dynamic"
                            ):
                                # 关键：用 signals_all[sym]（已被配对信号覆盖）作为 raw_z，
                                # 避免配对模式下被 scoring_engine 的 xs 重新覆盖。
                                raw_z = float(signals_all[sym])
                                cross_strength = float(np.clip(abs(raw_z), 0.0, 1.0))
                                pos_scale = (
                                    signal_layer.xs_position_base
                                    + (
                                        signal_layer.xs_position_ceiling
                                        - signal_layer.xs_position_base
                                    )
                                    * cross_strength
                                )
                                if cta_sig * raw_z < 0.0:
                                    pos_scale *= signal_layer.xs_opposite_penalty
                                pos_scale = float(np.clip(pos_scale, 0.0, 1.0))
                                pos_scales[sym] = pos_scale
                                hybrid = cta_sig * pos_scale
                            else:
                                # 默认：线性加权
                                hybrid = (1.0 - cw) * z + cw * cta_sig
                                pos_scales[sym] = 1.0
                            signals_all[sym] = float(np.clip(hybrid, -1.0, 1.0))
                        state.dynamic_pos_scales = pos_scales
                        # ── per-bar 录制（sweep 专用） ──
                        if self.record_per_bar:
                            self.per_bar_log.append(
                                {
                                    "bar_idx": current_bar,
                                    "date": str(current_date),
                                    "signals": dict(signals_all),
                                    "pos_scales": dict(pos_scales),
                                }
                            )

                    # 风险预算调整
                    risk_estimates: Dict[str, float] = {}
                    if risk_estimates_provider:
                        for sym in signals_all:
                            est = risk_estimates_provider(sym)
                            if est is not None and est > 0:
                                risk_estimates[sym] = est

                    target_weights = portfolio_manager.allocate_weights(
                        signals_all,
                        method=weight_method,
                        risk_estimates=risk_estimates if risk_estimates else None,
                    )
                    target_weights = risk_controller.check_concentration_dict(
                        target_weights,
                        max_concentration=config.max_position_pct,
                    )
                    # 方向二 dynamic 模式：把 pos_scale 应用到 final weights
                    # （equal_weight 模式下 allocator 只看符号，必须显式缩放）
                    if (
                        signal_layer is not None
                        and signal_layer.mode == "hybrid"
                        and getattr(signal_layer, "hybrid_blend_method", "linear")
                        == "dynamic"
                        and state.dynamic_pos_scales
                    ):
                        for sym in list(target_weights.keys()):
                            scale = state.dynamic_pos_scales.get(sym, 1.0)
                            target_weights[sym] = target_weights[sym] * scale
                    state.target_weights = target_weights
                    state.last_weights = dict(target_weights)

                # 4) 交易执行
                if not state.finalized:
                    return

                target_w = state.target_weights.get(symbol, 0.0)

                # 混合 CTA 信号过滤（仅纯横截面模式需要）
                if (
                    signal_layer is not None
                    and signal_layer.mode not in ("hybrid", "cta")
                    and precomputed is not None
                ):
                    cta_composite = (
                        precomputed.get("carry", 0.0) * 0.3
                        + precomputed.get("vol_mean_reversion", 0.0) * 0.3
                        + precomputed.get("donchian_breakout", 0.0) * 0.2
                        + precomputed.get("momentum_ma", 0.0) * 0.1
                        + precomputed.get("tsi_garch", 0.0) * 0.05
                        + precomputed.get("pair_trading", 0.0) * 0.05
                    )
                    hybrid = (1.0 - 0.5) * (
                        target_w / (abs(target_w) + 1e-10)
                    ) + 0.5 * np.clip(cta_composite, -1, 1)
                    hybrid = float(np.clip(hybrid, -1.0, 1.0))
                    if abs(hybrid) < entry_threshold:
                        target_w = 0.0

                if self._check_stop_loss(
                    ctx, current_close, symbol, current_date, current_bar
                ):
                    self._close_all(ctx)
                    self._clear_entry_state(symbol)
                    return

                if abs(target_w) < 1e-6:
                    self._close_all(ctx)
                    self._clear_entry_state(symbol)
                    return

                direction = 1 if target_w > 0 else -1
                effective_size = min(abs(target_w), position_size)
                has_long = _get_pos_shares(ctx, "long") > 0
                has_short = _get_pos_shares(ctx, "short") > 0
                old_pos = 1 if has_long else -1 if has_short else 0
                self._execute_rebalance(
                    ctx, direction, effective_size, has_long, has_short
                )

                if current_close is not None and (
                    old_pos == 0
                    or (old_pos > 0 and direction < 0)
                    or (old_pos < 0 and direction > 0)
                ):
                    dir_label = "long" if direction > 0 else "short"
                    state.entry_price[symbol] = current_close
                    state.entry_bar[symbol] = current_bar
                    state.entry_direction[symbol] = dir_label
                    state.highest_since_entry[symbol] = current_close
                    state.lowest_since_entry[symbol] = current_close
                return

            # ─────────────────────────────────────────────────────
            # CTA 模式：每 bar 执行，四层退出 + 风险预算
            # ─────────────────────────────────────────────────────
            if cta_mode:
                assert exit_policy is not None  # narrow type
                close_arr = getattr(ctx, "close", None)
                high_arr = getattr(ctx, "high", None)
                low_arr = getattr(ctx, "low", None)
                if close_arr is None or len(close_arr) < 10:
                    return

                signal = signal_provider(ctx, symbol) if signal_provider else 0.0
                market_state = (
                    market_state_provider(symbol)
                    if market_state_provider
                    else "oscillation"
                )
                sigma = sigma_provider(symbol) if sigma_provider else None

                has_long = _get_pos_shares(ctx, "long") > 0
                has_short = _get_pos_shares(ctx, "short") > 0
                current_pos = 1 if has_long else -1 if has_short else 0

                # 更新全局风控
                exit_policy.update_global_risk(ctx.dt, symbol, close_arr)

                if current_pos != 0:
                    # 更新极值
                    pstate = state.pos_state.get(symbol, {})
                    if pstate:
                        exit_policy.update_extreme(
                            pstate, current_pos, current_close or 0.0
                        )
                        state.pos_state[symbol] = pstate

                    # 四层退出
                    reason = exit_policy.check_exits(
                        symbol=symbol,
                        current_pos=current_pos,
                        current_price=current_close or 0.0,
                        current_bar=current_bar,
                        signal=signal,
                        market_state=market_state,
                        close=close_arr,
                        high=high_arr,
                        low=low_arr,
                        pos_state=pstate,
                    )
                    if reason:
                        self._close_all(ctx)
                        self._clear_entry_state(symbol)
                        state.pos_state.pop(symbol, None)
                        return

                    # 持仓中且无退出条件 → 维持
                    return

                # ── 空仓：入场逻辑 ──
                if abs(signal) < entry_threshold:
                    return

                target_w = exit_policy.compute_risk_budget_position(
                    signal=signal,
                    market_state=market_state,
                    current_price=current_close or 0.0,
                    close=close_arr,
                    high=high_arr,
                    low=low_arr,
                    sigma=sigma,
                )
                target_w = np.clip(target_w, -position_size, position_size)
                dir_label = "long" if target_w > 0 else "short"
                self._execute_rebalance(
                    ctx,
                    1 if target_w > 0 else -1,
                    abs(target_w),
                    has_long,
                    has_short,
                )
                # 记录开仓
                if current_close is not None:
                    state.entry_price[symbol] = current_close
                    state.entry_bar[symbol] = current_bar
                    state.entry_direction[symbol] = dir_label
                    state.highest_since_entry[symbol] = current_close
                    state.lowest_since_entry[symbol] = current_close
                    state.pos_state[symbol] = CTAExitPolicy.make_pos_state(
                        current_close,
                        1 if target_w > 0 else -1,
                        current_bar,
                    )
                return

            # ─────────────────────────────────────────────────────
            # 蓝图模式（默认）：多品种横截面
            # ─────────────────────────────────────────────────────

            # 1) 滚动IC
            self._update_rolling_ic(symbol, current_close)

            # 2) 提取因子得分
            factor_scores = scoring_engine.extract_factor_scores(ctx, strategy_params)
            if factor_scores:
                state.prev_factor_scores[symbol] = dict(factor_scores)
            if current_close is not None:
                state.prev_close[symbol] = current_close

            # 3) 判断是否调仓日
            is_rebalance = scoring_engine.is_rebalance_day(current_date)

            if not is_rebalance:
                # 非调仓日：仅检查止损
                if self._check_stop_loss(
                    ctx, current_close, symbol, current_date, current_bar
                ):
                    self._close_all(ctx)
                    self._clear_entry_state(symbol)
                return

            # 4) 横截面收集
            if state.rebalance_date != current_date:
                state.rebalance_date = current_date
                state.collected_symbols_set = set()
                state.finalized = False
                state.target_weights = {}

            state.collected_symbols_set.add(symbol)
            scoring_engine.update_cross_section(
                symbol,
                factor_scores,
                dt=current_date,
            )

            # 5) finalize 横截面
            if (
                not state.finalized
                and len(state.collected_symbols_set) >= state.total_symbols
            ):
                scoring_engine.finalize_cross_section()
                state.finalized = True
                scoring_engine.mark_rebalanced(current_date)

                signals: Dict[str, float] = {}
                # ── 方向三：pair signal 覆盖（2026-06-20 修复） ──
                # 当 signal_abstraction 在 pair 模式下，使用配对 z-score 而非多因子 composite
                use_pair = (
                    signal_layer is not None
                    and signal_layer.is_pair_trading_source()
                )
                for sym in state.collected_symbols_set:
                    if use_pair:
                        pair_z = signal_layer.get_pair_cross_section_scores(
                            sym, scoring_engine._pair_ctx
                        )
                        signals[sym] = float(np.clip(pair_z, -1.0, 1.0))
                    else:
                        signals[sym] = scoring_engine.compute_composite_score(sym)
                _logger.debug("调仓日 %s 综合得分: %s", current_date, signals)

                risk_estimates: Dict[str, float] = {}
                if risk_estimates_provider:
                    for sym in signals:
                        est = risk_estimates_provider(sym)
                        if est is not None and est > 0:
                            risk_estimates[sym] = est

                target_weights = portfolio_manager.allocate_weights(
                    signals,
                    method=weight_method,
                    risk_estimates=risk_estimates if risk_estimates else None,
                )
                target_weights = risk_controller.check_concentration_dict(
                    target_weights,
                    max_concentration=config.max_position_pct,
                )
                state.target_weights = target_weights
                state.last_weights = dict(target_weights)

            if not state.finalized:
                return

            target_w = state.target_weights.get(symbol, 0.0)
            # ── 方向三：pair signal 阈值判断（2026-06-20 修复） ──
            # 当 signal_abstraction 在 pair 模式下，target_weights 已由 pair z-score 推导
            # （line 818 portfolio_manager.allocate_weights），使用 target_w 自身的归一化幅度
            # 作为 entry threshold 判断，避免再次调用多因子 composite_score 覆盖 pair 信号。
            use_pair_threshold = (
                signal_layer is not None
                and signal_layer.is_pair_trading_source()
            )
            if use_pair_threshold:
                score = abs(target_w)
            else:
                score = scoring_engine.compute_composite_score(symbol)
            if abs(score) < entry_threshold:
                target_w = 0.0

            if self._check_stop_loss(
                ctx, current_close, symbol, current_date, current_bar
            ):
                self._close_all(ctx)
                self._clear_entry_state(symbol)
                return

            if abs(target_w) < 1e-6:
                self._close_all(ctx)
                self._clear_entry_state(symbol)
                return

            direction = 1 if target_w > 0 else -1
            effective_size = min(abs(target_w), position_size)
            has_long = _get_pos_shares(ctx, "long") > 0
            has_short = _get_pos_shares(ctx, "short") > 0
            old_pos = 1 if has_long else -1 if has_short else 0
            self._execute_rebalance(ctx, direction, effective_size, has_long, has_short)

            if current_close is not None and (
                old_pos == 0
                or (old_pos > 0 and direction < 0)
                or (old_pos < 0 and direction > 0)
            ):
                dir_label = "long" if direction > 0 else "short"
                state.entry_price[symbol] = current_close
                state.entry_bar[symbol] = current_bar
                state.entry_direction[symbol] = dir_label
                state.highest_since_entry[symbol] = current_close
                state.lowest_since_entry[symbol] = current_close

        executor_fn.__name__ = "blueprint_executor"
        return executor_fn

    # ────────────────────────────────────────────────────────────
    # 内部方法
    # ────────────────────────────────────────────────────────────
    def _update_rolling_ic(self, symbol: str, current_close: Optional[float]) -> None:
        """用上一bar因子得分 + 当前bar收益更新滚动IC引擎。"""
        return

    def _check_stop_loss(
        self,
        ctx: ExecContext,
        current_close: Optional[float],
        symbol: str,
        current_date: Any,
        current_bar: int,
    ) -> bool:
        """复合止损检查。

        使用 state 中显式追踪的 entry_price/highest/lowest，
        优先走 RiskController 复合止损路径。
        """
        if current_close is None or current_close <= 0:
            return False

        entry_p = self.state.entry_price.get(symbol)
        if entry_p is None or entry_p <= 0:
            return False

        entry_bar = self.state.entry_bar.get(symbol, current_bar)
        direction = self.state.entry_direction.get(symbol, "long")  # 用存储的方向
        highest = self.state.highest_since_entry.get(symbol, current_close)
        lowest = self.state.lowest_since_entry.get(symbol, current_close)
        atr = _get_atr(ctx)

        if (
            self.risk_controller is not None
            and self.risk_controller.composite_stop is not None
        ):
            try:
                result = self.risk_controller.check_composite_stop(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_p,
                    current_price=current_close,
                    highest_since_entry=highest,
                    lowest_since_entry=lowest,
                    entry_day=entry_bar,
                    current_day=current_bar,
                    atr_value=atr if atr > 0 else None,
                    auto_register_entry=False,
                )
                if result.triggered:
                    _logger.info(
                        "%s 触发%s: %s",
                        symbol,
                        direction,
                        result.trigger_reason,
                    )
                    return True
            except Exception as e:
                _logger.debug("复合止损异常: %s", e)

        # 回退：固定百分比止损
        pnl_pct = (current_close - entry_p) / entry_p
        stop_pct = self.config.stop_loss_pct
        if direction == "long" and pnl_pct < -stop_pct:
            _logger.info("%s 触发固定止损: pnl=%.2f%%", symbol, pnl_pct * 100)
            return True
        if direction == "short" and pnl_pct > stop_pct:
            _logger.info("%s 触发固定止损: pnl=%.2f%%", symbol, pnl_pct * 100)
            return True

        return False

    def _clear_entry_state(self, symbol: str) -> None:
        """清除品种的止损追踪状态。"""
        self.state.entry_price.pop(symbol, None)
        self.state.entry_bar.pop(symbol, None)
        self.state.entry_direction.pop(symbol, None)
        self.state.highest_since_entry.pop(symbol, None)
        self.state.lowest_since_entry.pop(symbol, None)

    @staticmethod
    def _execute_rebalance(
        ctx: ExecContext,
        direction: int,
        effective_size: float,
        has_long: bool,
        has_short: bool,
    ) -> None:
        """执行调仓。"""
        if direction > 0 and not has_long:
            if has_short:
                ctx.cover_all_shares()
            ctx.buy_shares = ctx.calc_target_shares(effective_size)
        elif direction < 0 and not has_short:
            if has_long:
                ctx.sell_all_shares()
            ctx.sell_shares = ctx.calc_target_shares(effective_size)
        elif direction == 0:
            if has_long:
                ctx.sell_all_shares()
            if has_short:
                ctx.cover_all_shares()

    @staticmethod
    def _close_all(ctx: ExecContext) -> None:
        if ctx.pos(ctx.symbol, "long") is not None:
            ctx.sell_all_shares()
        if ctx.pos(ctx.symbol, "short") is not None:
            ctx.cover_all_shares()


# 给 RiskController 加一个 dict 版的集中度检查（避免污染核心类）
def _patch_risk_controller_for_blueprint() -> None:
    """为 RiskController 添加 check_concentration_dict 便捷方法。"""
    if hasattr(RiskController, "check_concentration_dict"):
        return

    def check_concentration_dict(
        self,
        weights: Dict[str, float],
        max_concentration: float = 0.4,
    ) -> Dict[str, float]:
        if not weights:
            return {}
        result: Dict[str, float] = {}
        for sym, w in weights.items():
            if abs(w) > max_concentration:
                result[sym] = max_concentration * (1.0 if w > 0 else -1.0)
            else:
                result[sym] = float(w)
        return result

    RiskController.check_concentration_dict = check_concentration_dict  # type: ignore[attr-defined]


_patch_risk_controller_for_blueprint()
