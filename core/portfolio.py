"""
组合管理模块。

管理多策略组合，根据因子打分调仓模式协调策略执行。
PyBroker 原生支持通过 strategy.add_execution() 添加多个执行函数，
本模块在此基础上增加：
- 因子权重管理
- 多策略资金分配
- 组合级别的绩效汇总
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Callable

from core.config import DEFAULT_FACTOR_WEIGHTS

_logger = logging.getLogger(__name__)


class WeightMethod(Enum):
    """权重分配方法枚举。"""

    EQUAL_WEIGHT = "equal_weight"          # 等权
    RISK_PARITY = "risk_parity"            # 风险倒数等权
    SCORE_WEIGHTED = "score_weighted"      # 按得分绝对值加权
    TOP_N = "top_n"                        # 只持仓 Top N，等权


class PortfolioManager:
    """
    多策略组合管理器。

    负责将多个策略注册到 PyBroker 的 Strategy 中，
    并根据因子权重分配各策略的仓位。

    Attributes:
        strategies: 已注册的策略实例字典 {名称: 实例}
        weights: 策略权重字典 {名称: 权重}
        total_allocation: 总仓位上限（占总权益比例）
    """

    def __init__(
        self,
        total_allocation: float = 0.8,
        factor_weights: Optional[Dict[str, float]] = None,
    ):
        self.strategies: Dict[str, object] = {}
        self.weights: Dict[str, float] = factor_weights or DEFAULT_FACTOR_WEIGHTS.copy()
        self.total_allocation = total_allocation

    def add_strategy(self, name: str, strategy: object, weight: Optional[float] = None):
        """
        添加策略到组合。

        Args:
            name: 策略名称
            strategy: 策略实例
            weight: 策略权重，为 None 时使用默认权重或等权分配
        """
        self.strategies[name] = strategy
        if weight is not None:
            self.weights[name] = weight
        elif name in DEFAULT_FACTOR_WEIGHTS:
            self.weights[name] = DEFAULT_FACTOR_WEIGHTS[name]
        else:
            self.weights[name] = 1.0
        self.set_weights(self.weights)

    def remove_strategy(self, name: str):
        """
        从组合中移除策略。

        Args:
            name: 策略名称
        """
        if name in self.strategies:
            del self.strategies[name]
        if name in self.weights:
            del self.weights[name]

    def set_weights(self, weights: Dict[str, float]):
        """
        设置策略权重。

        权重会被归一化，使总和为1。
        过滤掉不在 self.strategies 中的策略键。
        权重必须是非负数。

        Args:
            weights: 策略权重字典

        Raises:
            ValueError: 如果存在负数权重
        """
        negative = {k: v for k, v in weights.items() if v < 0}
        if negative:
            raise ValueError(f"策略权重不能为负数: {negative}")
        filtered = {k: v for k, v in weights.items() if k in self.strategies}
        total = sum(filtered.values())
        if total > 0:
            self.weights = {k: v / total for k, v in filtered.items()}
        else:
            self.weights = filtered

    def get_adjusted_position_size(self, strategy_name: str, base_size: float) -> float:
        """
        根据策略权重和总仓位上限调整仓位大小。

        Args:
            strategy_name: 策略名称
            base_size: 策略基础仓位比例

        Returns:
            调整后的仓位比例
        """
        weight = self.weights.get(strategy_name, 0.25)
        return base_size * weight * self.total_allocation

    def register_all_to_pybroker(
        self,
        pybroker_strategy,
        symbols: List[str],
        rollover_wrapper: Optional[Callable] = None,
    ):
        """
        将所有已注册策略添加到 PyBroker 的 Strategy 实例中。

        每个策略的指标列表通过 register_indicators() 获取，
        并传递给 add_execution 的 indicators 参数。

        Args:
            pybroker_strategy: PyBroker 的 Strategy 实例
            symbols: 交易合约代码列表
            rollover_wrapper: 可选的展期包装函数
        """
        for _name, strat in self.strategies.items():
            indicators = []
            if hasattr(strat, "register_indicators"):
                indicators = strat.register_indicators()

            exec_fn = strat.execute
            if rollover_wrapper is not None:
                exec_fn = rollover_wrapper(exec_fn)

            pybroker_strategy.add_execution(
                fn=exec_fn, symbols=symbols, indicators=indicators
            )

    def get_portfolio_summary(self) -> Dict:
        """
        获取组合摘要信息。

        Returns:
            组合摘要字典
        """
        return {
            "strategy_count": len(self.strategies),
            "strategies": list(self.strategies.keys()),
            "weights": self.weights,
            "total_allocation": self.total_allocation,
        }

    # -----------------------------------------------------------------------
    # P1-任务9整改：品种轮动决策（基于综合得分）
    # -----------------------------------------------------------------------
    def select_top_by_signal(
        self,
        scores: Dict[str, float],
        top_n: int = 3,
        min_score: float = 0.0,
    ) -> List[str]:
        """
        根据综合得分选择前 N 个品种。

        P1-任务9整改：从 FactorScoringEngine 移出品种轮动职责，
        统一由 PortfolioManager 负责。

        Args:
            scores: {symbol: 综合得分，得分范围 [-1, 1]}
            top_n: 选出的品种数（正数=做多 Top N，负数=做空 Top |N|）
            min_score: 最低得分阈值（绝对值），低于此值的品种被过滤

        Returns:
            入选品种代码列表（按得分绝对值从大到小排序）
        """
        if not scores:
            return []
        # 过滤掉绝对值低于阈值的得分
        filtered = {s: v for s, v in scores.items() if abs(v) >= min_score}
        if not filtered:
            return []
        # 按绝对值降序排列
        sorted_symbols = sorted(
            filtered.keys(),
            key=lambda s: abs(filtered[s]),
            reverse=True,
        )
        n = abs(int(top_n))
        if n <= 0:
            return []
        return sorted_symbols[:n]

    # -----------------------------------------------------------------------
    # P0-2整改：资金分配方法（蓝图：target_weights = portfolio_manager.allocate_weights）
    # -----------------------------------------------------------------------
    def allocate_weights(
        self,
        signals: Dict[str, float],
        method: str = "equal_weight",
        total_allocation: Optional[float] = None,
        risk_estimates: Optional[Dict[str, float]] = None,
        top_n: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        根据综合得分给各品种分配目标权重。

        蓝图（P0-2 整改）：
            target_weights = portfolio_manager.allocate_weights(
                signals, method='risk_parity'
            )

        Args:
            signals: {symbol: 综合得分，范围 [-1, 1]}
            method: 权重方法，支持 equal_weight / risk_parity / score_weighted / top_n
            total_allocation: 总仓位上限（None 时使用 self.total_allocation）
            risk_estimates: {symbol: 风险估计（如 ATR%）}，risk_parity 方法使用
            top_n: top_n 方法下选出的品种数

        Returns:
            {symbol: 目标权重}，权重之和 = total_allocation
        """
        if not signals:
            return {}

        total_alloc = total_allocation if total_allocation is not None else self.total_allocation

        try:
            method_enum = WeightMethod(method)
        except ValueError:
            _logger.warning("未知权重方法 %s，回退到 equal_weight", method)
            method_enum = WeightMethod.EQUAL_WEIGHT

        # 过滤掉零分品种
        active = {s: v for s, v in signals.items() if abs(v) > 1e-8}
        if not active:
            return {}

        if method_enum == WeightMethod.TOP_N:
            n = top_n or 3
            selected = self.select_top_by_signal(active, top_n=n)
            if not selected:
                return {}
            per = total_alloc / len(selected)
            return {s: per * (1.0 if active[s] > 0 else -1.0) for s in selected}

        if method_enum == WeightMethod.SCORE_WEIGHTED:
            # 按 |score| 加权
            abs_vals = {s: abs(v) for s, v in active.items()}
            total = sum(abs_vals.values())
            if total <= 0:
                return {}
            return {
                s: (v / total) * total_alloc * (1.0 if active[s] > 0 else -1.0)
                for s, v in abs_vals.items()
            }

        if method_enum == WeightMethod.RISK_PARITY:
            # 风险倒数等权：weight_i = (1/risk_i) / Σ(1/risk_j) * total_alloc
            inv_risks: Dict[str, float] = {}
            for sym in active:
                if risk_estimates and sym in risk_estimates and risk_estimates[sym] > 0:
                    inv_risks[sym] = 1.0 / risk_estimates[sym]
                else:
                    # 风险未知时退化为 1（等权）
                    inv_risks[sym] = 1.0
            total_inv = sum(inv_risks.values())
            if total_inv <= 0:
                return {}
            return {
                s: (inv_risks[s] / total_inv) * total_alloc * (1.0 if active[s] > 0 else -1.0)
                for s in active
            }

        # EQUAL_WEIGHT（默认）
        per = total_alloc / len(active)
        return {s: per * (1.0 if active[s] > 0 else -1.0) for s in active}

    def adjust_weights_by_risk(
        self,
        weights: Dict[str, float],
        concentration_limit: float = 0.4,
    ) -> Dict[str, float]:
        """
        风险调整：限制单品种集中度。

        蓝图（P0-2 整改）：
            target_weights = risk_controller.adjust(target_weights, ctx)
        简化版：直接限制单品种绝对值占比。

        Args:
            weights: {symbol: 目标权重}
            concentration_limit: 单品种最大集中度

        Returns:
            调整后的权重（保持总权重之和不变）
        """
        if not weights:
            return {}
        result: Dict[str, float] = {}
        for sym, w in weights.items():
            if abs(w) > concentration_limit:
                result[sym] = concentration_limit * (1.0 if w > 0 else -1.0)
            else:
                result[sym] = float(w)
        return result

    def get_target_weights(
        self,
        scores: Dict[str, float],
        top_n: int = 3,
        min_score: float = 0.0,
    ) -> Dict[str, float]:
        """
        根据综合得分生成目标权重。

        P1-任务9整改：基于 select_top_by_signal 的选股结果，
        按得分符号分配多空、按 |得分| 归一化权重。

        Args:
            scores: {symbol: 综合得分}
            top_n: 多/空各选 Top N
            min_score: 最低得分阈值

        Returns:
            {symbol: 目标权重}（已归一化，总和 = 1.0）
        """
        if not scores:
            return {}
        long_top = self.select_top_by_signal(
            {s: v for s, v in scores.items() if v > 0},
            top_n=top_n,
            min_score=min_score,
        )
        short_top = self.select_top_by_signal(
            {s: v for s, v in scores.items() if v < 0},
            top_n=-top_n,
            min_score=min_score,
        )

        raw_weights: Dict[str, float] = {}
        for sym in long_top:
            raw_weights[sym] = abs(scores[sym])
        for sym in short_top:
            raw_weights[sym] = -abs(scores[sym])

        total = sum(abs(w) for w in raw_weights.values())
        if total <= 0:
            return {}
        return {s: w / total for s, w in raw_weights.items()}
