"""
组合管理模块。

管理多策略组合，根据市场环境动态调整策略权重，
协调多个 PyBroker 策略的执行。

PyBroker 原生支持通过 strategy.add_execution() 添加多个执行函数，
每个执行函数可以对应不同的策略。本模块在此基础上增加：
- 环境动态权重调整
- 多策略资金分配
- 组合级别的绩效汇总
"""

from typing import Dict, List, Optional, Callable
from .environment import EnvironmentAdapter


class PortfolioManager:
    """
    多策略组合管理器。

    负责将多个策略注册到 PyBroker 的 Strategy 中，
    并根据市场环境动态调整各策略的仓位权重。

    Attributes:
        strategies: 已注册的策略实例字典 {名称: 实例}
        weights: 策略权重字典 {名称: 权重}
        env_adapter: 环境适配器实例
        total_allocation: 总仓位上限（占总权益比例）
    """

    DEFAULT_WEIGHTS = {
        "dual_ma": 0.4,
        "rsi": 0.3,
        "spread": 0.3,
    }

    def __init__(
        self,
        env_adapter: Optional[EnvironmentAdapter] = None,
        total_allocation: float = 0.8,
    ):
        self.strategies: Dict[str, object] = {}
        self.weights: Dict[str, float] = self.DEFAULT_WEIGHTS.copy()
        self.env_adapter = env_adapter or EnvironmentAdapter()
        self.total_allocation = total_allocation

    def add_strategy(self, name: str, strategy: object, weight: Optional[float] = None):
        """
        添加策略到组合。

        Args:
            name: 策略名称
            strategy: 策略实例
            weight: 策略权重，为 None 时使用默认权重或动态计算
        """
        self.strategies[name] = strategy
        if weight is not None:
            self.weights[name] = weight
        elif name not in self.DEFAULT_WEIGHTS:
            temp_weight = 1.0 / (len(self.strategies) + 1)
            self.weights[name] = temp_weight
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

        Args:
            weights: 策略权重字典
        """
        filtered = {k: v for k, v in weights.items() if k in self.strategies}
        total = sum(filtered.values())
        if total > 0:
            self.weights = {k: v / total for k, v in filtered.items()}
        else:
            self.weights = filtered

    def update_weights_by_regime(self, regime: str):
        """
        根据市场状态更新策略权重。

        Args:
            regime: 市场状态，'trend' 或 'range'
        """
        regime_weights = self.env_adapter.get_regime_weights(regime)
        mapping = {"trend": "dual_ma", "reversal": "rsi", "spread": "spread"}
        new_weights = {}
        for regime_key, strategy_name in mapping.items():
            if strategy_name in self.strategies:
                new_weights[strategy_name] = regime_weights.get(regime_key, 0.33)
        self.set_weights(new_weights)

    def get_adjusted_position_size(self, strategy_name: str, base_size: float) -> float:
        """
        根据策略权重和总仓位上限调整仓位大小。

        Args:
            strategy_name: 策略名称
            base_size: 策略基础仓位比例

        Returns:
            调整后的仓位比例
        """
        weight = self.weights.get(strategy_name, 0.33)
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
