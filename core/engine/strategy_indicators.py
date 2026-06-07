"""
策略指标注册表 — 解耦 backtest_runner 中的指标计算硬编码。

设计目标：
  - 每个子策略可以独立注册自己的 PyBroker 指标构建函数
  - backtest_runner 不再硬编码任何指标计算逻辑
  - 新增策略时只需注册指标构建函数，无需修改 backtest_runner

使用方式：
  from core.engine.strategy_indicators import StrategyIndicatorRegistry, IndicatorSpec

  # 注册趋势策略指标
  StrategyIndicatorRegistry.register(
      "trend",
      build_trend_indicators,
      indicator_names=["trend_signal"],
  )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class IndicatorSpec:
    """
    策略指标规格。

    描述一个子策略所需的 PyBroker 指标集合。
    """

    # 策略名称
    strategy_name: str = ""

    # 指标构建函数：params → [(name, fn), ...]
    builder: Optional[Callable[[Dict[str, Any]], List[tuple]]] = None

    # 该策略产生的指标名称列表（用于 switch_engine 自动发现）
    indicator_names: List[str] = field(default_factory=list)

    # 指标名到因子名的映射（可选，默认使用 indicator_names）
    indicator_to_factor: Dict[str, str] = field(default_factory=dict)


class StrategyIndicatorRegistry:
    """
    策略指标注册表。

    集中管理所有子策略的 PyBroker 指标定义，
    供 backtest_runner 和 switch_engine 查询使用。

    线程安全：使用类级别字典，模块加载时一次性注册。
    """

    _specs: Dict[str, IndicatorSpec] = {}
    _builders: Dict[str, Callable[[Dict[str, Any]], List[tuple]]] = {}

    @classmethod
    def register(
        cls,
        strategy_name: str,
        builder: Callable[[Dict[str, Any]], List[tuple]],
        indicator_names: Optional[List[str]] = None,
        indicator_to_factor: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        注册策略指标构建函数。

        Args:
            strategy_name: 策略名称（如 "trend", "term_structure"）
            builder: 指标构建函数，签名为 (params: Dict) -> List[tuple]
                     每个 tuple 为 (name: str, fn: Callable)
            indicator_names: 该策略产生的指标名称列表
            indicator_to_factor: 指标名 → 因子名映射
        """
        spec = IndicatorSpec(
            strategy_name=strategy_name,
            builder=builder,
            indicator_names=indicator_names or [],
            indicator_to_factor=indicator_to_factor or {},
        )
        cls._specs[strategy_name] = spec
        cls._builders[strategy_name] = builder
        logger.info("注册策略指标: %s (指标: %s)", strategy_name, indicator_names)

    @classmethod
    def build_for_strategy(
        cls,
        strategy_name: str,
        params: Dict[str, Any],
    ) -> List[tuple]:
        """
        为指定策略构建 PyBroker 指标列表。

        Args:
            strategy_name: 策略名称
            params: 策略参数（已合并默认参数和自定义参数）

        Returns:
            [(name, fn), ...] 列表，可直接传给 pybroker.indicator()
        """
        builder = cls._builders.get(strategy_name)
        if builder is None:
            logger.warning("策略 %s 未注册指标构建函数", strategy_name)
            return []
        try:
            return builder(params)
        except Exception as e:
            logger.error("策略 %s 指标构建失败: %s", strategy_name, e)
            return []

    @classmethod
    def build_all(
        cls,
        strategy_params: Dict[str, Dict[str, Any]],
    ) -> List[tuple]:
        """
        为所有已注册策略构建 PyBroker 指标列表（去重）。

        Args:
            strategy_params: {策略名: 参数字典}

        Returns:
            去重后的 [(name, fn), ...] 列表
        """
        all_indicators: List[tuple] = []
        seen_names: set = set()

        for sname, params in strategy_params.items():
            indicators = cls.build_for_strategy(sname, params)
            for name, fn in indicators:
                if name not in seen_names:
                    seen_names.add(name)
                    all_indicators.append((name, fn))

        return all_indicators

    @classmethod
    def get_indicator_to_factor_map(cls) -> Dict[str, str]:
        """
        获取所有已注册策略的 指标名 → 因子名 映射。

        Returns:
            {indicator_name: factor_name} 字典
        """
        mapping: Dict[str, str] = {}
        for spec in cls._specs.values():
            if spec.indicator_to_factor:
                mapping.update(spec.indicator_to_factor)
            else:
                # 默认映射：indicator_name 去掉 _signal 后缀
                for name in spec.indicator_names:
                    factor = name.replace("_signal", "")
                    mapping[name] = factor
        return mapping

    @classmethod
    def get_indicator_names(cls) -> List[str]:
        """获取所有已注册策略的指标名称列表。"""
        names: List[str] = []
        for spec in cls._specs.values():
            names.extend(spec.indicator_names)
        return names

    @classmethod
    def is_registered(cls, strategy_name: str) -> bool:
        """检查策略是否已注册指标。"""
        return strategy_name in cls._builders

    @classmethod
    def get_indicator_value(cls, ctx: Any, name: str) -> Optional[float]:
        """
        从 ExecContext 读取指定指标的当前值（P1-1 整改）。

        统一信号获取接口：所有子策略信号应通过本方法获取，
        而不是每个策略实现 compute_signal(ctx, factor_data)。

        Args:
            ctx: PyBroker ExecContext
            name: 指标名称

        Returns:
            指标值（已 clip 到 [-1, 1]），未注册/不可用时返回 None
        """
        if ctx is None or name is None:
            return None
        try:
            raw = ctx.indicator(name) if hasattr(ctx, "indicator") else None
        except Exception:
            raw = None
        if raw is None:
            return None
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        if val != val:  # NaN
            return None
        import math
        if math.isinf(val):
            return None
        return max(-1.0, min(1.0, val))

    @classmethod
    def clear(cls) -> None:
        """清除所有注册（主要用于测试）。"""
        cls._specs.clear()
        cls._builders.clear()


# ---------------------------------------------------------------------------
# 策略退出钩子注册表
# ---------------------------------------------------------------------------

@dataclass
class StrategyExitHook:
    """
    策略退出钩子定义。

    描述一个策略在特定条件下需要平仓退出的逻辑。
    """

    # 策略名称
    strategy_name: str = ""

    # 退出条件检查函数：签名为 (ctx, indicator_values, strategy_params) -> bool
    # 返回 True 表示应平仓退出
    checker: Optional[Callable[[Any, Dict[str, float], Dict[str, Any]], bool]] = None

    # 退出原因描述
    reason: str = ""


class StrategyExitHookRegistry:
    """
    策略退出钩子注册表。

    集中管理所有子策略的退出条件，供 strategy_executor 查询使用。
    解耦策略特定退出逻辑与执行器核心代码。
    """

    _hooks: Dict[str, StrategyExitHook] = {}

    @classmethod
    def register(
        cls,
        strategy_name: str,
        checker: Callable[[Any, Dict[str, float], Dict[str, Any]], bool],
        reason: str = "",
    ) -> None:
        """
        注册策略退出钩子。

        Args:
            strategy_name: 策略名称
            checker: 退出条件检查函数
            reason: 退出原因描述
        """
        hook = StrategyExitHook(
            strategy_name=strategy_name,
            checker=checker,
            reason=reason,
        )
        cls._hooks[strategy_name] = hook
        logger.info("注册退出钩子: %s (原因: %s)", strategy_name, reason)

    @classmethod
    def check_exit(
        cls,
        strategy_name: str,
        ctx: Any,
        indicator_values: Dict[str, float],
        strategy_params: Dict[str, Any],
    ) -> bool:
        """
        检查指定策略是否触发退出条件。

        Args:
            strategy_name: 策略名称
            ctx: PyBroker ExecContext
            indicator_values: 当前指标值字典
            strategy_params: 策略参数

        Returns:
            True 表示应平仓退出
        """
        hook = cls._hooks.get(strategy_name)
        if hook is None or hook.checker is None:
            return False
        try:
            return hook.checker(ctx, indicator_values, strategy_params)
        except Exception as e:
            logger.warning("策略 %s 退出钩子检查异常: %s", strategy_name, e)
            return False

    @classmethod
    def is_registered(cls, strategy_name: str) -> bool:
        """检查策略是否已注册退出钩子。"""
        return strategy_name in cls._hooks

    @classmethod
    def clear(cls) -> None:
        """清除所有注册（主要用于测试）。"""
        cls._hooks.clear()


__all__ = [
    "IndicatorSpec",
    "StrategyIndicatorRegistry",
    "StrategyExitHook",
    "StrategyExitHookRegistry",
]