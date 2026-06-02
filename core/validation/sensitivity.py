"""
参数敏感性分析模块。

对关键参数进行扰动测试，评估策略对参数变化的敏感度：
  - 每个参数±20%扰动
  - 记录Sharpe/最大回撤/年化收益的变化
  - Sharpe变化>30%视为高敏感

规则15要求：参数敏感性分析，Sharpe变化>30%视为高敏感。
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)

# 默认扰动比例
DEFAULT_PERTURBATION = 0.20

# 高敏感阈值
HIGH_SENSITIVITY_THRESHOLD = 0.30


@dataclass
class SensitivityResult:
    """单参数敏感性分析结果。"""

    param_name: str = ""
    base_value: float = 0.0
    low_value: float = 0.0
    high_value: float = 0.0
    base_sharpe: float = 0.0
    low_sharpe: float = 0.0
    high_sharpe: float = 0.0
    sharpe_change_pct: float = 0.0
    is_high_sensitivity: bool = False

    def summary(self) -> str:
        """返回敏感性摘要。"""
        sens_str = "⚠️高敏感" if self.is_high_sensitivity else "✅低敏感"
        return (
            f"[{self.param_name}] {self.base_value:.4f}→"
            f"[{self.low_value:.4f}, {self.high_value:.4f}] | "
            f"Sharpe: {self.base_sharpe:.4f}→"
            f"[{self.low_sharpe:.4f}, {self.high_sharpe:.4f}] | "
            f"变化{self.sharpe_change_pct:.1%} | {sens_str}"
        )


@dataclass
class FullSensitivityResult:
    """完整敏感性分析结果。"""

    results: List[SensitivityResult] = field(default_factory=list)
    high_sensitivity_params: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """返回完整摘要。"""
        lines = ["参数敏感性分析结果："]
        for r in self.results:
            lines.append(f"  {r.summary()}")
        if self.high_sensitivity_params:
            lines.append(
                f"高敏感参数：{', '.join(self.high_sensitivity_params)}"
            )
        return "\n".join(lines)


class SensitivityAnalyzer:
    """
    参数敏感性分析器。

    对关键参数进行±20%扰动，评估策略对参数变化的敏感度。

    用法:
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze(
            params={"ma_window": 20, "atr_mult": 2.0},
            backtest_func=my_backtest_func,
        )
    """

    def __init__(
        self,
        perturbation: float = DEFAULT_PERTURBATION,
        high_sensitivity_threshold: float = HIGH_SENSITIVITY_THRESHOLD,
    ):
        """
        初始化敏感性分析器。

        Args:
            perturbation: 扰动比例（默认20%）
            high_sensitivity_threshold: 高敏感阈值（默认30%）
        """
        self.perturbation = perturbation
        self.high_sensitivity_threshold = high_sensitivity_threshold

    def analyze(
        self,
        params: Dict[str, float],
        backtest_func: Callable[[Dict[str, float]], Dict[str, float]],
        param_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> FullSensitivityResult:
        """
        执行参数敏感性分析。

        Args:
            params: 基准参数 {参数名: 基准值}
            backtest_func: 回测函数，接受参数字典，返回 {"sharpe": float, ...}
            param_constraints: 参数约束 {参数名: (最小值, 最大值)}

        Returns:
            FullSensitivityResult 完整分析结果
        """
        constraints = param_constraints or {}

        # 基准回测
        base_result = backtest_func(params)
        base_sharpe = base_result.get("sharpe", 0.0)

        results: List[SensitivityResult] = []
        high_sens_params: List[str] = []

        for param_name, base_value in params.items():
            # 计算扰动值
            low_value = base_value * (1 - self.perturbation)
            high_value = base_value * (1 + self.perturbation)

            # 应用约束
            if param_name in constraints:
                min_val, max_val = constraints[param_name]
                low_value = max(low_value, min_val)
                high_value = min(high_value, max_val)

            # 整数参数处理
            if isinstance(base_value, int) or (isinstance(base_value, float) and base_value == int(base_value)):
                low_value = max(1, int(round(low_value)))
                high_value = int(round(high_value))

            # 低值回测
            params_low = dict(params)
            params_low[param_name] = low_value
            low_result = backtest_func(params_low)
            low_sharpe = low_result.get("sharpe", 0.0)

            # 高值回测
            params_high = dict(params)
            params_high[param_name] = high_value
            high_result = backtest_func(params_high)
            high_sharpe = high_result.get("sharpe", 0.0)

            # 计算Sharpe变化幅度
            sharpe_range = max(abs(high_sharpe - base_sharpe), abs(low_sharpe - base_sharpe))
            if abs(base_sharpe) > 1e-8:
                change_pct = sharpe_range / abs(base_sharpe)
            else:
                change_pct = 1.0 if sharpe_range > 1e-4 else 0.0

            is_high = change_pct > self.high_sensitivity_threshold

            result = SensitivityResult(
                param_name=param_name,
                base_value=base_value,
                low_value=low_value,
                high_value=high_value,
                base_sharpe=base_sharpe,
                low_sharpe=low_sharpe,
                high_sharpe=high_sharpe,
                sharpe_change_pct=change_pct,
                is_high_sensitivity=is_high,
            )

            results.append(result)

            if is_high:
                high_sens_params.append(param_name)
                logger.warning(
                    f"高敏感参数：{param_name}，Sharpe变化{change_pct:.1%}"
                )

        full_result = FullSensitivityResult(
            results=results,
            high_sensitivity_params=high_sens_params,
        )

        logger.info(full_result.summary())
        return full_result
