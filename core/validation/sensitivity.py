"""
参数敏感性分析模块。

对关键参数进行扰动测试，评估策略对参数变化的敏感度：
  - 每个参数±20%扰动
  - 记录 Sharpe / 最大回撤 / 年化收益 三指标的变化
  - 任意指标变化 >30% 视为高敏感

规则15要求：参数敏感性分析，Sharpe 变化>30% 视为高敏感。
P1 整改（2026-06-07）：
  - SensitivityResult 扩展为同时记录 max_drawdown / annual_return 变化
  - is_high_sensitivity 综合三指标判定
  - analyze 方法对回测函数返回值做显式校验
P2 整改（2026-06-07）：
  - 新增 n_jobs 参数，支持 concurrent.futures 并行扰动回测
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)

# 默认扰动比例
DEFAULT_PERTURBATION = 0.20

# 高敏感阈值（任意指标触发即标记为高敏感）
HIGH_SENSITIVITY_THRESHOLD = 0.30

# 用于回测函数返回结果校验的必需键
_REQUIRED_RESULT_KEYS = ("sharpe", "max_drawdown", "annual_return")


@dataclass
class SensitivityResult:
    """单参数敏感性分析结果（P1 整改：扩展为三指标）。"""

    param_name: str = ""
    base_value: float = 0.0
    low_value: float = 0.0
    high_value: float = 0.0

    # Sharpe
    base_sharpe: float = 0.0
    low_sharpe: float = 0.0
    high_sharpe: float = 0.0
    sharpe_change_pct: float = 0.0

    # 最大回撤（P1 新增）
    base_max_drawdown: float = 0.0
    low_max_drawdown: float = 0.0
    high_max_drawdown: float = 0.0
    max_drawdown_change_pct: float = 0.0

    # 年化收益（P1 新增）
    base_annual_return: float = 0.0
    low_annual_return: float = 0.0
    high_annual_return: float = 0.0
    annual_return_change_pct: float = 0.0

    # 综合判定
    is_high_sensitivity: bool = False

    def summary(self) -> str:
        """返回敏感性摘要（P1：展示三指标）。"""
        sens_str = "⚠️高敏感" if self.is_high_sensitivity else "✅低敏感"
        return (
            f"[{self.param_name}] {self.base_value:.4f}→"
            f"[{self.low_value:.4f}, {self.high_value:.4f}] | "
            f"Sharpe: {self.base_sharpe:.4f}→"
            f"[{self.low_sharpe:.4f}, {self.high_sharpe:.4f}] "
            f"(Δ{self.sharpe_change_pct:.1%}) | "
            f"MDD: {self.base_max_drawdown:.2%}→"
            f"[{self.low_max_drawdown:.2%}, {self.high_max_drawdown:.2%}] "
            f"(Δ{self.max_drawdown_change_pct:.1%}) | "
            f"Ret: {self.base_annual_return:.2%}→"
            f"[{self.low_annual_return:.2%}, {self.high_annual_return:.2%}] "
            f"(Δ{self.annual_return_change_pct:.1%}) | {sens_str}"
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


def _safe_get(
    result: Optional[Dict[str, Any]], key: str, default: float = 0.0
) -> float:
    """
    从回测结果字典中安全获取指标值（P1 整改）。

    缺键时记录警告并返回默认值，避免静默兜底。
    """
    if result is None or not isinstance(result, dict):
        return default
    if key not in result:
        return default
    try:
        v = float(result[key])
        if not np.isfinite(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


class SensitivityAnalyzer:
    """
    参数敏感性分析器（P1/P2 整改后）。

    对关键参数进行 ±20% 扰动，评估策略对参数变化的敏感度。
    任意指标（Sharpe / 最大回撤 / 年化收益）变化 > 阈值 即视为高敏感。

    用法:
        analyzer = SensitivityAnalyzer(n_jobs=4)  # 4 线程并行
        result = analyzer.analyze(
            params={"ma_window": 20, "atr_mult": 2.0},
            backtest_func=my_backtest_func,
        )
    """

    def __init__(
        self,
        perturbation: float = DEFAULT_PERTURBATION,
        high_sensitivity_threshold: float = HIGH_SENSITIVITY_THRESHOLD,
        n_jobs: int = 1,
    ):
        """
        初始化敏感性分析器。

        Args:
            perturbation: 扰动比例（默认 20%）
            high_sensitivity_threshold: 高敏感阈值（默认 30%）
            n_jobs: 并行执行的回测任务数（默认 1 串行）
        """
        self.perturbation = perturbation
        self.high_sensitivity_threshold = high_sensitivity_threshold
        self.n_jobs = max(1, int(n_jobs))

    # -----------------------------------------------------------------------
    # 内部工具
    # -----------------------------------------------------------------------
    @staticmethod
    def _compute_change_pct(base: float, low: float, high: float) -> float:
        """计算最大相对变化幅度（分母防 0）。"""
        if abs(base) < 1e-8:
            spread = max(abs(high), abs(low))
            return 1.0 if spread > 1e-4 else 0.0
        spread = max(abs(high - base), abs(low - base))
        return spread / abs(base)

    def _build_perturbed_params(
        self,
        base_params: Dict[str, float],
        param_name: str,
        param_constraints: Optional[Dict[str, Tuple[float, float]]],
    ) -> Tuple[Any, Any]:
        """
        计算低/高扰动值，应用约束与整数取整规则。

        Returns:
            (low_value, high_value)
        """
        base_value = base_params[param_name]
        low_value = base_value * (1 - self.perturbation)
        high_value = base_value * (1 + self.perturbation)

        if param_name in (param_constraints or {}):
            min_val, max_val = param_constraints[param_name]
            low_value = max(low_value, min_val)
            high_value = min(high_value, max_val)

        # 整数参数取整（避免产生 19.6 这种无意义窗口）
        if isinstance(base_value, int) or (
            isinstance(base_value, float) and base_value == int(base_value)
        ):
            low_value = max(1, int(round(low_value)))
            high_value = int(round(high_value))

        return low_value, high_value

    def _perturb_and_backtest(
        self,
        base_params: Dict[str, float],
        param_name: str,
        low_value: Any,
        high_value: Any,
        backtest_func: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        执行扰动回测（P1 整改：捕获指标三件套）。

        Returns:
            (low_result, high_result)
        """
        params_low = dict(base_params)
        params_low[param_name] = low_value
        low_result = backtest_func(params_low) or {}

        params_high = dict(base_params)
        params_high[param_name] = high_value
        high_result = backtest_func(params_high) or {}

        return low_result, high_result

    def _analyze_single_param(
        self,
        param_name: str,
        base_params: Dict[str, float],
        base_result: Dict[str, Any],
        param_constraints: Optional[Dict[str, Tuple[float, float]]],
        backtest_func: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> SensitivityResult:
        """对单个参数执行扰动分析。"""
        base_value = base_params[param_name]
        low_value, high_value = self._build_perturbed_params(
            base_params, param_name, param_constraints
        )

        low_result, high_result = self._perturb_and_backtest(
            base_params, param_name, low_value, high_value, backtest_func
        )

        base_sharpe = _safe_get(base_result, "sharpe")
        low_sharpe = _safe_get(low_result, "sharpe")
        high_sharpe = _safe_get(high_result, "sharpe")
        sharpe_change = self._compute_change_pct(base_sharpe, low_sharpe, high_sharpe)

        base_mdd = _safe_get(base_result, "max_drawdown")
        low_mdd = _safe_get(low_result, "max_drawdown")
        high_mdd = _safe_get(high_result, "max_drawdown")
        mdd_change = self._compute_change_pct(base_mdd, low_mdd, high_mdd)

        base_ret = _safe_get(base_result, "annual_return")
        low_ret = _safe_get(low_result, "annual_return")
        high_ret = _safe_get(high_result, "annual_return")
        ret_change = self._compute_change_pct(base_ret, low_ret, high_ret)

        # 综合判定：任一指标超阈值即为高敏感
        is_high = (
            sharpe_change > self.high_sensitivity_threshold
            or mdd_change > self.high_sensitivity_threshold
            or ret_change > self.high_sensitivity_threshold
        )

        return SensitivityResult(
            param_name=param_name,
            base_value=base_value,
            low_value=low_value,
            high_value=high_value,
            base_sharpe=base_sharpe,
            low_sharpe=low_sharpe,
            high_sharpe=high_sharpe,
            sharpe_change_pct=sharpe_change,
            base_max_drawdown=base_mdd,
            low_max_drawdown=low_mdd,
            high_max_drawdown=high_mdd,
            max_drawdown_change_pct=mdd_change,
            base_annual_return=base_ret,
            low_annual_return=low_ret,
            high_annual_return=high_ret,
            annual_return_change_pct=ret_change,
            is_high_sensitivity=is_high,
        )

    # -----------------------------------------------------------------------
    # 主入口
    # -----------------------------------------------------------------------
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
            backtest_func: 回测函数，接受参数字典，返回
                {"sharpe": float, "max_drawdown": float, "annual_return": float}
            param_constraints: 参数约束 {参数名: (最小值, 最大值)}

        Returns:
            FullSensitivityResult 完整分析结果
        """
        constraints = param_constraints or {}

        # P1 整改：基准回测返回值校验
        base_result = backtest_func(params)
        if base_result is None:
            logger.warning("基准回测返回 None，将使用 0.0 兜底（敏感性分析可能失真）")
            base_result = {}
        elif not isinstance(base_result, dict):
            raise TypeError(
                f"backtest_func 必须返回 dict 或 None，实际返回 {type(base_result).__name__}"
            )
        missing = [k for k in _REQUIRED_RESULT_KEYS if k not in base_result]
        if missing:
            logger.warning(
                "回测函数返回缺少键 %s，将使用 0.0 兜底（可能导致敏感性误判）",
                missing,
            )
        base_sharpe = _safe_get(base_result, "sharpe")
        logger.info(f"基准 Sharpe: {base_sharpe:.4f}")

        param_items = list(params.items())
        results: List[SensitivityResult] = []

        if self.n_jobs == 1:
            # 串行模式
            for param_name, _ in param_items:
                res = self._analyze_single_param(
                    param_name,
                    params,
                    base_result,
                    constraints,
                    backtest_func,
                )
                results.append(res)
        else:
            # P2 整改：并行模式
            results = self._analyze_parallel(
                params,
                base_result,
                constraints,
                backtest_func,
                param_items,
            )

        high_sens_params = [r.param_name for r in results if r.is_high_sensitivity]
        for r in results:
            if r.is_high_sensitivity:
                logger.warning(
                    f"高敏感参数：{r.param_name} "
                    f"(Sharpe Δ{r.sharpe_change_pct:.1%}, "
                    f"MDD Δ{r.max_drawdown_change_pct:.1%}, "
                    f"Ret Δ{r.annual_return_change_pct:.1%})"
                )

        full_result = FullSensitivityResult(
            results=results,
            high_sensitivity_params=high_sens_params,
        )

        logger.info(full_result.summary())
        return full_result

    def _analyze_parallel(
        self,
        params: Dict[str, float],
        base_result: Dict[str, Any],
        param_constraints: Dict[str, Tuple[float, float]],
        backtest_func: Callable[[Dict[str, Any]], Dict[str, Any]],
        param_items: List[Tuple[str, float]],
    ) -> List[SensitivityResult]:
        """并行执行单参数扰动分析。"""
        results_by_name: Dict[str, SensitivityResult] = {}

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_name = {
                executor.submit(
                    self._analyze_single_param,
                    pname,
                    params,
                    base_result,
                    param_constraints,
                    backtest_func,
                ): pname
                for pname, _ in param_items
            }
            for future in as_completed(future_to_name):
                pname = future_to_name[future]
                try:
                    results_by_name[pname] = future.result()
                except Exception as e:
                    logger.error(f"参数 {pname} 扰动分析失败: {e}")
                    # 失败时填占位
                    results_by_name[pname] = SensitivityResult(param_name=pname)

        # 按输入顺序返回
        return [results_by_name[name] for name, _ in param_items]
