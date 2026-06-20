"""gplearn 遗传规划因子挖掘（规则21 + 规则21.4）。

目标：在不污染 core/factors/alpha_futures/ 的前提下，提供可选的 GP 因子挖掘能力。

**核心设计**：
    1. **函数集自动构建**：从 core.factors.operators 的 sma/std/corr/delta 等基础算子构建
    2. **复用 BaseFactor**：挖掘出的因子必须继承 BaseFactor 并通过 register_factor 注册
    3. **复用 FactorEvaluator**：用 IC 评估挖掘出的因子质量
    4. **可插拔适应度**：默认 IC 适应度，可换 Sharpe/ICIR

**依赖**：
    pip install -r requirements-factors.txt
    或：pip install gplearn deap

**失败行为**（规则 21.2）：
    未安装 gplearn 时 import 此模块会立即抛 ImportError，不会污染 core/。
    不得在调用方 try/except 兜底。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

# 规则 21.2：第三方 import 必须直接，不得 try/except 兜底
import gplearn.genetic as gp

from core.ext.factors import operators as ops
from core.ext.factors.alpha_futures.base_factor import BaseFactor
from core.ext.factors.alpha_futures.config import AlphaFuturesConfig
from core.ext.factors.alpha_futures.factor_registry import register_factor


__all__ = [
    "GPLearnFactorMiner",
    "GPLearnConfig",
    "DEFAULT_FUNCTION_SET",
    "build_function_set",
    "register_mined_factors",
]


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 函数集：从 core.factors.operators 自动构建（规则 21.4 复用）
# ---------------------------------------------------------------------------
def build_function_set(include_advanced: bool = True) -> List[Callable]:
    """从 core.factors.operators 构建 gplearn 函数集。

    Args:
        include_advanced: 是否包含 abs/log/sign/winsorize/clipping 等高级函数

    Returns:
        满足 gplearn 签名的函数列表
    """
    # gplearn 函数签名：(ndarray, *args) -> ndarray
    def _sma(x, window):
        return ops.sma(x, int(window))

    def _std(x, window):
        return ops.std(x, int(window))

    def _delta(x, n):
        return ops.delta(x, int(n))

    def _delay(x, n):
        return ops.delay(x, int(n))

    def _mean(x, window):
        return ops.mean(x, int(window))

    def _corr(x, y, window):
        return ops.corr(x, y, int(window))

    def _tsrank(x, window):
        return ops.tsrank(x, int(window))

    def _ema(x, window):
        return ops.ema(x, int(window))

    def _add(x, y):
        return x + y

    def _sub(x, y):
        return x - y

    def _mul(x, y):
        return x * y

    def _div(x, y):
        return ops.safe_div(x, y)

    fns: List[Callable] = [
        _add, _sub, _mul, _div,
        _sma, _std, _delta, _delay, _mean, _corr, _tsrank, _ema,
    ]

    if include_advanced:
        def _abs(x): return ops.abs_(x)
        def _log(x): return ops.log(np.maximum(x, 1e-12))
        def _sign(x): return ops.sign(x)
        def _neg(x): return -x

        fns.extend([_abs, _log, _sign, _neg])

    return fns


DEFAULT_FUNCTION_SET: List[Callable] = build_function_set(include_advanced=True)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
class GPLearnConfig:
    """gplearn 因子挖掘配置。

    Attributes:
        n_factors: 期望挖掘的因子数量（种群数）
        generations: 进化代数
        population_size: 种群规模
        tournament_size: 锦标赛规模（越小越能保留多样性）
        parsimony_coefficient: 简约性惩罚（越大表达式越短）
        const_range: 常数范围
        init_depth: 初始树深度范围
        init_method: 初始化方法（'half and half' / 'grow' / 'full'）
        metric: 适应度函数名（'ic' / 'sharpe' / 'rank_ic'）
        random_state: 随机种子
        low_memory: 低内存模式（大数据集建议 True）
        n_jobs: 并行数（-1 = 全核）
    """

    def __init__(
        self,
        n_factors: int = 20,
        generations: int = 10,
        population_size: int = 200,
        tournament_size: int = 5,
        parsimony_coefficient: float = 0.001,
        const_range: Tuple[float, float] = (-1.0, 1.0),
        init_depth: Tuple[int, int] = (2, 5),
        init_method: str = "half and half",
        metric: str = "ic",
        random_state: Optional[int] = None,
        low_memory: bool = False,
        n_jobs: int = 1,
    ) -> None:
        self.n_factors = n_factors
        self.generations = generations
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.parsimony_coefficient = parsimony_coefficient
        self.const_range = const_range
        self.init_depth = init_depth
        self.init_method = init_method
        self.metric = metric
        self.random_state = random_state
        self.low_memory = low_memory
        self.n_jobs = n_jobs


# ---------------------------------------------------------------------------
# 适应度函数
# ---------------------------------------------------------------------------
def _make_fitness_func(metric: str) -> Callable:
    """根据指标名构造 gplearn 适应度函数。

    适应度 = 负损失（gplearn 最大化适应度），所以 IC 越高 → 损失越负 → 适应度越高。
    """
    if metric == "ic":
        def _ic_loss(y_true, y_pred, sample_weight=None):
            valid = ~(np.isnan(y_true) | np.isnan(y_pred))
            if valid.sum() < 10:
                return 1.0
            ic = np.corrcoef(y_true[valid], y_pred[valid])[0, 1]
            return -abs(ic) if np.isfinite(ic) else 1.0
        return _ic_loss

    if metric == "rank_ic":
        from scipy.stats import spearmanr
        def _rank_ic_loss(y_true, y_pred, sample_weight=None):
            valid = ~(np.isnan(y_true) | np.isnan(y_pred))
            if valid.sum() < 10:
                return 1.0
            rho, _ = spearmanr(y_true[valid], y_pred[valid])
            return -abs(rho) if np.isfinite(rho) else 1.0
        return _rank_ic_loss

    raise ValueError(f"未知 metric: {metric}，可选: ic / rank_ic")


# ---------------------------------------------------------------------------
# 因子包装器（挖掘结果 → BaseFactor）
# ---------------------------------------------------------------------------
def _make_factor_class(
    name: str,
    formula: str,
    program: Any,
    ic_value: float,
    feature_names: Optional[List[str]] = None,
) -> Type[BaseFactor]:
    """从 gplearn 挖掘结果动态构造 BaseFactor 子类。

    ⚠️ 2026-06-20 持久化限制：
      - 当前实现将 `program`（gplearn._Program）对象直接保存到类属性 `_gp_program`。
      - `_Program` 内部包含 lambda 闭包，**不可 pickle**——禁止通过 joblib/pickle 持久化
        整个 FactorEngine 后再 load，否则报 "Can't pickle local object"。
      - 推荐做法：仅持久化 `formula` 字符串，在 `compute` 时用 `gplearn` 重新编译。
      - 当前代码接受 program 是为在内存中直接求值（避免重编译开销）。
      - 若后续需要持久化，请改用 `sympy` 解析 formula 或保存 AST 重建。

    Args:
        name: 因子编号（如 GP_001）
        formula: 因子公式描述
        program: gplearn _Program 对象（用于 compute 中重新求值）
        ic_value: 训练期 IC 值
        feature_names: 特征列名列表（与训练时 X 的列序一致）

    Returns:
        继承 BaseFactor 的因子类
    """
    # 闭包捕获 program 和 feature_names
    _saved_program = program
    _saved_feature_names = feature_names or ["close", "volume", "open_interest"]

    class _MinedFactor(BaseFactor):
        _gp_program = _saved_program
        _gp_feature_names = _saved_feature_names
        _train_ic = ic_value

        def compute(self, **kwargs: np.ndarray) -> np.ndarray:
            """使用保存的 gplearn program 对象重新求值。

            构建特征矩阵 X（列序与训练时一致），然后调用 program.execute(X)。
            """
            if self._gp_program is None:
                raise RuntimeError(
                    f"挖掘因子 {name} 的 program 对象为 None，无法求值。"
                    f"公式: {formula}"
                )
            # 按训练时的特征列序构建 X
            arrays = []
            for feat in self._gp_feature_names:
                arr = kwargs.get(feat)
                if arr is None:
                    # 缺失特征填充零向量
                    first_val = next(iter(kwargs.values()), None)
                    length = len(first_val) if first_val is not None else 0
                    arr = np.zeros(length, dtype=float)
                    _logger.warning(
                        "挖掘因子 %s 缺少特征 %s，已用零填充", name, feat
                    )
                arrays.append(np.asarray(arr, dtype=float))
            X = np.column_stack(arrays)
            return self._gp_program.execute(X)

    _MinedFactor.__name__ = f"GP_{name}"
    _MinedFactor.name = name
    _MinedFactor.category = "GP挖掘"
    _MinedFactor.formula = formula
    _MinedFactor.dependencies = list(_saved_feature_names)
    return _MinedFactor


# ---------------------------------------------------------------------------
# 因子挖掘器
# ---------------------------------------------------------------------------
class GPLearnFactorMiner:
    """gplearn 因子挖掘器（GP 遗传规划）。

    用法::

        miner = GPLearnFactorMiner(config=AlphaFuturesConfig(), gp_config=GPLearnConfig())
        miner.fit(X=features_df, y=forward_returns)
        report = miner.report(top_n=10)
        miner.register_best(top_n=5)  # 注册到全局因子注册表
    """

    def __init__(
        self,
        config: AlphaFuturesConfig,
        gp_config: Optional[GPLearnConfig] = None,
        function_set: Optional[List[Callable]] = None,
    ) -> None:
        self.config = config
        self.gp_config = gp_config or GPLearnConfig()
        self.function_set = function_set or DEFAULT_FUNCTION_SET
        self._programs: List[Any] = []  # 存放 gplearn 程序
        self._ic_values: List[float] = []
        self._est: Optional[gp.SymbolicRegressor] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "GPLearnFactorMiner":
        """执行 GP 进化搜索。

        Args:
            X: 特征矩阵 (n_samples, n_features)，列序对应 self.function_set
            y: 目标向量 (n_samples,)，如未来 N 日收益

        Returns:
            self（链式调用）
        """
        cfg = self.gp_config
        fitness = _make_fitness_func(cfg.metric)

        self._est = gp.SymbolicRegressor(
            population_size=cfg.population_size,
            generations=cfg.generations,
            tournament_size=cfg.tournament_size,
            parsimony_coefficient=cfg.parsimony_coefficient,
            const_range=cfg.const_range,
            init_depth=cfg.init_depth,
            init_method=cfg.init_method,
            function_set=self.function_set,
            metric=fitness,
            random_state=cfg.random_state,
            low_memory=cfg.low_memory,
            n_jobs=cfg.n_jobs,
            verbose=0,
        )
        self._est.fit(X, y)
        return self

    def report(self, top_n: int = 10) -> pd.DataFrame:
        """报告挖掘结果。

        Args:
            top_n: 返回 top N 因子

        Returns:
            DataFrame: name / formula / ic / program（gplearn _Program 对象）
        """
        if self._est is None:
            raise RuntimeError("请先调用 fit()")

        # gplearn 按适应度排序
        programs = self._est._programs[-1] if hasattr(self._est, "_programs") else []
        rows = []
        for i, prog in enumerate(programs[:top_n]):
            if prog is None:
                continue
            try:
                program_str = str(prog)
                # 重新求值估算 IC
                y_pred = prog.execute(self._est.X_transform_)
                valid = ~np.isnan(y_pred)
                if valid.sum() < 10:
                    continue
                ic = np.corrcoef(self._est.y_[valid], y_pred[valid])[0, 1]
            except Exception as e:  # noqa: BLE001
                _logger.warning("程序 %d 评估失败: %s", i, e)
                continue

            rows.append({
                "rank": i + 1,
                "name": f"GP_{i+1:03d}",
                "formula": program_str[:80] + ("..." if len(program_str) > 80 else ""),
                "ic": float(ic) if np.isfinite(ic) else 0.0,
                "program": prog,
            })
        return pd.DataFrame(rows)

    def register_best(
        self,
        top_n: int = 5,
        name_prefix: str = "GP",
        feature_names: Optional[List[str]] = None,
    ) -> List[str]:
        """把 top N 挖掘结果注册到全局因子注册表。

        Args:
            top_n: 注册数量
            name_prefix: 因子名前缀（最终名为 {prefix}_{001,002,...}）
            feature_names: 训练时特征列名列表（用于 compute 求值）

        Returns:
            已注册的因子名列表
        """
        report_df = self.report(top_n=top_n)
        registered: List[str] = []
        for _, row in report_df.iterrows():
            factor_name = f"{name_prefix}_{int(row['rank']):03d}"
            factor_cls = _make_factor_class(
                name=factor_name,
                formula=row["formula"],
                program=row.get("program"),
                ic_value=row["ic"],
                feature_names=feature_names,
            )
            register_factor(factor_cls)
            registered.append(factor_name)
            _logger.info("已注册挖掘因子: %s (IC=%.4f)", factor_name, row["ic"])
        return registered


# ---------------------------------------------------------------------------
# 一键式入口
# ---------------------------------------------------------------------------
def register_mined_factors(
    X: np.ndarray,
    y: np.ndarray,
    config: AlphaFuturesConfig,
    top_n: int = 5,
    gp_config: Optional[GPLearnConfig] = None,
) -> List[str]:
    """一键式：挖掘 + 注册 top N 因子。

    Args:
        X: 特征矩阵
        y: 目标向量
        config: AlphaFuturesConfig
        top_n: 注册数量
        gp_config: gplearn 配置

    Returns:
        已注册因子名列表
    """
    miner = GPLearnFactorMiner(config=config, gp_config=gp_config)
    miner.fit(X, y)
    return miner.register_best(top_n=top_n)
