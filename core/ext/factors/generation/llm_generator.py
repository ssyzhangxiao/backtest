"""LLM 因子生成器（规则21 + 规则21.4）。

目标：使用大语言模型（GPT-4 / Claude）从自然语言描述生成因子表达式。

**核心设计**：
    1. **领域 Prompt 模板**：内置期货因子领域知识，引导 LLM 输出合规公式
    2. **沙箱执行**：LLM 输出是 Python 表达式，调用前用 AST 校验 + 安全求值
    3. **复用 BaseFactor**：生成的因子必须继承 BaseFactor 并通过 register_factor 注册
    4. **复用核心算子**：只能调用 core.factors.operators 中的算子 + 基础数学函数
    5. **可插拔 LLM 后端**：默认 OpenAI，可换 Anthropic / 本地模型

**依赖**：
    pip install -r requirements-llm.txt
    或：pip install openai>=1.0.0 anthropic

**失败行为**（规则 21.2）：
    未安装 openai 时 import 此模块会立即抛 ImportError，不会污染 core/。
    不得在调用方 try/except 兜底。
"""

from __future__ import annotations

import ast
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

# 规则 21.2：第三方 import 必须直接，不得 try/except 兜底
from openai import OpenAI

from core.ext.factors import operators as ops
from core.ext.factors.alpha_futures.base_factor import BaseFactor
from core.ext.factors.alpha_futures.config import AlphaFuturesConfig
from core.ext.factors.alpha_futures.factor_registry import register_factor


__all__ = [
    "LLMFactorGenerator",
    "LLMGeneratorConfig",
    "FACTOR_DOMAIN_PROMPT",
    "ALLOWED_OPS",
    "ALLOWED_FUNCS",
    "register_llm_factors",
]


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 领域 Prompt 模板
# ---------------------------------------------------------------------------
FACTOR_DOMAIN_PROMPT: str = """你是商品期货量化因子专家。请基于以下约束生成因子表达式：

## 可用算子（必须使用）
- 基础: sma(x, n), std(x, n), mean(x, n), ema(x, n)
- 差分: delta(x, n), delay(x, n)
- 相关: corr(x, y, n), tsrank(x, n)
- 算术: +, -, *, /（除法自动 safe_div）
- 数学: abs(x), log(x), sign(x), neg(x)

## 输入字段（只能使用这些）
- close, open, high, low, volume, open_interest
- 衍生: returns, hl_range, oc_range（已计算的常用衍生量）

## 输出要求
1. 仅返回一个 Python 表达式字符串，禁止其他文字
2. 表达式必须仅引用上述算子和字段
3. 公式逻辑清晰，长度 ≤ 100 字符
4. 建议加注释（如 # 注释）但仅在 # 后面

## 示例
输出: -corr(close, volume, 20)
输出: delta(close, 5) / close
输出: tsrank(volume, 20) - tsrank(close, 20)  # 量价背离

## 任务
基于用户描述，输出 1 个因子表达式："""


# ---------------------------------------------------------------------------
# 沙箱：允许的算子与函数白名单
# ---------------------------------------------------------------------------
ALLOWED_OPS: Dict[str, Callable] = {
    "sma": lambda x, n: ops.sma(x, int(n)),
    "std": lambda x, n: ops.std(x, int(n)),
    "mean": lambda x, n: ops.mean(x, int(n)),
    "ema": lambda x, n: ops.ema(x, int(n)),
    "delta": lambda x, n: ops.delta(x, int(n)),
    "delay": lambda x, n: ops.delay(x, int(n)),
    "corr": lambda x, y, n: ops.corr(x, y, int(n)),
    "tsrank": lambda x, n: ops.tsrank(x, int(n)),
}

ALLOWED_FUNCS: Dict[str, Callable] = {
    "abs": np.abs,
    "log": lambda x: np.log(np.maximum(x, 1e-12)),
    "sign": np.sign,
    "neg": lambda x: -x,
    "sqrt": np.sqrt,
}

ALLOWED_INPUTS: Set[str] = {
    "close", "open", "high", "low", "volume", "open_interest",
    "returns", "hl_range", "oc_range",
}


# ---------------------------------------------------------------------------
# AST 校验（防止恶意代码）
# ---------------------------------------------------------------------------
class _FactorFormulaValidator(ast.NodeVisitor):
    """AST 校验器：拒绝任何未在白名单的函数/字段/语法。"""

    def __init__(self) -> None:
        self.errors: List[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        # 检查函数名
        if isinstance(node.func, ast.Name):
            if node.func.id not in ALLOWED_OPS and node.func.id not in ALLOWED_FUNCS:
                self.errors.append(f"未授权函数: {node.func.id}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in ALLOWED_INPUTS and node.id not in ALLOWED_OPS \
                and node.id not in ALLOWED_FUNCS:
            self.errors.append(f"未授权变量: {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.errors.append(f"禁止属性访问: .{node.attr}")

    def visit_Import(self, node: ast.Import) -> None:
        self.errors.append("禁止 import 语句")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.errors.append("禁止 from ... import 语句")


def validate_formula(formula: str) -> List[str]:
    """校验因子公式的合法性。返回错误列表（空列表 = 合法）。"""
    # 去掉注释
    code = "\n".join(line.split("#")[0] for line in formula.split("\n")).strip()
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as e:
        return [f"语法错误: {e}"]

    validator = _FactorFormulaValidator()
    validator.visit(tree)
    return validator.errors


# ---------------------------------------------------------------------------
# 沙箱求值
# ---------------------------------------------------------------------------
def safe_eval_formula(formula: str, data: Dict[str, np.ndarray]) -> np.ndarray:
    """安全求值因子公式。

    Args:
        formula: 公式字符串
        data: 输入字段字典 {close: ndarray, volume: ndarray, ...}

    Returns:
        因子值 ndarray

    Raises:
        ValueError: 公式不合法或求值失败
    """
    errors = validate_formula(formula)
    if errors:
        raise ValueError(f"公式校验失败: {errors}")

    code = "\n".join(line.split("#")[0] for line in formula.split("\n")).strip()
    safe_globals: Dict[str, Any] = {}
    safe_globals.update(ALLOWED_OPS)
    safe_globals.update(ALLOWED_FUNCS)
    safe_locals: Dict[str, Any] = dict(data)

    return eval(code, safe_globals, safe_locals)  # noqa: S307 — 已通过 AST 校验


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
class LLMGeneratorConfig:
    """LLM 因子生成配置。

    Attributes:
        model: 模型名（"gpt-4o" / "gpt-4o-mini" / "claude-3-5-sonnet-20241022"）
        n_candidates: 每次调用生成的候选数
        max_retries: API 调用失败重试次数
        temperature: 采样温度（0.0 = 确定性，1.0 = 创造性）
        max_tokens: 单次响应 token 上限
        timeout_seconds: API 超时
        base_url: 自定义 OpenAI 兼容端点（用于 Azure / 本地 vLLM）
        api_key: API 密钥（默认从环境变量 OPENAI_API_KEY 读取）
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        n_candidates: int = 5,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 500,
        timeout_seconds: float = 30.0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.n_candidates = n_candidates
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# LLM 因子生成器
# ---------------------------------------------------------------------------
class LLMFactorGenerator:
    """LLM 因子生成器。

    用法::

        gen = LLMFactorGenerator(
            config=AlphaFuturesConfig(),
            llm_config=LLMGeneratorConfig(model="gpt-4o-mini"),
        )
        formulas = gen.generate("生成一个动量反转因子")
        valid = gen.filter_valid(formulas, data={"close": close, "volume": volume})
        gen.register_best(valid[:3], name_prefix="LLM")
    """

    def __init__(
        self,
        config: AlphaFuturesConfig,
        llm_config: Optional[LLMGeneratorConfig] = None,
    ) -> None:
        self.config = config
        self.llm_config = llm_config or LLMGeneratorConfig()
        if not self.llm_config.api_key:
            raise ValueError(
                "LLMFactorGenerator 需要 OPENAI_API_KEY 环境变量或 api_key 参数"
            )
        self._client = OpenAI(
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.base_url,
            timeout=self.llm_config.timeout_seconds,
        )

    def generate(self, description: str) -> List[str]:
        """根据自然语言描述生成候选因子公式。

        Args:
            description: 自然语言描述，如 "量价背离反转因子"

        Returns:
            候选公式字符串列表（未做校验）
        """
        cfg = self.llm_config
        prompt = FACTOR_DOMAIN_PROMPT + "\n" + description

        for attempt in range(cfg.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=cfg.model,
                    messages=[
                        {"role": "system", "content": FACTOR_DOMAIN_PROMPT},
                        {"role": "user", "content": description},
                    ],
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    n=cfg.n_candidates,
                )
                formulas = [
                    choice.message.content.strip()
                    for choice in resp.choices
                    if choice.message.content
                ]
                _logger.info("LLM 生成 %d 个候选公式", len(formulas))
                return formulas
            except Exception as e:  # noqa: BLE001
                _logger.warning(
                    "LLM 调用失败 (attempt %d/%d): %s",
                    attempt + 1, cfg.max_retries, e,
                )
                if attempt + 1 < cfg.max_retries:
                    time.sleep(2 ** attempt)
                else:
                    raise

        return []

    def filter_valid(
        self,
        formulas: List[str],
        data: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """筛选合法且能求值的公式。

        Args:
            formulas: 候选公式列表
            data: 测试数据，用于试求值

        Returns:
            合法公式的详细信息列表：{name, formula, valid}
        """
        results: List[Dict[str, Any]] = []
        for i, formula in enumerate(formulas, 1):
            errors = validate_formula(formula)
            if errors:
                _logger.debug("公式 %d 校验失败: %s", i, errors)
                continue
            try:
                values = safe_eval_formula(formula, data)
                if not isinstance(values, np.ndarray) or values.shape != data["close"].shape:
                    _logger.debug("公式 %d 输出 shape 不匹配: %s", i, getattr(values, "shape", None))
                    continue
                results.append({
                    "rank": len(results) + 1,
                    "name": f"LLM_{len(results)+1:03d}",
                    "formula": formula,
                    "valid": True,
                })
            except Exception as e:  # noqa: BLE001
                _logger.debug("公式 %d 求值失败: %s", i, e)
                continue
        return results

    def register_best(
        self,
        valid_formulas: List[Dict[str, Any]],
        name_prefix: str = "LLM",
    ) -> List[str]:
        """把合法公式注册为 BaseFactor 因子。

        Args:
            valid_formulas: filter_valid() 的输出
            name_prefix: 因子名前缀

        Returns:
            已注册的因子名列表
        """
        registered: List[str] = []
        for entry in valid_formulas:
            factor_name = f"{name_prefix}_{entry['rank']:03d}"

            class _LLMFactor(BaseFactor):
                def compute(self, **kwargs: np.ndarray) -> np.ndarray:
                    return safe_eval_formula(formula, kwargs)

            _LLMFactor.__name__ = f"LLMFactor_{factor_name}"
            _LLMFactor.name = factor_name
            _LLMFactor.category = "LLM生成"
            _LLMFactor.formula = entry["formula"]
            _LLMFactor.dependencies = list(ALLOWED_INPUTS)
            register_factor(_LLMFactor)
            registered.append(factor_name)
            _logger.info("已注册 LLM 因子: %s (formula=%s)", factor_name, entry["formula"])
        return registered


# ---------------------------------------------------------------------------
# 一键式入口
# ---------------------------------------------------------------------------
def register_llm_factors(
    description: str,
    data: Dict[str, np.ndarray],
    config: AlphaFuturesConfig,
    top_n: int = 3,
    llm_config: Optional[LLMGeneratorConfig] = None,
) -> List[str]:
    """一键式：LLM 生成 + 筛选 + 注册 top N 因子。

    Args:
        description: 自然语言描述
        data: 测试数据 {close: ndarray, volume: ndarray, ...}
        config: AlphaFuturesConfig
        top_n: 注册数量
        llm_config: LLM 配置

    Returns:
        已注册因子名列表
    """
    gen = LLMFactorGenerator(config=config, llm_config=llm_config)
    formulas = gen.generate(description)
    valid = gen.filter_valid(formulas, data=data)
    return gen.register_best(valid[:top_n], name_prefix="LLM")
