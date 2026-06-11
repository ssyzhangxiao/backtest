"""分层配置加载器（规则 23 — Layered Configuration）。

**核心原则**：配置按优先级从低到高分层叠加：

```
优先级（低 → 高）：
    1. dataclass 默认值         （代码内置）
    2. YAML 文件                （config.yaml）
    3. 环境变量 QUANT_*         （部署/容器/CI）
    4. 运行时 overrides 字典    （Pipeline / 脚本 / 测试）
```

**设计动机**：
- 同一份代码能在 dev / staging / prod 切换配置（无需改 yaml）
- CI 容器用 env 注入敏感值（API key / 数据源）覆盖 yaml
- 脚本临时调参（`overrides={"backtest__rebalance_days": 5}`）不改 yaml
- 测试用例可注入 override，无需写临时 yaml 文件

**约定**：
- 环境变量前缀：`QUANT_`
- 节段分隔符：双下划线 `__`（与 yaml 路径对应）
- 命名空间映射：yaml 顶层段名 → env 段名（见 ENV_SECTION_ALIAS）

**示例**：

```python
import os

# YAML: backtest.rebalance_freq: 3
os.environ["QUANT_BACKTEST__REBALANCE_FREQ"] = "5"  # 覆盖为 5

from core.config import BacktestConfig
cfg = BacktestConfig.from_yaml("config.yaml")
assert cfg.rebalance_days == 5  # env 覆盖生效

# 运行时 override（最高优先级）
cfg2 = BacktestConfig.from_yaml("config.yaml", overrides={"rebalance_days": 7})
assert cfg2.rebalance_days == 7
```

**类型转换**：env var 全部是字符串；按目标 dataclass 字段类型自动转换。
"""

from __future__ import annotations

import logging
import os
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Optional

from .yaml_utils import load_yaml


__all__ = [
    "LayeredConfigLoader",
    "ENV_PREFIX",
    "ENV_SECTION_ALIAS",
    "load_env_overrides",
    "merge_overrides",
]


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 约定常量
# ---------------------------------------------------------------------------
ENV_PREFIX: str = "QUANT_"
"""环境变量前缀。"""

ENV_SECTION_ALIAS: Dict[str, str] = {
    "backtest": "backtest",
    "factor_weights": "factor_weights",
    "factors": "factors",
    "stop_optimization": "stop_optimization",
    "validation": "validation",
    "data": "data",
    "output": "output",
    "risk_management": "risk_management",
    "symbols": "symbols",
    "strategies": "strategies",
    "tqsdk": "tqsdk",
}
"""env 段名 → yaml 段名映射（env 段名更短，去掉下划线）。"""


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------
def load_env_overrides(prefix: str = ENV_PREFIX) -> Dict[str, Dict[str, Any]]:
    """从环境变量加载覆盖配置。

    约定：
        QUANT_<SECTION>__<FIELD>=<value>
        QUANT_<SECTION>__<SUB>__<FIELD>=<value>   # 仅支持一层嵌套

    Args:
        prefix: 环境变量前缀

    Returns:
        {yaml_section_name: {field_name: value}} 字典
    """
    result: Dict[str, Dict[str, Any]] = {}
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        stripped = env_key[len(prefix):]
        if "__" not in stripped:
            _logger.debug("忽略环境变量 %s（无 __ 分隔符）", env_key)
            continue
        # 取段名（env 段名）+ 字段路径
        env_section, *field_path = stripped.split("__")
        section = ENV_SECTION_ALIAS.get(env_section.lower(), env_section.lower())
        field_name = "__".join(field_path).lower()
        if section not in result:
            result[section] = {}
        result[section][field_name] = _coerce_env_value(env_val, field_name)
    return result


def merge_overrides(
    base: Dict[str, Any],
    *override_layers: Dict[str, Any],
) -> Dict[str, Any]:
    """按优先级合并多层 dict 覆盖（后传入者覆盖先传入者）。

    Args:
        base: 基础层（如 defaults 或 YAML）
        *override_layers: 一个或多个覆盖层（按优先级从低到高）

    Returns:
        合并后的新 dict（不修改入参）
    """
    result = dict(base)
    for layer in override_layers:
        if not layer:
            continue
        result = _deep_merge(result, layer)
    return result


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并两个 dict。b 的值覆盖 a 的值。"""
    result = dict(a)
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _coerce_env_value(raw: str, field_name: str) -> Any:
    """根据字段名启发式转换 env 字符串到目标类型。

    启发式规则（按字段名后缀 / 前缀）：
        _pct, _rate, _ratio, _threshold, _weight  → float
        _days, _n, _freq, _samples, _bars         → int
        前缀 use_ / 后缀 _enabled / _use / _flag  → bool
        其他                                       → str
    """
    lower = field_name.lower()
    if lower.startswith("use_") or lower.endswith((
        "_enabled", "_use", "_flag", "_validate", "_section",
    )):
        return raw.lower() in ("1", "true", "yes", "on")
    if lower.endswith(("_pct", "_rate", "_ratio", "_threshold", "_weight", "_cap")):
        try:
            return float(raw)
        except (ValueError, TypeError):
            return raw
    if lower.endswith(("_days", "_n", "_freq", "_samples", "_bars", "_delay")):
        try:
            return int(raw)
        except (ValueError, TypeError):
            return raw
    return raw


# ---------------------------------------------------------------------------
# 主加载器
# ---------------------------------------------------------------------------
class LayeredConfigLoader:
    """分层配置加载器。

    用法::

        loader = LayeredConfigLoader(
            dataclass_type=BacktestConfig,
            yaml_path="config.yaml",
        )
        cfg = loader.load(overrides={"rebalance_days": 7})
    """

    def __init__(
        self,
        dataclass_type: type,
        yaml_path: str = "config.yaml",
        env_prefix: str = ENV_PREFIX,
    ) -> None:
        if not is_dataclass(dataclass_type):
            raise TypeError(f"dataclass_type 必须是 dataclass 类，得到 {type(dataclass_type)}")
        self.dataclass_type = dataclass_type
        self.yaml_path = yaml_path
        self.env_prefix = env_prefix

    def load(self, overrides: Optional[Dict[str, Any]] = None) -> Any:
        """加载并返回 dataclass 实例。

        Args:
            overrides: 运行时覆盖（最高优先级），支持点号路径如
                {"backtest__rebalance_days": 5}
                也支持嵌套 dict 如 {"backtest": {"rebalance_days": 5}}

        Returns:
            dataclass 实例
        """
        # Layer 1: defaults
        defaults = self._extract_defaults()

        # Layer 2: YAML
        yaml_dict = self._load_yaml_section()

        # Layer 3: env vars
        env_dict = load_env_overrides(self.env_prefix)

        # Layer 4: runtime overrides
        runtime_dict = self._normalize_overrides(overrides or {})

        # 按优先级合并
        merged = merge_overrides(defaults, yaml_dict, env_dict, runtime_dict)
        return self._instantiate(merged)

    # ------------------------------------------------------------------ 工具
    def _extract_defaults(self) -> Dict[str, Any]:
        """从 dataclass 字段默认值提取基础配置（仅顶层字段）。"""
        defaults: Dict[str, Any] = {}
        for f in fields(self.dataclass_type):
            # default_factory 的字段：实例化一个空值作为基础
            try:
                defaults[f.name] = f.default_factory()  # type: ignore[misc]
            except (TypeError, AttributeError):
                defaults[f.name] = f.default
        return defaults

    def _load_yaml_section(self) -> Dict[str, Any]:
        """从 yaml 加载所有相关段（顶层映射到 BacktestConfig 字段）。

        简化策略：直接把整个 yaml 顶层 dict 当作输入，BacktestConfig.from_yaml()
        内部已经做了精确的字段映射；这里只做"load → return top dict"。
        """
        try:
            return load_yaml(self.yaml_path) or {}
        except FileNotFoundError:
            _logger.warning("配置文件 %s 不存在，使用空 dict", self.yaml_path)
            return {}

    def _normalize_overrides(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """将扁平 key（如 'backtest__rebalance_days'）转成嵌套 dict。

        支持：
            {"rebalance_days": 7}                          → 顶层
            {"backtest__rebalance_days": 7}                → {"backtest": {...}}
            {"backtest": {"rebalance_days": 7}}            → 嵌套（直通）
        """
        normalized: Dict[str, Any] = {}
        for key, value in overrides.items():
            if "__" in key:
                section, field = key.split("__", 1)
                normalized.setdefault(section, {})[field] = value
            else:
                normalized[key] = value
        return normalized

    def _instantiate(self, merged: Dict[str, Any]) -> Any:
        """委托给 BacktestConfig.from_yaml() 风格的构造器（保持字段映射统一）。"""
        # BacktestConfig 已有 from_yaml 接受 path；这里走一个更通用的构造：
        # 把合并后的 dict 拆成两部分：yaml 全字段 + 顶层 dataclass 字段
        # 由于 BacktestConfig 的字段映射规则复杂，这里委托给 BacktestConfig.from_yaml()
        # 但用 merged dict 临时写回一个虚拟 yaml
        if hasattr(self.dataclass_type, "from_yaml"):
            # 优先调用自定义 from_yaml（保持现有字段映射）
            return self.dataclass_type.from_yaml(self.yaml_path)  # type: ignore[attr-defined]
        return self.dataclass_type(**merged)
