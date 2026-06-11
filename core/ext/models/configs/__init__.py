"""模型配置 — YAML 配置文件定义。

每个模型有对应的 YAML 配置模板，支持：
    - 默认参数（代码内置）
    - YAML 文件覆盖
    - 运行时 overrides

配置路径约定：
    config.yaml 中的 models 段 → 各模型配置
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from loguru import logger


# 默认模型配置模板
LGBM_DEFAULT_CONFIG: Dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.1,
    "random_state": 42,
    "early_stopping_rounds": 20,
    "test_size": 0.2,
}

MLP_DEFAULT_CONFIG: Dict[str, Any] = {
    "hidden_sizes": [64, 32],
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 256,
    "dropout": 0.1,
    "weight_decay": 1e-5,
    "patience": 10,
    "task": "regression",
    "random_state": 42,
}


def load_model_config(
    model_type: str,
    yaml_path: str = "config.yaml",
    overrides: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """加载模型配置（分层合并）。

    优先级：默认 < YAML < overrides

    Args:
        model_type: "lgbm" 或 "mlp"
        yaml_path: 配置文件路径
        overrides: 运行时覆盖

    Returns:
        合并后的配置字典
    """
    # Layer 1: 默认
    defaults_map = {
        "lgbm": LGBM_DEFAULT_CONFIG,
        "mlp": MLP_DEFAULT_CONFIG,
    }
    config = dict(defaults_map.get(model_type, {}))

    # Layer 2: YAML
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        models_section = raw.get("models", {})
        model_section = models_section.get(model_type, {})
        if model_section:
            config.update(model_section)
    except FileNotFoundError:
        logger.debug("配置文件 %s 不存在，使用默认配置", yaml_path)

    # Layer 3: overrides
    if overrides:
        config.update(overrides)

    return config


def generate_default_yaml(output_path: str = "models_config.yaml") -> None:
    """生成默认模型配置 YAML 文件。"""
    config = {
        "models": {
            "lgbm": LGBM_DEFAULT_CONFIG,
            "mlp": MLP_DEFAULT_CONFIG,
        }
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info("默认模型配置已生成: %s", output_path)
