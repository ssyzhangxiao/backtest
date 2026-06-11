"""预测模型（models/）— LGBM / MLP 预测器。

模块结构：
  - base.py        BasePredictor 抽象基类
  - lgbm.py        LightGBM 预测器（按需安装 lightgbm）
  - mlp.py         PyTorch MLP 预测器（按需安装 torch）
  - configs/       模型配置管理

复用约束（规则21.4）：
    必须复用 core/factors/factor_pipeline.py 的 pipeline 编排接口，
    训练/推理时复用 core/factors/factor_evaluator.py 的评估指标。
"""

from __future__ import annotations

from core.ext.models.base import BasePredictor
from core.ext.models.lgbm import LGBMPredictor
from core.ext.models.mlp import MLPPredictor
from core.ext.models.configs import load_model_config

__all__ = [
    "BasePredictor",
    "LGBMPredictor",
    "MLPPredictor",
    "load_model_config",
]
