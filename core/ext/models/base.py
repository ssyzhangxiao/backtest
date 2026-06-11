"""预测模型抽象基类。

所有预测模型（LGBM、MLP 等）必须继承 BasePredictor，
实现 fit / predict / save / load 四个核心方法。

复用约束（规则21.4）：
    - 训练数据通过 core/factors/factor_pipeline.py 的 FactorPipeline 获取
    - 评估指标复用 core/factors/factor_evaluator.py 的 FactorEvaluator
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class BasePredictor(ABC):
    """预测模型抽象基类。

    子类必须实现：
        - fit(X, y, **kwargs) -> self
        - predict(X) -> np.ndarray
        - save(path) -> None
        - load(path) -> self

    用法::

        class MyPredictor(BasePredictor):
            def __init__(self, config):
                super().__init__(config)
                self.model = None

            def fit(self, X, y, **kwargs):
                self.model = ...
                return self

            def predict(self, X):
                return self.model.predict(X)

            def save(self, path):
                ...

            def load(self, path):
                ...
                return self
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._is_fitted: bool = False
        self._feature_names: List[str] = []
        self._train_stats: Dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        """模型是否已训练。"""
        return self._is_fitted

    @property
    def feature_names(self) -> List[str]:
        """训练时使用的特征名列表。"""
        return self._feature_names

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs,
    ) -> "BasePredictor":
        """训练模型。

        Args:
            X: 特征 DataFrame（行=样本，列=因子）
            y: 标签 Series（前瞻收益或分类标签）
            **kwargs: 模型特定参数

        Returns:
            self（支持链式调用）
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测。

        Args:
            X: 特征 DataFrame

        Returns:
            预测值数组（回归为连续值，分类为概率）
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """保存模型到磁盘。"""

    @abstractmethod
    def load(self, path: Path) -> "BasePredictor":
        """从磁盘加载模型。"""

    def fit_predict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs,
    ) -> np.ndarray:
        """训练并预测（便利方法）。"""
        self.fit(X, y, **kwargs)
        return self.predict(X)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """评估模型预测质量。

        Args:
            X: 特征
            y: 真实标签

        Returns:
            {"ic": ..., "rmse": ..., "direction_accuracy": ...}
        """
        pred = self.predict(X)
        if len(pred) != len(y):
            logger.warning("预测值与标签长度不一致: %d vs %d", len(pred), len(y))
            return {"ic": 0.0, "rmse": float("inf"), "direction_accuracy": 0.0}

        pred_arr = np.asarray(pred, dtype=float)
        true_arr = np.asarray(y, dtype=float)

        # IC（Spearman 相关系数）
        mask = np.isfinite(pred_arr) & np.isfinite(true_arr)
        if mask.sum() < 5:
            return {"ic": 0.0, "rmse": float("inf"), "direction_accuracy": 0.0}

        pred_clean = pred_arr[mask]
        true_clean = true_arr[mask]

        # Rank IC
        from scipy.stats import spearmanr
        ic, _ = spearmanr(pred_clean, true_clean)

        # RMSE
        rmse = float(np.sqrt(np.mean((pred_clean - true_clean) ** 2)))

        # 方向准确率
        if np.std(true_clean) > 1e-10:
            direction_pred = np.sign(pred_clean)
            direction_true = np.sign(true_clean)
            direction_acc = float(np.mean(direction_pred == direction_true))
        else:
            direction_acc = 0.0

        return {
            "ic": round(float(ic), 6),
            "rmse": round(rmse, 6),
            "direction_accuracy": round(direction_acc, 4),
        }

    def _save_meta(self, path: Path) -> None:
        """保存元信息（特征名、训练统计等）。"""
        meta = {
            "config": self.config,
            "feature_names": self._feature_names,
            "train_stats": self._train_stats,
            "is_fitted": self._is_fitted,
            "predictor_type": self.__class__.__name__,
        }
        meta_path = path / "meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    def _load_meta(self, path: Path) -> Dict[str, Any]:
        """加载元信息。"""
        meta_path = path / "meta.json"
        if not meta_path.exists():
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._feature_names = meta.get("feature_names", [])
        self._train_stats = meta.get("train_stats", {})
        self._is_fitted = meta.get("is_fitted", False)
        return meta
