"""LightGBM 预测器。

基于因子特征预测前瞻收益，支持回归和分类两种模式。
依赖：lightgbm（按需安装，requirements-models.txt）

复用约束（规则21.4）：
    - 特征通过 core/factors/factor_pipeline.py 的 FactorPipeline 获取
    - 评估指标复用 core/factors/factor_evaluator.py 的 FactorEvaluator
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.ext.models.base import BasePredictor


class LGBMPredictor(BasePredictor):
    """LightGBM 预测器。

    支持回归（objective=regression）和分类（objective=binary/multiclass）。

    用法::

        predictor = LGBMPredictor(config={
            "objective": "regression",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 200,
        })
        predictor.fit(X_train, y_train)
        pred = predictor.predict(X_test)
        metrics = predictor.evaluate(X_test, y_test)
    """

    # 默认参数
    DEFAULT_CONFIG: Dict[str, Any] = {
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
        "verbose": -1,
        "early_stopping_rounds": 20,
        "test_size": 0.2,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(config=merged)
        self._model = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[tuple] = None,
        **kwargs,
    ) -> "LGBMPredictor":
        """训练 LightGBM 模型。

        Args:
            X: 特征 DataFrame
            y: 标签 Series
            eval_set: 可选验证集 (X_val, y_val)
            **kwargs: 额外传递给 lgb.train 的参数
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM 未安装，请运行: pip install lightgbm "
                "或 pip install -r requirements-models.txt"
            )

        self._feature_names = list(X.columns)

        # 分离 LGB 参数和 fit 参数（不修改 self.config，避免多次 fit 时参数丢失）
        early_stopping = self.config.get("early_stopping_rounds", 20)
        test_size = self.config.get("test_size", 0.2)
        n_estimators = self.config.get("n_estimators", 200)

        # 构建 LGB 训练参数（排除 fit 专用参数）
        lgb_params = {
            k: v for k, v in self.config.items()
            if k not in ("early_stopping_rounds", "test_size", "n_estimators")
        }

        # 构建训练/验证集
        if eval_set is None and test_size > 0 and len(X) > 50:
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self._feature_names)

        valid_data = None
        callbacks = []
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, feature_name=self._feature_names, reference=train_data)
            callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
            callbacks.append(lgb.log_evaluation(period=0))

        # 训练
        self._model = lgb.train(
            params=lgb_params,
            train_set=train_data,
            num_boost_round=n_estimators,
            valid_sets=[valid_data] if valid_data else None,
            callbacks=callbacks,
        )

        self._is_fitted = True
        self._train_stats = {
            "n_samples": len(X_train),
            "n_features": len(self._feature_names),
            "best_iteration": self._model.best_iteration if hasattr(self._model, "best_iteration") else n_estimators,
        }

        logger.info(
            "LGBM 训练完成: %d 样本, %d 特征, best_iteration=%d",
            self._train_stats["n_samples"],
            self._train_stats["n_features"],
            self._train_stats["best_iteration"],
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测。"""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        return self._model.predict(X, num_iteration=self._model.best_iteration)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """获取特征重要性。

        Args:
            importance_type: "gain" 或 "split"

        Returns:
            特征重要性 Series（降序排列）
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("模型未训练")
        imp = self._model.feature_importance(importance_type=importance_type)
        return pd.Series(imp, index=self._feature_names).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        """保存模型。"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            self._model.save_model(str(path / "model.lgb"))
        self._save_meta(path)
        logger.info("LGBM 模型已保存到: %s", path)

    def load(self, path: Path) -> "LGBMPredictor":
        """加载模型。"""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM 未安装")

        path = Path(path)
        meta = self._load_meta(path)
        self.config.update(meta.get("config", {}))
        model_path = path / "model.lgb"
        if model_path.exists():
            self._model = lgb.Booster(model_file=str(model_path))
            self._is_fitted = True
        logger.info("LGBM 模型已加载: %s", path)
        return self
