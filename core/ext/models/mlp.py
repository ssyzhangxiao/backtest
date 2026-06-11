"""MLP 预测器（PyTorch）。

基于因子特征的多层感知机预测模型，支持回归和分类。
依赖：torch（按需安装，requirements-models.txt）

复用约束（规则21.4）：
    - 特征通过 core/factors/factor_pipeline.py 的 FactorPipeline 获取
    - 评估指标复用 core/factors/factor_evaluator.py 的 FactorEvaluator
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.ext.models.base import BasePredictor


class MLPPredictor(BasePredictor):
    """PyTorch MLP 预测器。

    用法::

        predictor = MLPPredictor(config={
            "hidden_sizes": [64, 32],
            "learning_rate": 1e-3,
            "epochs": 100,
            "batch_size": 256,
            "dropout": 0.1,
        })
        predictor.fit(X_train, y_train)
        pred = predictor.predict(X_test)
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "hidden_sizes": [64, 32],
        "learning_rate": 1e-3,
        "epochs": 100,
        "batch_size": 256,
        "dropout": 0.1,
        "weight_decay": 1e-5,
        "patience": 10,
        "task": "regression",  # "regression" or "classification"
        "random_state": 42,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(config=merged)
        self._model = None
        self._device = None

    def _build_model(self, input_dim: int):
        """构建 MLP 模型。"""
        import torch
        import torch.nn as nn

        hidden_sizes = self.config.get("hidden_sizes", [64, 32])
        dropout = self.config.get("dropout", 0.1)
        task = self.config.get("task", "regression")

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h

        # 输出层
        output_dim = 1 if task == "regression" else 2
        layers.append(nn.Linear(prev_dim, output_dim))

        model = nn.Sequential(*layers)
        return model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs,
    ) -> "MLPPredictor":
        """训练 MLP 模型。

        Args:
            X: 特征 DataFrame
            y: 标签 Series
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "PyTorch 未安装，请运行: pip install torch "
                "或 pip install -r requirements-models.txt"
            )

        self._feature_names = list(X.columns)
        task = self.config.get("task", "regression")
        lr = self.config.get("learning_rate", 1e-3)
        epochs = self.config.get("epochs", 100)
        batch_size = self.config.get("batch_size", 256)
        weight_decay = self.config.get("weight_decay", 1e-5)
        patience = self.config.get("patience", 10)

        # 设备
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 数据准备
        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(np.float32)

        # 替换 NaN/Inf
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)

        X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self._device)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32).to(self._device)

        if task == "classification":
            y_tensor = y_tensor.long()

        # 构建模型
        self._model = self._build_model(X_arr.shape[1]).to(self._device)

        # 损失函数和优化器
        if task == "regression":
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # 训练
        self._model.train()
        n_samples = len(X_arr)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Mini-batch
            indices = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]

                X_batch = X_tensor[batch_idx]
                y_batch = y_tensor[batch_idx]

                optimizer.zero_grad()
                output = self._model(X_batch)

                if task == "regression":
                    loss = criterion(output.squeeze(), y_batch)
                else:
                    loss = criterion(output, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            # Early stopping
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        self._is_fitted = True
        self._train_stats = {
            "n_samples": n_samples,
            "n_features": X_arr.shape[1],
            "best_loss": round(best_loss, 6),
            "final_epoch": epoch + 1,
            "device": str(self._device),
        }

        logger.info(
            "MLP 训练完成: %d 样本, %d 特征, best_loss=%.6f, epochs=%d",
            n_samples, X_arr.shape[1], best_loss, epoch + 1,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测。"""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch 未安装")

        if not self._is_fitted or self._model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")

        self._model.eval()
        X_arr = np.nan_to_num(X.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            output = self._model(X_tensor)

        task = self.config.get("task", "regression")
        if task == "classification":
            probs = torch.softmax(output, dim=1)
            return probs[:, 1].cpu().numpy()  # 正类概率
        return output.squeeze().cpu().numpy()

    def save(self, path: Path) -> None:
        """保存模型。"""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch 未安装")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._model is not None:
            torch.save(self._model.state_dict(), path / "model.pt")

        self._save_meta(path)
        logger.info("MLP 模型已保存到: %s", path)

    def load(self, path: Path) -> "MLPPredictor":
        """加载模型。

        设备恢复策略：
          1. 优先使用保存时的设备（从 meta.json 读取）
          2. 若保存设备为 CUDA 但当前不可用，回退到 CPU
          3. 通过 map_location 确保张量正确映射到目标设备
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch 未安装")

        path = Path(path)
        meta = self._load_meta(path)
        self.config.update(meta.get("config", {}))

        model_path = path / "model.pt"
        if model_path.exists():
            # 恢复保存时的设备，CUDA 不可用时回退 CPU
            saved_device_str = meta.get("train_stats", {}).get("device", "cpu")
            if "cuda" in saved_device_str and not torch.cuda.is_available():
                logger.warning(
                    "模型保存于 %s 但当前 CUDA 不可用，回退到 CPU", saved_device_str
                )
                self._device = torch.device("cpu")
            else:
                self._device = torch.device(
                    saved_device_str if torch.cuda.is_available() else "cpu"
                )

            n_features = len(self._feature_names) if self._feature_names else 1
            self._model = self._build_model(n_features).to(self._device)
            self._model.load_state_dict(
                torch.load(model_path, map_location=self._device, weights_only=True)
            )
            self._model.eval()
            self._is_fitted = True

        logger.info("MLP 模型已加载: %s (设备: %s)", path, self._device)
        return self
