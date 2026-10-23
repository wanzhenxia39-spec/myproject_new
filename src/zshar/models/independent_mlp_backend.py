"""Small PyTorch MLP trained independently for one attribute."""

from __future__ import annotations

import os
import random
import time
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .base import BaseAttributeBackend, ordered_labels
from .serialization import BACKEND_METADATA, load_json, save_json


MODEL_FILE = "independent_mlp.pt"


def set_torch_seed(seed: int) -> None:
    # Required by CUDA >= 10.2 when deterministic PyTorch algorithms are enabled.
    # Set this before importing torch so CuBLAS initializes with a reproducible
    # workspace configuration.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def resolve_device(requested: str):
    import torch

    requested = str(requested or "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(f"Requested MLP device {requested}, but CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    try:
        return torch.device(requested)
    except Exception as exc:
        raise ValueError(f"Invalid --mlp_device: {requested}") from exc


def dropout_schedule(dropout: float, layer_count: int) -> list[float]:
    return [max(0.0, float(dropout) - 0.1 * index) for index in range(layer_count)]


def build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int, dropout: float):
    import torch.nn as nn

    layers: list[nn.Module] = []
    previous = int(input_dim)
    for width, drop in zip(hidden_dims, dropout_schedule(dropout, len(hidden_dims))):
        layers.extend([nn.Linear(previous, int(width)), nn.ReLU(), nn.Dropout(drop)])
        previous = int(width)
    layers.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*layers)


def normalization_stats(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = X.to_numpy(dtype=np.float32)
    mean = np.nanmean(values, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.nanstd(values, axis=0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0).astype(np.float32)
    return mean, std


def normalize_features(X: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    values = X.to_numpy(dtype=np.float32)
    values = np.where(np.isfinite(values), values, mean)
    return ((values - mean) / std).astype(np.float32)


def balanced_class_weights(encoded: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(encoded, minlength=num_classes).astype(np.float32)
    weights = len(encoded) / (num_classes * np.maximum(counts, 1.0))
    return weights.astype(np.float32)


class IndependentMLPBackend(BaseAttributeBackend):
    backend_name = "mlp_independent"

    def __init__(
        self,
        label=None,
        hidden_dims=(256, 128),
        dropout=0.3,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        epochs=200,
        patience=20,
        device="cuda:0",
        random_state=42,
        time_limit=None,
        **config,
    ):
        hidden_dims = [int(value) for value in hidden_dims]
        super().__init__(
            label=label,
            hidden_dims=hidden_dims,
            dropout=float(dropout),
            lr=float(lr),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            epochs=int(epochs),
            patience=int(patience),
            device=str(device),
            random_state=int(random_state),
            time_limit=None if time_limit is None else float(time_limit),
            **config,
        )
        self.model = None
        self.labels: list[str] = []
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.best_epoch = 0
        self.best_val_loss = float("inf")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        seed = int(self.config["random_state"])
        set_torch_seed(seed)
        device = resolve_device(self.config["device"])
        self.feature_names = list(X_train.columns)
        self.labels = ordered_labels(y_train.astype(str).tolist())
        label_to_index = {label: index for index, label in enumerate(self.labels)}
        y_encoded = np.asarray([label_to_index[str(value)] for value in y_train], dtype=np.int64)
        self.mean, self.std = normalization_stats(X_train)
        train_values = normalize_features(X_train, self.mean, self.std)
        self.model = build_mlp(
            train_values.shape[1],
            self.config["hidden_dims"],
            len(self.labels),
            self.config["dropout"],
        ).to(device)
        class_weights = torch.tensor(
            balanced_class_weights(y_encoded, len(self.labels)), dtype=torch.float32, device=device
        )
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_values), torch.from_numpy(y_encoded)),
            batch_size=self.config["batch_size"],
            shuffle=True,
            generator=generator,
        )
        val_values = normalize_features(X_val, self.mean, self.std) if X_val is not None else train_values
        val_labels = y_val if y_val is not None else y_train
        val_encoded = np.asarray([label_to_index.get(str(value), -1) for value in val_labels], dtype=np.int64)
        valid_mask = val_encoded >= 0
        if valid_mask.any():
            val_x_tensor = torch.from_numpy(val_values[valid_mask]).to(device)
            val_y_tensor = torch.from_numpy(val_encoded[valid_mask]).to(device)
        else:
            val_x_tensor = torch.from_numpy(train_values).to(device)
            val_y_tensor = torch.from_numpy(y_encoded).to(device)
        best_state = None
        stale_epochs = 0
        fit_started = time.monotonic()
        for epoch in range(1, self.config["epochs"] + 1):
            self.model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(batch_x.to(device)), batch_y.to(device))
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                val_loss = float(criterion(self.model(val_x_tensor), val_y_tensor).item())
            if val_loss < self.best_val_loss - 1e-7:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config["patience"]:
                    break
            time_limit = self.config.get("time_limit")
            if time_limit is not None and time.monotonic() - fit_started >= float(time_limit):
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to("cpu")
        return self

    def _logits(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        if self.model is None or self.mean is None or self.std is None:
            raise RuntimeError("MLP model is not fitted.")
        device = resolve_device(self.config["device"])
        self.model.to(device).eval()
        values = torch.from_numpy(normalize_features(X, self.mean, self.std))
        outputs = []
        with torch.no_grad():
            for start in range(0, len(values), self.config["batch_size"]):
                outputs.append(self.model(values[start : start + self.config["batch_size"]].to(device)).cpu())
        self.model.to("cpu")
        return torch.cat(outputs, dim=0).numpy() if outputs else np.empty((0, len(self.labels)))

    def predict(self, X: pd.DataFrame) -> pd.Series:
        indices = np.argmax(self._logits(X), axis=1)
        return pd.Series([self.labels[index] for index in indices], index=X.index, name=self.label)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        logits = self._logits(X)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return pd.DataFrame(exp / exp.sum(axis=1, keepdims=True), index=X.index, columns=self.labels)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return self._logits(X)

    def save(self, path: str) -> None:
        import torch

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                "network_config": {
                    "input_dim": len(self.feature_names),
                    "hidden_dims": self.config["hidden_dims"],
                    "dropout": self.config["dropout"],
                    "output_dim": len(self.labels),
                },
                "label_mapping": {label: index for index, label in enumerate(self.labels)},
                "mean": torch.from_numpy(self.mean),
                "std": torch.from_numpy(self.std),
            },
            directory / MODEL_FILE,
        )
        save_json(
            directory / BACKEND_METADATA,
            {
                "backend": self.backend_name,
                "label": self.label,
                "labels": self.labels,
                "config": self.config,
                "feature_names": self.feature_names,
                "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss,
            },
        )

    @classmethod
    def load(cls, path: str, **kwargs: Any):
        import torch

        payload = load_json(Path(path) / BACKEND_METADATA)
        model = cls(label=payload.get("label"), **payload.get("config", {}))
        state = torch.load(Path(path) / MODEL_FILE, map_location="cpu", weights_only=True)
        network = state["network_config"]
        model.model = build_mlp(
            network["input_dim"], network["hidden_dims"], network["output_dim"], network["dropout"]
        )
        model.model.load_state_dict(state["state_dict"])
        model.labels = list(payload.get("labels", state["label_mapping"].keys()))
        model.mean = state["mean"].numpy()
        model.std = state["std"].numpy()
        model.feature_names = list(payload.get("feature_names", []))
        model.best_epoch = int(payload.get("best_epoch", 0))
        model.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        return model

    def get_metadata(self):
        result = super().get_metadata()
        result.update({"labels": self.labels, "best_epoch": self.best_epoch, "best_val_loss": self.best_val_loss})
        return result
