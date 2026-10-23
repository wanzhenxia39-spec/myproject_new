"""Native fastai and PyTorch tabular neural-network backends.

Both implementations train directly through their named library and expose
the same single-attribute interface as the other independent backends.  No
AutoGluon classes, preprocessing, model selection, or ensembling are used.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .base import BaseAttributeBackend, ordered_labels
from .independent_mlp_backend import (
    balanced_class_weights,
    normalization_stats,
    normalize_features,
    resolve_device,
    set_torch_seed,
)
from .serialization import BACKEND_METADATA, load_json, save_json


FASTAI_MODEL_FILE = "neuralnet_fastai.pt"
TORCH_MODEL_FILE = "neuralnet_torch.pt"


def _hidden_dims(values: Sequence[int]) -> list[int]:
    result = [int(value) for value in values]
    if not result or any(value <= 0 for value in result):
        raise ValueError("hidden_dims must contain at least one positive integer.")
    return result


def _encode_labels(
    y: pd.Series, labels: Sequence[str], *, allow_unknown: bool = False
) -> np.ndarray:
    mapping = {str(label): index for index, label in enumerate(labels)}
    if allow_unknown:
        return np.asarray([mapping.get(str(value), -1) for value in y], dtype=np.int64)
    return np.asarray([mapping[str(value)] for value in y], dtype=np.int64)


def _validation_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame],
    y_val: Optional[pd.Series],
) -> tuple[pd.DataFrame, pd.Series]:
    if X_val is None or y_val is None or len(X_val) == 0:
        return X_train, y_train
    return X_val, y_val


def build_fastai_network(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    dropout: float,
):
    """Build a numerical tabular MLP from native fastai layers."""
    import torch.nn as nn
    from fastai.layers import LinBnDrop

    layers: list[nn.Module] = []
    previous = int(input_dim)
    for index, width in enumerate(hidden_dims):
        layer_dropout = max(0.0, float(dropout) - 0.1 * index)
        layers.append(
            LinBnDrop(
                previous,
                int(width),
                bn=True,
                p=layer_dropout,
                act=nn.ReLU(),
            )
        )
        previous = int(width)
    layers.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*layers)


def build_torch_tabular_network(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    dropout: float,
):
    """Build a native PyTorch MLP with LayerNorm for numerical HAR features."""
    import torch.nn as nn

    layers: list[nn.Module] = []
    previous = int(input_dim)
    for index, width in enumerate(hidden_dims):
        layer_dropout = max(0.0, float(dropout) - 0.1 * index)
        layers.extend(
            [
                nn.Linear(previous, int(width)),
                nn.ReLU(),
                nn.LayerNorm(int(width)),
                nn.Dropout(layer_dropout),
            ]
        )
        previous = int(width)
    layers.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*layers)


class _NativeNeuralBackend(BaseAttributeBackend):
    """Shared preprocessing, probability output, and metadata behavior."""

    model_file = "model.pt"

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
        **config,
    ):
        dimensions = _hidden_dims(hidden_dims)
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if float(lr) <= 0.0 or float(weight_decay) < 0.0:
            raise ValueError("lr must be positive and weight_decay non-negative.")
        if min(int(batch_size), int(epochs), int(patience)) <= 0:
            raise ValueError("batch_size, epochs, and patience must be positive.")
        super().__init__(
            label=label,
            hidden_dims=dimensions,
            dropout=float(dropout),
            lr=float(lr),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            epochs=int(epochs),
            patience=int(patience),
            device=str(device),
            random_state=int(random_state),
            **config,
        )
        self.model = None
        self.labels: list[str] = []
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.best_epoch = 0
        self.best_val_loss = float("inf")

    def _build_model(self, input_dim: int, output_dim: int):
        raise NotImplementedError

    def _logits(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        if self.model is None or self.mean is None or self.std is None:
            raise RuntimeError(f"{self.backend_name} model is not fitted.")
        device = resolve_device(self.config["device"])
        self.model.to(device).eval()
        values = torch.from_numpy(normalize_features(X, self.mean, self.std))
        chunks = []
        with torch.no_grad():
            for start in range(0, len(values), self.config["batch_size"]):
                chunks.append(
                    self.model(values[start : start + self.config["batch_size"]].to(device)).cpu()
                )
        self.model.to("cpu")
        return (
            torch.cat(chunks, dim=0).numpy()
            if chunks
            else np.empty((0, len(self.labels)), dtype=np.float32)
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        indices = np.argmax(self._logits(X), axis=1)
        return pd.Series(
            [self.labels[index] for index in indices], index=X.index, name=self.label
        )

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        logits = self._logits(X)
        if len(logits) == 0:
            return pd.DataFrame(index=X.index, columns=self.labels, dtype=float)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        probabilities = exponentiated / np.sum(exponentiated, axis=1, keepdims=True)
        return pd.DataFrame(probabilities, index=X.index, columns=self.labels)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return self._logits(X)

    def save(self, path: str) -> None:
        import torch

        if self.model is None or self.mean is None or self.std is None:
            raise RuntimeError(f"Cannot save an unfitted {self.backend_name} model.")
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu() for key, value in self.model.state_dict().items()
                },
                "input_dim": len(self.feature_names),
                "output_dim": len(self.labels),
                "mean": torch.from_numpy(self.mean),
                "std": torch.from_numpy(self.std),
            },
            directory / self.model_file,
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

        directory = Path(path)
        payload = load_json(directory / BACKEND_METADATA)
        model = cls(label=payload.get("label"), **payload.get("config", {}))
        state = torch.load(directory / cls.model_file, map_location="cpu", weights_only=True)
        model.labels = [str(label) for label in payload.get("labels", [])]
        model.feature_names = list(payload.get("feature_names", []))
        model.mean = state["mean"].numpy()
        model.std = state["std"].numpy()
        model.model = model._build_model(
            input_dim=int(state["input_dim"]), output_dim=int(state["output_dim"])
        )
        model.model.load_state_dict(state["state_dict"])
        model.model.to("cpu")
        model.best_epoch = int(payload.get("best_epoch", 0))
        model.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        return model

    def get_metadata(self):
        result = super().get_metadata()
        result.update(
            {
                "library": self.library_name,
                "labels": self.labels,
                "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss,
            }
        )
        return result


class NeuralNetFastAIBackend(_NativeNeuralBackend):
    """Independent fastai learner for one attribute."""

    backend_name = "neuralnet_fastai"
    library_name = "fastai"
    model_file = FASTAI_MODEL_FILE

    def _build_model(self, input_dim: int, output_dim: int):
        return build_fastai_network(
            input_dim=input_dim,
            hidden_dims=self.config["hidden_dims"],
            output_dim=output_dim,
            dropout=self.config["dropout"],
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        try:
            import torch
            from fastai.callback.schedule import fit_one_cycle  # noqa: F401
            from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
            from fastai.data.core import DataLoaders
            from fastai.learner import Learner
            from fastai.losses import CrossEntropyLossFlat
            from fastai.metrics import accuracy
        except ImportError as exc:
            raise ImportError(
                "neuralnet_fastai backend requires fastai and torch in the active environment."
            ) from exc
        from torch.utils.data import DataLoader, TensorDataset

        seed = int(self.config["random_state"])
        set_torch_seed(seed)
        device = resolve_device(self.config["device"])
        self.feature_names = list(X_train.columns)
        self.labels = ordered_labels(y_train.astype(str).tolist())
        y_encoded = _encode_labels(y_train, self.labels)
        validation_X, validation_y = _validation_data(X_train, y_train, X_val, y_val)
        validation_encoded = _encode_labels(
            validation_y, self.labels, allow_unknown=True
        )
        self.mean, self.std = normalization_stats(X_train)
        train_values = normalize_features(X_train, self.mean, self.std)
        validation_values = normalize_features(validation_X, self.mean, self.std)
        known_validation = validation_encoded >= 0
        if known_validation.any():
            validation_values = validation_values[known_validation]
            validation_encoded = validation_encoded[known_validation]
        else:
            validation_values = train_values
            validation_encoded = y_encoded

        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_values), torch.from_numpy(y_encoded)),
            batch_size=self.config["batch_size"],
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(validation_values),
                torch.from_numpy(validation_encoded),
            ),
            batch_size=self.config["batch_size"],
            shuffle=False,
        )
        data = DataLoaders(train_loader, validation_loader, device=device)
        self.model = self._build_model(train_values.shape[1], len(self.labels)).to(device)
        weights = torch.tensor(
            balanced_class_weights(y_encoded, len(self.labels)),
            dtype=torch.float32,
            device=device,
        )

        with tempfile.TemporaryDirectory(prefix="zshar_fastai_") as temporary_dir:
            learner = Learner(
                data,
                self.model,
                loss_func=CrossEntropyLossFlat(weight=weights),
                metrics=accuracy,
                path=Path(temporary_dir),
                model_dir="models",
            )
            callbacks = [
                EarlyStoppingCallback(
                    monitor="valid_loss", min_delta=1e-7, patience=self.config["patience"]
                ),
                SaveModelCallback(monitor="valid_loss", fname="best", with_opt=False),
            ]
            learner.fit_one_cycle(
                self.config["epochs"],
                lr_max=self.config["lr"],
                wd=self.config["weight_decay"],
                cbs=callbacks,
            )
            learner.load("best", with_opt=False)
            validation_losses = [float(row[0]) for row in learner.recorder.values if row]
            if validation_losses:
                self.best_epoch = int(np.argmin(validation_losses) + 1)
                self.best_val_loss = float(np.min(validation_losses))
            self.model = learner.model

        self.model.to("cpu")
        return self


class NeuralNetTorchBackend(_NativeNeuralBackend):
    """Independent native PyTorch numerical-tabular network for one attribute."""

    backend_name = "neuralnet_torch"
    library_name = "torch"
    model_file = TORCH_MODEL_FILE

    def _build_model(self, input_dim: int, output_dim: int):
        return build_torch_tabular_network(
            input_dim=input_dim,
            hidden_dims=self.config["hidden_dims"],
            output_dim=output_dim,
            dropout=self.config["dropout"],
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        seed = int(self.config["random_state"])
        set_torch_seed(seed)
        device = resolve_device(self.config["device"])
        self.feature_names = list(X_train.columns)
        self.labels = ordered_labels(y_train.astype(str).tolist())
        y_encoded = _encode_labels(y_train, self.labels)
        validation_X, validation_y = _validation_data(X_train, y_train, X_val, y_val)
        validation_encoded = _encode_labels(
            validation_y, self.labels, allow_unknown=True
        )
        self.mean, self.std = normalization_stats(X_train)
        train_values = normalize_features(X_train, self.mean, self.std)
        validation_values = normalize_features(validation_X, self.mean, self.std)
        known_validation = validation_encoded >= 0
        if known_validation.any():
            validation_values = validation_values[known_validation]
            validation_encoded = validation_encoded[known_validation]
        else:
            validation_values = train_values
            validation_encoded = y_encoded

        self.model = self._build_model(train_values.shape[1], len(self.labels)).to(device)
        class_weights = torch.tensor(
            balanced_class_weights(y_encoded, len(self.labels)),
            dtype=torch.float32,
            device=device,
        )
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(1, self.config["patience"] // 3)
        )
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_values), torch.from_numpy(y_encoded)),
            batch_size=self.config["batch_size"],
            shuffle=True,
            generator=generator,
        )
        validation_features = torch.from_numpy(validation_values).to(device)
        validation_labels = torch.from_numpy(validation_encoded).to(device)

        best_state = None
        stale_epochs = 0
        for epoch in range(1, self.config["epochs"] + 1):
            self.model.train()
            for batch_features, batch_labels in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_features.to(device))
                loss = criterion(logits, batch_labels.to(device))
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                validation_loss = float(
                    criterion(self.model(validation_features), validation_labels).item()
                )
            scheduler.step(validation_loss)
            if validation_loss < self.best_val_loss - 1e-7:
                self.best_epoch = epoch
                self.best_val_loss = validation_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config["patience"]:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to("cpu")
        return self
