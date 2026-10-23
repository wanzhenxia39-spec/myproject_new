"""Native ordinal attribute predictors for zero-shot HAR.

``OrdinalLogisticBackend`` implements a proportional-odds cumulative-logit
model. ``CORALMLPBackend`` replaces its linear score with a nonlinear MLP and
optimizes the CORAL cumulative binary loss. Both models enforce increasing
cutpoints, so their cumulative probabilities are rank-consistent by design.
Neither implementation uses AutoGluon.
"""

from __future__ import annotations

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


ORDINAL_LOGISTIC_MODEL_FILE = "ordinal_logistic.pt"
CORAL_MLP_MODEL_FILE = "coral_mlp.pt"


def _validate_hidden_dims(values: Sequence[int]) -> list[int]:
    result = [int(value) for value in values]
    if not result or any(value <= 0 for value in result):
        raise ValueError("hidden_dims must contain at least one positive integer.")
    return result


def _encode_labels(
    values: pd.Series, labels: Sequence[str], *, allow_unknown: bool = False
) -> np.ndarray:
    mapping = {str(label): index for index, label in enumerate(labels)}
    if allow_unknown:
        return np.asarray(
            [mapping.get(str(value), -1) for value in values], dtype=np.int64
        )
    return np.asarray([mapping[str(value)] for value in values], dtype=np.int64)


def _class_probabilities_from_logits(logits):
    """Convert P(y > k) logits into K mutually exclusive class probabilities."""
    import torch

    cumulative = torch.sigmoid(logits)
    probabilities = torch.cat(
        [
            1.0 - cumulative[:, :1],
            cumulative[:, :-1] - cumulative[:, 1:],
            cumulative[:, -1:],
        ],
        dim=1,
    )
    probabilities = probabilities.clamp_min(1e-8)
    return probabilities / probabilities.sum(dim=1, keepdim=True)


def build_monotonic_ordinal_network(
    input_dim: int,
    output_dim: int,
    *,
    hidden_dims: Sequence[int],
    dropout: float,
):
    """Build a shared-score ordinal network with strictly ordered cutpoints."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    if int(output_dim) < 2:
        raise ValueError("Ordinal networks require at least two output classes.")
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

    class MonotonicOrdinalNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
            self.score = nn.Linear(previous, 1, bias=False)
            self.first_cutpoint = nn.Parameter(torch.tensor(-1.0))
            if output_dim > 2:
                initial_gap = torch.log(torch.expm1(torch.tensor(1.0)))
                self.raw_cutpoint_gaps = nn.Parameter(
                    initial_gap.repeat(int(output_dim) - 2)
                )
            else:
                self.register_parameter("raw_cutpoint_gaps", None)

        def cutpoints(self):
            first = self.first_cutpoint.reshape(1)
            if self.raw_cutpoint_gaps is None:
                return first
            positive_gaps = functional.softplus(self.raw_cutpoint_gaps) + 1e-4
            return torch.cat([first, first + torch.cumsum(positive_gaps, dim=0)])

        def forward(self, features):
            score = self.score(self.backbone(features))
            # P(y > k) = sigmoid(score - cutpoint_k). Increasing cutpoints
            # therefore guarantee non-increasing cumulative probabilities.
            return score - self.cutpoints().reshape(1, -1)

    return MonotonicOrdinalNetwork()


class _OrdinalTorchBackend(BaseAttributeBackend):
    """Shared native-PyTorch training and serialization for ordinal models."""

    model_file = "ordinal_model.pt"
    library_name = "torch"
    objective_name = "ordinal"

    def __init__(
        self,
        label=None,
        hidden_dims=(),
        dropout=0.0,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        epochs=200,
        patience=20,
        device="cuda:0",
        random_state=42,
        balance_classes=True,
        **config,
    ):
        dimensions = [int(value) for value in hidden_dims]
        if any(value <= 0 for value in dimensions):
            raise ValueError("hidden_dims values must be positive.")
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
            balance_classes=bool(balance_classes),
            **config,
        )
        self.model = None
        self.labels: list[str] = []
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.best_epoch = 0
        self.best_val_loss = float("inf")

    def _build_model(self, input_dim: int, output_dim: int):
        return build_monotonic_ordinal_network(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=self.config["hidden_dims"],
            dropout=self.config["dropout"],
        )

    def _prepare_loss(self, y_encoded: np.ndarray, num_classes: int, device):
        raise NotImplementedError

    def _loss(self, logits, targets, loss_context):
        raise NotImplementedError

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        seed = int(self.config["random_state"])
        set_torch_seed(seed)
        device = resolve_device(self.config["device"])
        self.feature_names = list(X_train.columns)
        self.labels = ordered_labels(y_train.astype(str).tolist())
        y_encoded = _encode_labels(y_train, self.labels)
        if X_val is None or y_val is None or len(X_val) == 0:
            validation_X, validation_encoded = X_train, y_encoded
        else:
            validation_X = X_val
            validation_encoded = _encode_labels(
                y_val, self.labels, allow_unknown=True
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
        loss_context = self._prepare_loss(y_encoded, len(self.labels), device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, self.config["patience"] // 3),
        )
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_values), torch.from_numpy(y_encoded)),
            batch_size=self.config["batch_size"],
            shuffle=True,
            generator=generator,
        )
        validation_features = torch.from_numpy(validation_values).to(device)
        validation_targets = torch.from_numpy(validation_encoded).to(device)

        best_state = None
        stale_epochs = 0
        for epoch in range(1, self.config["epochs"] + 1):
            self.model.train()
            for batch_features, batch_targets in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_features.to(device))
                loss = self._loss(logits, batch_targets.to(device), loss_context)
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                validation_loss = float(
                    self._loss(
                        self.model(validation_features),
                        validation_targets,
                        loss_context,
                    ).item()
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
            else np.empty((0, len(self.labels) - 1), dtype=np.float32)
        )

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        import torch

        logits = self._logits(X)
        if len(logits) == 0:
            return pd.DataFrame(index=X.index, columns=self.labels, dtype=float)
        with torch.no_grad():
            probabilities = _class_probabilities_from_logits(
                torch.from_numpy(logits)
            ).numpy()
        return pd.DataFrame(probabilities, index=X.index, columns=self.labels)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probabilities = self.predict_proba(X)
        indices = np.argmax(probabilities.to_numpy(), axis=1)
        return pd.Series(
            [self.labels[index] for index in indices], index=X.index, name=self.label
        )

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
                "objective": self.objective_name,
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
        cutpoints = []
        if self.model is not None:
            try:
                cutpoints = [
                    float(value)
                    for value in self.model.cutpoints().detach().cpu().tolist()
                ]
            except (AttributeError, RuntimeError):
                cutpoints = []
        result.update(
            {
                "library": self.library_name,
                "objective": self.objective_name,
                "labels": self.labels,
                "cutpoints": cutpoints,
                "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss,
            }
        )
        return result


class OrdinalLogisticBackend(_OrdinalTorchBackend):
    """Linear proportional-odds cumulative-logit attribute predictor."""

    backend_name = "ordinal_logistic"
    objective_name = "proportional_odds_negative_log_likelihood"
    model_file = ORDINAL_LOGISTIC_MODEL_FILE

    def __init__(self, label=None, **config):
        # An empty backbone makes the shared ordinal score strictly linear.
        config.pop("hidden_dims", None)
        config.pop("dropout", None)
        super().__init__(label=label, hidden_dims=(), dropout=0.0, **config)

    def _prepare_loss(self, y_encoded: np.ndarray, num_classes: int, device):
        import torch

        if not self.config.get("balance_classes", True):
            return None
        return torch.tensor(
            balanced_class_weights(y_encoded, num_classes),
            dtype=torch.float32,
            device=device,
        )

    def _loss(self, logits, targets, loss_context):
        import torch

        probabilities = _class_probabilities_from_logits(logits)
        row_indices = torch.arange(len(targets), device=targets.device)
        negative_log_likelihood = -torch.log(
            probabilities[row_indices, targets].clamp_min(1e-8)
        )
        if loss_context is not None:
            negative_log_likelihood = negative_log_likelihood * loss_context[targets]
        return negative_log_likelihood.mean()


class CORALMLPBackend(_OrdinalTorchBackend):
    """Nonlinear CORAL MLP with a shared score and monotonic cutpoints."""

    backend_name = "coral_mlp"
    objective_name = "coral_cumulative_binary_cross_entropy"
    model_file = CORAL_MLP_MODEL_FILE

    def __init__(self, label=None, hidden_dims=(256, 128), dropout=0.3, **config):
        super().__init__(
            label=label,
            hidden_dims=_validate_hidden_dims(hidden_dims),
            dropout=dropout,
            **config,
        )

    def _prepare_loss(self, y_encoded: np.ndarray, num_classes: int, device):
        import torch

        thresholds = np.arange(num_classes - 1, dtype=np.int64)
        cumulative_targets = y_encoded[:, None] > thresholds[None, :]
        if not self.config.get("balance_classes", True):
            return None
        positives = cumulative_targets.sum(axis=0).astype(np.float32)
        negatives = len(y_encoded) - positives
        positive_weights = negatives / np.maximum(positives, 1.0)
        return torch.tensor(positive_weights, dtype=torch.float32, device=device)

    def _loss(self, logits, targets, loss_context):
        import torch
        import torch.nn.functional as functional

        thresholds = torch.arange(logits.shape[1], device=targets.device)
        cumulative_targets = (targets[:, None] > thresholds[None, :]).float()
        return functional.binary_cross_entropy_with_logits(
            logits,
            cumulative_targets,
            pos_weight=loss_context,
        )
