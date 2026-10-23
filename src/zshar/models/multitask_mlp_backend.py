"""Shared-backbone multi-task MLP for joint attribute prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .base import ORDINAL_VALUE_MAP, classification_metrics, ordered_labels
from .independent_mlp_backend import (
    balanced_class_weights,
    dropout_schedule,
    normalization_stats,
    normalize_features,
    resolve_device,
    set_torch_seed,
)
from .serialization import BACKEND_METADATA, MANAGER_MANIFEST, load_json, save_json


MODEL_FILE = "multitask_mlp.pt"


def build_multitask_network(
    input_dim: int,
    hidden_dims: Sequence[int],
    head_dims: Mapping[str, int],
    dropout: float,
    head_feature_indices: Optional[Mapping[str, Sequence[int]]] = None,
):
    import torch
    import torch.nn as nn

    layers: list[nn.Module] = []
    previous = int(input_dim)
    for width, drop in zip(hidden_dims, dropout_schedule(dropout, len(hidden_dims))):
        layers.extend([nn.Linear(previous, int(width)), nn.ReLU(), nn.Dropout(drop)])
        previous = int(width)

    class MultiTaskNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
            self.heads = nn.ModuleDict(
                {head_key: nn.Linear(previous, int(output_dim)) for head_key, output_dim in head_dims.items()}
            )
            self.mask_names: dict[str, str] = {}
            for head_key, indices in dict(head_feature_indices or {}).items():
                mask = torch.zeros(int(input_dim), dtype=torch.float32)
                mask[list(map(int, indices))] = 1.0
                buffer_name = f"feature_mask_{head_key}"
                self.register_buffer(buffer_name, mask)
                self.mask_names[head_key] = buffer_name

        def forward(self, features):
            if not self.mask_names:
                shared = self.backbone(features)
                return {head_key: head(shared) for head_key, head in self.heads.items()}
            return {
                head_key: head(
                    self.backbone(features * getattr(self, self.mask_names[head_key]))
                )
                for head_key, head in self.heads.items()
            }

    return MultiTaskNetwork()


class MultiTaskAttributeModelManager:
    manager_type = "multitask"
    backend_name = "mlp_multitask"

    def __init__(
        self,
        attribute_cols: Optional[Sequence[str]] = None,
        hidden_dims=(256, 128),
        dropout=0.3,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        epochs=200,
        patience=20,
        device="cuda:0",
        random_state=42,
        attribute_weights: Optional[Mapping[str, float]] = None,
        attribute_feature_columns: Optional[Mapping[str, Sequence[str]]] = None,
        **config,
    ):
        self.attribute_cols = list(attribute_cols) if attribute_cols is not None else []
        self.config = {
            "hidden_dims": [int(value) for value in hidden_dims],
            "dropout": float(dropout),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "patience": int(patience),
            "device": str(device),
            "random_state": int(random_state),
            "attribute_weights": dict(attribute_weights or {}),
            **config,
        }
        self.feature_names: list[str] = []
        self.attribute_feature_columns = {
            str(attribute): [str(column) for column in columns]
            for attribute, columns in dict(attribute_feature_columns or {}).items()
        }
        self.head_feature_indices: dict[str, list[int]] = {}
        self.labels: dict[str, list[str]] = {}
        self.constants: dict[str, str] = {}
        self.head_keys: dict[str, str] = {}
        self.model = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.best_epoch = 0
        self.best_val_loss = float("inf")
        self.validation_report: dict[str, Any] = {}
        self.problem_types: dict[str, str] = {}

    def fit_all(self, X_train, Y_train, X_val, Y_val, root_path=None, label_count_threshold=1):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        seed = int(self.config["random_state"])
        set_torch_seed(seed)
        device = resolve_device(self.config["device"])
        self.attribute_cols = list(Y_train.columns)
        self.feature_names = list(X_train.columns)
        extra = sorted(set(self.attribute_feature_columns).difference(self.attribute_cols))
        missing = sorted(set(self.attribute_cols).difference(self.attribute_feature_columns))
        if self.attribute_feature_columns and (extra or missing):
            raise ValueError(
                "attribute_feature_columns must cover the trained schema exactly: "
                f"missing={missing}, extra={extra}"
            )
        feature_positions = {name: index for index, name in enumerate(self.feature_names)}
        self.mean, self.std = normalization_stats(X_train)
        train_values = normalize_features(X_train, self.mean, self.std)
        val_values = normalize_features(X_val, self.mean, self.std)
        learned_attrs: list[str] = []
        encoded_train: list[np.ndarray] = []
        encoded_val: list[np.ndarray] = []
        class_weight_tensors: dict[str, Any] = {}
        head_dims: dict[str, int] = {}
        for attr_index, attr in enumerate(self.attribute_cols):
            counts = Y_train[attr].dropna().astype(str).value_counts()
            if counts.empty:
                raise ValueError(f"Attribute [{attr}] has no training labels.")
            if len(counts) > 1 and int(counts.min()) < int(label_count_threshold):
                raise ValueError(
                    f"Attribute [{attr}] has classes below label_count_threshold={label_count_threshold}: "
                    f"{counts.to_dict()}"
                )
            labels = ordered_labels(Y_train[attr].astype(str).tolist())
            self.labels[attr] = labels
            if len(labels) == 1:
                self.constants[attr] = labels[0]
                self.problem_types[attr] = "constant"
                continue
            self.problem_types[attr] = "binary" if len(labels) == 2 else "multiclass"
            head_key = f"head_{attr_index}"
            self.head_keys[attr] = head_key
            head_dims[head_key] = len(labels)
            if self.attribute_feature_columns:
                selected = self.attribute_feature_columns[attr]
                if not selected or len(selected) != len(set(selected)):
                    raise ValueError(
                        f"Feature subset for attribute {attr!r} must be nonempty and duplicate-free"
                    )
                absent = [name for name in selected if name not in feature_positions]
                if absent:
                    raise ValueError(
                        f"Input is missing {len(absent)} bound features for attribute {attr!r}: "
                        f"{absent[:10]}"
                    )
                self.head_feature_indices[head_key] = [
                    feature_positions[name] for name in selected
                ]
            learned_attrs.append(attr)
            mapping = {label: index for index, label in enumerate(labels)}
            train_encoded = np.asarray([mapping[str(value)] for value in Y_train[attr]], dtype=np.int64)
            val_encoded = np.asarray([mapping.get(str(value), -1) for value in Y_val[attr]], dtype=np.int64)
            encoded_train.append(train_encoded)
            encoded_val.append(val_encoded)
            class_weight_tensors[attr] = torch.tensor(
                balanced_class_weights(train_encoded, len(labels)), dtype=torch.float32, device=device
            )
        train_targets = np.column_stack(encoded_train).astype(np.int64) if encoded_train else np.empty((len(X_train), 0), dtype=np.int64)
        val_targets = np.column_stack(encoded_val).astype(np.int64) if encoded_val else np.empty((len(X_val), 0), dtype=np.int64)
        self.model = build_multitask_network(
            train_values.shape[1],
            self.config["hidden_dims"],
            head_dims,
            self.config["dropout"],
            self.head_feature_indices,
        ).to(device)
        if learned_attrs:
            criteria = {
                attr: torch.nn.CrossEntropyLoss(weight=class_weight_tensors[attr]) for attr in learned_attrs
            }
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"]
            )
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(
                TensorDataset(torch.from_numpy(train_values), torch.from_numpy(train_targets)),
                batch_size=self.config["batch_size"],
                shuffle=True,
                generator=generator,
            )
            val_x = torch.from_numpy(val_values).to(device)
            val_y = torch.from_numpy(val_targets).to(device)
            best_state = None
            stale_epochs = 0
            for epoch in range(1, self.config["epochs"] + 1):
                self.model.train()
                for batch_x, batch_y in loader:
                    optimizer.zero_grad(set_to_none=True)
                    outputs = self.model(batch_x.to(device))
                    loss = sum(
                        float(self.config["attribute_weights"].get(attr, 1.0))
                        * criteria[attr](outputs[self.head_keys[attr]], batch_y[:, index].to(device))
                        for index, attr in enumerate(learned_attrs)
                    )
                    loss.backward()
                    optimizer.step()
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(val_x)
                    loss_terms = []
                    for index, attr in enumerate(learned_attrs):
                        mask = val_y[:, index] >= 0
                        if bool(mask.any()):
                            loss_terms.append(
                                float(self.config["attribute_weights"].get(attr, 1.0))
                                * criteria[attr](outputs[self.head_keys[attr]][mask], val_y[:, index][mask])
                            )
                    val_loss = float(torch.stack(loss_terms).sum().item()) if loss_terms else 0.0
                if val_loss < self.best_val_loss - 1e-7:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= self.config["patience"]:
                        break
            if best_state is not None:
                self.model.load_state_dict(best_state)
        else:
            self.best_epoch = 0
            self.best_val_loss = 0.0
        self.model.to("cpu")
        predictions = self.predict_all(X_val)
        self.validation_report = self._validation_metrics(Y_val, predictions)
        if root_path:
            self.save(root_path)
        return self.validation_report

    def _forward(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        import torch

        device = resolve_device(self.config["device"])
        self.model.to(device).eval()
        missing = [name for name in self.feature_names if name not in X.columns]
        if missing:
            raise ValueError(f"Input is missing trained features: {missing[:10]}")
        values = torch.from_numpy(
            normalize_features(X.loc[:, self.feature_names], self.mean, self.std)
        )
        collected = {attr: [] for attr in self.head_keys}
        with torch.no_grad():
            for start in range(0, len(values), self.config["batch_size"]):
                outputs = self.model(values[start : start + self.config["batch_size"]].to(device))
                for attr, head_key in self.head_keys.items():
                    collected[attr].append(outputs[head_key].cpu())
        self.model.to("cpu")
        return {
            attr: torch.cat(chunks, dim=0).numpy() if chunks else np.empty((0, len(self.labels[attr])))
            for attr, chunks in collected.items()
        }

    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        logits = self._forward(X) if self.head_keys else {}
        result = {}
        for attr in self.attribute_cols:
            if attr in self.constants:
                result[attr] = pd.Series(self.constants[attr], index=X.index, dtype=object)
            else:
                indices = np.argmax(logits[attr], axis=1)
                result[attr] = pd.Series([self.labels[attr][index] for index in indices], index=X.index)
        return pd.DataFrame(result, index=X.index, columns=self.attribute_cols)

    def predict_proba_all(self, X: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Return softmax probabilities for every attribute head."""
        logits = self._forward(X) if self.head_keys else {}
        probabilities: dict[str, pd.DataFrame] = {}
        for attr in self.attribute_cols:
            if attr in self.constants:
                probabilities[attr] = pd.DataFrame(
                    {str(self.constants[attr]): np.ones(len(X), dtype=float)},
                    index=X.index,
                )
                continue
            values = logits[attr]
            shifted = values - np.max(values, axis=1, keepdims=True)
            exponentiated = np.exp(shifted)
            softmax = exponentiated / np.sum(exponentiated, axis=1, keepdims=True)
            probabilities[attr] = pd.DataFrame(
                softmax,
                columns=[str(label) for label in self.labels[attr]],
                index=X.index,
            )
        return probabilities

    def _validation_metrics(self, truth: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
        per_attribute = {
            attr: classification_metrics(truth[attr].astype(str), prediction[attr].astype(str))
            for attr in self.attribute_cols
        }
        exact_match = float((truth.astype(str) == prediction.astype(str)).all(axis=1).mean())
        true_numeric = truth.astype(str).replace(ORDINAL_VALUE_MAP).astype(float).to_numpy()
        pred_numeric = prediction.astype(str).replace(ORDINAL_VALUE_MAP).astype(float).to_numpy()
        cosine = np.sum(true_numeric * pred_numeric, axis=1) / (
            np.linalg.norm(true_numeric, axis=1) * np.linalg.norm(pred_numeric, axis=1) + 1e-10
        )
        return {
            "per_attribute": per_attribute,
            "exact_match": exact_match,
            "average_cosine_similarity": float(np.nanmean(cosine)),
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }

    def save(self, path: str) -> None:
        import torch

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        head_dims = {self.head_keys[attr]: len(self.labels[attr]) for attr in self.head_keys}
        torch.save(
            {
                "state_dict": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                "network_config": {
                    "input_dim": len(self.feature_names),
                    "hidden_dims": self.config["hidden_dims"],
                    "head_dims": head_dims,
                    "dropout": self.config["dropout"],
                    "head_feature_indices": self.head_feature_indices,
                },
                "label_mappings": {
                    attr: {label: index for index, label in enumerate(labels)}
                    for attr, labels in self.labels.items()
                },
                "mean": torch.from_numpy(self.mean),
                "std": torch.from_numpy(self.std),
            },
            directory / MODEL_FILE,
        )
        save_json(
            directory / BACKEND_METADATA,
            {
                "backend": self.backend_name,
                "manager_type": self.manager_type,
                "attribute_cols": self.attribute_cols,
                "feature_names": self.feature_names,
                "attribute_feature_columns": self.attribute_feature_columns,
                "labels": self.labels,
                "constants": self.constants,
                "head_keys": self.head_keys,
                "problem_types": self.problem_types,
                "config": self.config,
                "validation_report": self.validation_report,
                "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss,
            },
        )
        save_json(
            directory / MANAGER_MANIFEST,
            {
                "format_version": 2,
                "manager_type": self.manager_type,
                "classifier_backend": self.backend_name,
                "attribute_cols": self.attribute_cols,
                "problem_types": self.problem_types,
                "validation_report": self.validation_report,
                "attribute_feature_columns": self.attribute_feature_columns,
            },
        )

    @classmethod
    def load(cls, path: str, **kwargs: Any):
        import torch

        payload = load_json(Path(path) / BACKEND_METADATA)
        manager = cls(
            attribute_cols=payload["attribute_cols"],
            attribute_feature_columns=payload.get("attribute_feature_columns", {}),
            **payload.get("config", {}),
        )
        state = torch.load(Path(path) / MODEL_FILE, map_location="cpu", weights_only=True)
        network = state["network_config"]
        manager.model = build_multitask_network(
            network["input_dim"],
            network["hidden_dims"],
            network["head_dims"],
            network["dropout"],
            network.get("head_feature_indices", {}),
        )
        manager.model.load_state_dict(state["state_dict"])
        manager.feature_names = list(payload["feature_names"])
        manager.labels = {attr: list(labels) for attr, labels in payload["labels"].items()}
        manager.constants = dict(payload.get("constants", {}))
        manager.head_keys = dict(payload.get("head_keys", {}))
        manager.head_feature_indices = {
            str(key): list(map(int, value))
            for key, value in network.get("head_feature_indices", {}).items()
        }
        manager.problem_types = dict(payload.get("problem_types", {}))
        manager.validation_report = dict(payload.get("validation_report", {}))
        manager.best_epoch = int(payload.get("best_epoch", 0))
        manager.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        manager.mean = state["mean"].numpy()
        manager.std = state["std"].numpy()
        return manager

    def get_feature_importance(self, attr, X, y, num_shuffle_sets=3, random_state=42):
        if attr in self.constants:
            return pd.DataFrame(columns=["importance", "stddev"])
        rng = np.random.default_rng(random_state)
        baseline = classification_metrics(y.astype(str), self.predict_all(X)[attr].astype(str))["macro_f1"]
        importances = []
        for feature in X.columns:
            drops = []
            for _ in range(max(1, int(num_shuffle_sets))):
                permuted = X.copy()
                permuted[feature] = rng.permutation(permuted[feature].to_numpy())
                score = classification_metrics(y.astype(str), self.predict_all(permuted)[attr].astype(str))["macro_f1"]
                drops.append(baseline - score)
            importances.append((float(np.mean(drops)), float(np.std(drops))))
        return pd.DataFrame(importances, index=X.columns, columns=["importance", "stddev"]).sort_values(
            "importance", ascending=False
        )

    def get_metadata(self):
        return {
            "backend": self.backend_name,
            "manager_type": self.manager_type,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "constant_attributes": self.constants,
            "learned_head_count": len(self.head_keys),
            "attribute_feature_columns": self.attribute_feature_columns,
        }
