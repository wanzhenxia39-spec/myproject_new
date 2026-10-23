from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from zshar.evaluation.metrics import harmonic_mean


EPS = 1e-12
GZSL_BIAS_CORRECTIONS = (
    "temperature_seen_penalty",
    "temperature_only",
    "seen_penalty_only",
    "none",
)


def _validate_bias_correction(bias_correction: str) -> None:
    if bias_correction not in GZSL_BIAS_CORRECTIONS:
        choices = ", ".join(GZSL_BIAS_CORRECTIONS)
        raise ValueError(f"bias_correction must be one of: {choices}")


def temperature_scaled_probabilities(
    scores: pd.DataFrame, temperature: float = 1.0
) -> pd.DataFrame:
    """Convert compatibility scores to stable temperature-scaled probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = scores.to_numpy(dtype=float) / float(temperature)
    values -= np.max(values, axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True) + EPS
    return pd.DataFrame(probabilities, index=scores.index, columns=scores.columns)


def calibrated_scores(
    scores: pd.DataFrame,
    seen_activities: Sequence[str],
    *,
    gamma: float,
    temperature: float = 1.0,
    bias_correction: str = "temperature_seen_penalty",
) -> pd.DataFrame:
    """Apply the selected GZSL seen/unseen bias correction.

    ``temperature_seen_penalty`` preserves the original behavior: softmax the
    compatibility scores after temperature scaling, then subtract ``gamma``
    from seen candidates. ``temperature_only`` applies only temperature-scaled
    softmax and requires ``gamma=0``. ``seen_penalty_only`` subtracts ``gamma``
    directly from raw compatibility scores and deliberately skips
    softmax/temperature. ``none`` returns raw scores and requires identity
    parameters (``temperature=1`` and ``gamma=0``).
    """
    _validate_bias_correction(bias_correction)
    if bias_correction in {"seen_penalty_only", "none"}:
        if not np.isclose(float(temperature), 1.0):
            raise ValueError(
                f"temperature must be 1.0 when bias_correction={bias_correction!r}"
            )
        result = scores.astype(float).copy()
    else:
        result = temperature_scaled_probabilities(scores, temperature)
    if bias_correction == "temperature_only" and not np.isclose(float(gamma), 0.0):
        raise ValueError("gamma must be 0.0 when bias_correction='temperature_only'")
    if bias_correction == "none" and not np.isclose(float(gamma), 0.0):
        raise ValueError("gamma must be 0.0 when bias_correction='none'")
    seen = [name for name in seen_activities if name in result]
    result.loc[:, seen] = result.loc[:, seen] - float(gamma)
    return result


def gzsl_accuracy(
    truth: pd.Series,
    prediction: pd.Series,
    unseen_activities: Sequence[str],
) -> dict[str, float]:
    unseen = truth.astype(str).isin([str(item) for item in unseen_activities]).to_numpy()
    correct = truth.astype(str).to_numpy() == prediction.astype(str).to_numpy()
    seen_accuracy = float(correct[~unseen].mean()) if (~unseen).any() else 0.0
    unseen_accuracy = float(correct[unseen].mean()) if unseen.any() else 0.0
    return {
        "seen_accuracy": seen_accuracy,
        "unseen_accuracy": unseen_accuracy,
        "harmonic_mean": harmonic_mean(seen_accuracy, unseen_accuracy),
        "overall_accuracy": float(correct.mean()) if len(correct) else 0.0,
    }


def _ausuc(rows: Sequence[Mapping[str, float]]) -> float:
    """Integrate the upper seen/unseen envelope, removing duplicate x values."""
    upper_by_seen: dict[float, float] = {}
    for row in rows:
        seen = float(row["seen_accuracy"])
        upper_by_seen[seen] = max(upper_by_seen.get(seen, 0.0), float(row["unseen_accuracy"]))
    xs = np.asarray(sorted(upper_by_seen), dtype=float)
    ys = np.asarray([upper_by_seen[value] for value in xs], dtype=float)
    return float(abs(np.trapz(ys, xs))) if len(xs) > 1 else 0.0


def calibration_curve(
    scores: pd.DataFrame,
    truth: pd.Series,
    seen_activities: Sequence[str],
    unseen_activities: Sequence[str],
    gamma_values: Iterable[float],
    *,
    temperature: float = 1.0,
    bias_correction: str = "temperature_seen_penalty",
) -> tuple[list[dict[str, float]], float]:
    rows = []
    for gamma in gamma_values:
        prediction = calibrated_scores(
            scores,
            seen_activities,
            gamma=float(gamma),
            temperature=temperature,
            bias_correction=bias_correction,
        ).idxmax(axis=1)
        rows.append(
            {
                "gamma": float(gamma),
                **gzsl_accuracy(truth, prediction, unseen_activities),
            }
        )
    return rows, _ausuc(rows)


def build_gamma_grid(
    minimum: float = -1.0,
    maximum: float = 1.0,
    coarse_step: float = 0.05,
    fine_step: float = 0.001,
) -> list[float]:
    if maximum <= minimum or coarse_step <= 0 or fine_step <= 0:
        raise ValueError("Invalid calibration gamma range or step.")
    values: list[float] = []
    if minimum < 0:
        values.extend(np.arange(minimum, min(0.0, maximum) + coarse_step / 2, coarse_step))
    if maximum >= 0:
        values.extend(np.arange(max(0.0, minimum), maximum + fine_step / 2, fine_step))
    return sorted({round(float(value), 12) for value in values})


def _aggregate_curves(curves: Sequence[Sequence[Mapping[str, float]]]) -> list[dict[str, float]]:
    if not curves:
        raise ValueError("At least one calibration fold is required.")
    if any(len(curve) != len(curves[0]) for curve in curves):
        raise ValueError("Calibration fold gamma grids are not aligned.")
    result: list[dict[str, float]] = []
    for index in range(len(curves[0])):
        points = [curve[index] for curve in curves]
        if len({round(float(point["gamma"]), 12) for point in points}) != 1:
            raise ValueError("Calibration fold gamma grids are not aligned.")
        row: dict[str, float] = {"gamma": float(points[0]["gamma"])}
        for metric in ("seen_accuracy", "unseen_accuracy", "harmonic_mean", "overall_accuracy"):
            values = np.asarray([float(point[metric]) for point in points], dtype=float)
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_se"] = row[f"{metric}_std"] / np.sqrt(len(values)) if len(values) > 1 else 0.0
        result.append(row)
    return result


def _evaluate_temperature(
    folds: Sequence[Mapping[str, Any]],
    temperature: float,
    gamma_values: Sequence[float],
    bias_correction: str = "temperature_seen_penalty",
) -> dict[str, Any]:
    curves: list[list[dict[str, float]]] = []
    fold_ausucs: list[float] = []
    for fold in folds:
        curve, ausuc = calibration_curve(
            fold["scores"],
            pd.Series(fold["truth"], index=fold["scores"].index),
            fold["pseudo_seen"],
            fold["pseudo_unseen"],
            gamma_values,
            temperature=temperature,
            bias_correction=bias_correction,
        )
        curves.append(curve)
        fold_ausucs.append(float(ausuc))
    return {
        "temperature": float(temperature),
        "mean_ausuc": float(np.mean(fold_ausucs)),
        "fold_ausucs": fold_ausucs,
        "aggregate_curve": _aggregate_curves(curves),
    }


def _temperature_nll(
    folds: Sequence[Mapping[str, Any]], temperature: float
) -> tuple[float, list[float]]:
    """Return mean validation NLL for temperature-only calibration."""
    fold_nlls: list[float] = []
    for fold in folds:
        probabilities = temperature_scaled_probabilities(fold["scores"], temperature)
        truth = pd.Series(fold["truth"], index=probabilities.index).astype(str)
        column_indices = probabilities.columns.astype(str).get_indexer(truth)
        if (column_indices < 0).any():
            missing = sorted(set(truth[column_indices < 0]))
            raise ValueError(f"Calibration truth labels are absent from scores: {missing}")
        rows = np.arange(len(probabilities))
        truth_probability = probabilities.to_numpy(dtype=float)[rows, column_indices]
        fold_nlls.append(float(-np.log(truth_probability + EPS).mean()))
    return float(np.mean(fold_nlls)), fold_nlls


def _choose_gamma(
    curve: Sequence[Mapping[str, float]],
    *,
    mean_ausuc: float,
    min_seen_retention: float,
    min_validation_ausuc: float,
    one_standard_error_rule: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    identity = min(curve, key=lambda row: abs(float(row["gamma"])))
    seen_floor = float(identity["seen_accuracy"]) * float(min_seen_retention)
    eligible = [row for row in curve if float(row["seen_accuracy"]) >= seen_floor] or [identity]
    best = max(
        eligible,
        key=lambda row: (
            float(row["harmonic_mean"]),
            float(row["overall_accuracy"]),
            -abs(float(row["gamma"])),
        ),
    )
    selected, reason = best, "max_mean_h_with_seen_retention"
    if mean_ausuc < min_validation_ausuc:
        selected, reason = identity, "fallback_identity_low_validation_ausuc"
    elif float(best["harmonic_mean"]) <= float(identity["harmonic_mean"]):
        selected, reason = identity, "fallback_identity_no_harmonic_gain"
    elif one_standard_error_rule:
        threshold = float(best["harmonic_mean"]) - float(best["harmonic_mean_se"])
        within_one_se = [row for row in eligible if float(row["harmonic_mean"]) >= threshold]
        selected = min(
            within_one_se,
            key=lambda row: (
                abs(float(row["gamma"])),
                -float(row["harmonic_mean"]),
                -float(row["overall_accuracy"]),
            ),
        )
        reason = "one_standard_error_smallest_absolute_gamma"
    return dict(selected), {
        "selection_reason": reason,
        "identity_point": dict(identity),
        "seen_retention_floor": float(min_seen_retention),
        "minimum_seen_accuracy": seen_floor,
        "min_validation_ausuc": float(min_validation_ausuc),
    }


def tune_calibration(
    folds: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    gamma_values: Sequence[float],
    temperature_grid: Sequence[float],
    tpe_trials: int,
    tau_min: float,
    tau_max: float,
    seed: int,
    min_seen_retention: float = 0.5,
    min_validation_ausuc: float = 0.0,
    one_standard_error_rule: bool = True,
    bias_correction: str = "temperature_seen_penalty",
) -> dict[str, Any]:
    """Select temperature by mean pseudo-GZSL AUSUC, then a conservative gamma."""
    if mode not in {"grid", "tpe"}:
        raise ValueError("Automatic calibration mode must be 'grid' or 'tpe'.")
    if not folds:
        raise ValueError("Automatic GZSL calibration requires at least one fold.")
    _validate_bias_correction(bias_correction)
    if bias_correction == "none":
        raise ValueError("bias_correction='none' does not use automatic calibration")

    trials: list[dict[str, Any]] = []
    if bias_correction == "temperature_only":
        if mode == "grid":
            temperatures = sorted({float(value) for value in temperature_grid})
            if not temperatures or any(value <= 0 for value in temperatures):
                raise ValueError("calibration temperature grid must contain positive values")
            for value in temperatures:
                payload = _evaluate_temperature(
                    folds,
                    value,
                    [0.0],
                    bias_correction=bias_correction,
                )
                payload["mean_nll"], payload["fold_nlls"] = _temperature_nll(folds, value)
                trials.append(payload)
        else:
            if tpe_trials <= 0 or tau_min <= 0 or tau_max <= tau_min:
                raise ValueError("Invalid TPE trial count or temperature range.")
            try:
                import optuna
            except ImportError as exc:
                raise RuntimeError(
                    "TPE calibration requires optuna; install zshar[calibration] or use --calibration-mode grid."
                ) from exc

            payloads: dict[int, dict[str, Any]] = {}

            def temperature_objective(trial: Any) -> float:
                value = float(trial.suggest_float("temperature", tau_min, tau_max, log=True))
                payload = _evaluate_temperature(
                    folds,
                    value,
                    [0.0],
                    bias_correction=bias_correction,
                )
                payload["mean_nll"], payload["fold_nlls"] = _temperature_nll(folds, value)
                payloads[int(trial.number)] = payload
                return float(payload["mean_nll"])

            study = optuna.create_study(
                direction="minimize", sampler=optuna.samplers.TPESampler(seed=int(seed))
            )
            if tau_min <= 1.0 <= tau_max:
                study.enqueue_trial({"temperature": 1.0})
            study.optimize(temperature_objective, n_trials=int(tpe_trials))
            trials = [
                payloads[int(trial.number)]
                for trial in study.trials
                if trial.value is not None
            ]

        best = min(
            trials,
            key=lambda row: (
                float(row["mean_nll"]),
                abs(np.log(float(row["temperature"]))),
            ),
        )
        selected_point = dict(best["aggregate_curve"][0])
        return {
            "mode": mode,
            "bias_correction": bias_correction,
            "temperature_applied": True,
            "temperature_search_skipped": False,
            "score_transform": "temperature_softmax",
            "selection_metric": "mean_validation_negative_log_likelihood",
            "best_temperature": float(best["temperature"]),
            "ausuc_best_temperature": None,
            "best_gamma": 0.0,
            "best_ausuc": None,
            "best_operating_point": selected_point,
            "gamma_diagnostics": {
                "selection_reason": "seen_penalty_removed_gamma_fixed_zero",
                "identity_point": selected_point,
            },
            "best_aggregate_curve": best["aggregate_curve"],
            "best_fold_ausucs": best["fold_ausucs"],
            "best_mean_nll": float(best["mean_nll"]),
            "best_fold_nlls": list(best["fold_nlls"]),
            "trials": [
                {
                    "temperature": float(row["temperature"]),
                    "mean_nll": float(row["mean_nll"]),
                }
                for row in trials
            ],
        }

    temperature_search_skipped = bias_correction == "seen_penalty_only"
    if temperature_search_skipped:
        trials = [
            _evaluate_temperature(
                folds,
                1.0,
                gamma_values,
                bias_correction=bias_correction,
            )
        ]
    elif mode == "grid":
        temperatures = sorted({float(value) for value in temperature_grid})
        if not temperatures or any(value <= 0 for value in temperatures):
            raise ValueError("calibration temperature grid must contain positive values")
        trials = [
            _evaluate_temperature(
                folds,
                value,
                gamma_values,
                bias_correction=bias_correction,
            )
            for value in temperatures
        ]
    else:
        if tpe_trials <= 0 or tau_min <= 0 or tau_max <= tau_min:
            raise ValueError("Invalid TPE trial count or temperature range.")
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError(
                "TPE calibration requires optuna; install zshar[calibration] or use --calibration-mode grid."
            ) from exc

        payloads: dict[int, dict[str, Any]] = {}

        def objective(trial: Any) -> float:
            temperature = float(trial.suggest_float("temperature", tau_min, tau_max, log=True))
            payload = _evaluate_temperature(
                folds,
                temperature,
                gamma_values,
                bias_correction=bias_correction,
            )
            payloads[int(trial.number)] = payload
            return float(payload["mean_ausuc"])

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=int(seed))
        )
        if tau_min <= 1.0 <= tau_max:
            study.enqueue_trial({"temperature": 1.0})
        study.optimize(objective, n_trials=int(tpe_trials))
        trials = [payloads[int(trial.number)] for trial in study.trials if trial.value is not None]

    best = max(trials, key=lambda row: (float(row["mean_ausuc"]), -abs(np.log(row["temperature"]))))
    selected_point, diagnostics = _choose_gamma(
        best["aggregate_curve"],
        mean_ausuc=float(best["mean_ausuc"]),
        min_seen_retention=min_seen_retention,
        min_validation_ausuc=min_validation_ausuc,
        one_standard_error_rule=one_standard_error_rule,
    )
    selected_gamma = float(selected_point["gamma"])
    selected_temperature = 1.0 if abs(selected_gamma) <= EPS else float(best["temperature"])
    return {
        "mode": mode,
        "bias_correction": bias_correction,
        "temperature_applied": not temperature_search_skipped,
        "temperature_search_skipped": temperature_search_skipped,
        "score_transform": "identity" if temperature_search_skipped else "temperature_softmax",
        "selection_metric": "mean_validation_ausuc_across_pseudo_unseen_class_folds",
        "best_temperature": selected_temperature,
        "ausuc_best_temperature": float(best["temperature"]),
        "best_gamma": selected_gamma,
        "best_ausuc": float(best["mean_ausuc"]),
        "best_operating_point": selected_point,
        "gamma_diagnostics": diagnostics,
        "best_aggregate_curve": best["aggregate_curve"],
        "best_fold_ausucs": best["fold_ausucs"],
        "trials": [
            {"temperature": float(row["temperature"]), "mean_ausuc": float(row["mean_ausuc"])}
            for row in trials
        ],
    }
