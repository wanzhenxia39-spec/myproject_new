#!/usr/bin/env python3
"""Standalone activity-prototype distance calculation and paper heatmap export.

Default: ordinal_probability matching evaluated on point-mass prototypes.
For uniform weights this equals ordinal L1: d(i,j)=sum_k |p_ik-p_jk|/(2K).
Only numpy, pandas and matplotlib are needed; no sibling code is required.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
import re
import sys
sys.dont_write_bytecode = True

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd



DEFAULT_LEVEL_MAP = {"Low": 0.0, "Medium": 1.0, "High": 2.0}
DATASETS = ("pamap2", "mhealth", "uschad", "wisdm", "dsads")
MATCHING_EPS = 1e-10
METRIC_LABELS = {
    "ordinal_probability": "Ordinal matching distance (point-mass prototypes)",
    "ordinal_l1": "Normalized ordinal L1 distance",
    "manhattan": "Normalized Manhattan distance",
    "euclidean": "Normalized Euclidean distance (RMS)",
    "cosine": "Cosine distance",
    "hamming": "Attribute disagreement rate",
}
METRIC_FORMULAS = {
    "ordinal_probability": "sum_k(w_k * sum_l(delta(p_ik,l)*abs(z_l-z_jk))) / sum_k(w_k)",
    "ordinal_l1": "sum_k(w_k*abs(z_ik-z_jk)) / sum_k(w_k)",
    "manhattan": "sum_k(w_k*abs(z_ik-z_jk)) / sum_k(w_k)",
    "euclidean": "sqrt(sum_k(w_k*(z_ik-z_jk)**2) / sum_k(w_k))",
    "cosine": "1 - sum_k(w_k*z_ik*z_jk)/sqrt(sum_k(w_k*z_ik**2)*sum_k(w_k*z_jk**2))",
    "hamming": "sum_k(w_k*I[z_ik!=z_jk]) / sum_k(w_k)",
}


@dataclass(frozen=True)
class PrototypeTable:
    activities: tuple[str, ...]
    attributes: tuple[str, ...]
    # Values are normalized using the declared scale, never the observed range.
    values: np.ndarray


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_prototypes(
    path: str | Path, *, level_map: Mapping[str, float] | None = None,
    value_range: tuple[float, float] = (0.0, 2.0),
) -> PrototypeTable:
    """Read activity -> attribute -> level JSON, or CSV with an activity column.

    Numeric inputs use value_range too. For a binary method explicitly pass
    value_range=(0, 1) and level_map={"Low": 0, "High": 1}, if labels are used.
    No lexical sorting, missing-value imputation, or observed-range rescaling.
    """
    source = Path(path)
    if source.suffix.lower() == ".json":
        vectors = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    elif source.suffix.lower() == ".csv":
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if not fields or fields[0] != "activity" or len(set(fields)) != len(fields):
                raise ValueError("CSV needs unique headers with 'activity' as its first column")
            vectors = {}
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("CSV row length differs from the header")
                activity = row.pop("activity")
                if activity in vectors:
                    raise ValueError(f"Duplicate activity: {activity}")
                vectors[activity] = row
    else:
        raise ValueError("Prototype input must be .json or .csv")
    return encode_prototypes(vectors, level_map=level_map, value_range=value_range)


def encode_prototypes(
    vectors: Mapping, *, level_map: Mapping[str, float] | None = None,
    value_range: tuple[float, float] = (0.0, 2.0),
) -> PrototypeTable:
    """Validate a complete rectangular prototype matrix and encode its values."""
    if not isinstance(vectors, Mapping) or len(vectors) < 2:
        raise ValueError("At least two activity prototypes are required")
    activities = tuple(vectors)
    if any(not isinstance(name, str) or not name.strip() for name in activities):
        raise ValueError("Activity names must be nonempty strings")
    first = vectors[activities[0]]
    if not isinstance(first, Mapping) or not first:
        raise ValueError("Each activity needs a nonempty attribute dictionary")
    attributes = tuple(first)
    if any(not isinstance(name, str) or not name.strip() for name in attributes):
        raise ValueError("Attribute names must be nonempty strings")
    if any(not isinstance(row, Mapping) or set(row) != set(attributes) for row in vectors.values()):
        raise ValueError("All activities must have exactly the same attributes")
    lower, upper = map(float, value_range)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("value_range must have finite increasing endpoints")
    mapping = dict(DEFAULT_LEVEL_MAP if level_map is None else level_map)
    mapped_values = np.asarray(list(mapping.values()), dtype=float)
    if (not np.isfinite(mapped_values).all() or
            ((mapped_values < lower) | (mapped_values > upper)).any() or
            len(set(mapped_values)) != len(mapped_values)):
        raise ValueError("level_map needs distinct finite values within value_range")

    encoded = []
    for activity in activities:
        row = []
        for attribute in attributes:
            raw = vectors[activity][attribute]
            if raw is None or isinstance(raw, (bool, list, dict)):
                raise ValueError(f"Invalid level at {activity}/{attribute}: {raw!r}")
            try:
                value = float(mapping[raw] if isinstance(raw, str) and raw in mapping else raw)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Unknown level at {activity}/{attribute}: {raw!r}") from error
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"Level outside value_range at {activity}/{attribute}: {raw!r}")
            row.append((value - lower) / (upper - lower))
        encoded.append(row)
    return PrototypeTable(activities, attributes, np.asarray(encoded, dtype=float))



def pairwise_ordinal_distances(table: PrototypeTable) -> pd.DataFrame:
    """Normalized ordinal L1: sum(|raw_i-raw_j|)/(2K) for 0/1/2 levels."""
    # Preserve the original API/geometry for the combined coverage analysis.
    return pairwise_prototype_distances(table, metric="ordinal_l1")


def normalized_attribute_weights(table: PrototypeTable, weights: Mapping | None = None) -> np.ndarray:
    """Explicit, complete weights; zero validation accuracy uses the matcher's EPS floor."""
    if weights is None:
        raw = np.ones(len(table.attributes), dtype=float)
    else:
        if not isinstance(weights, Mapping) or set(weights) != set(table.attributes):
            raise ValueError("Weights must cover exactly the input attribute names")
        raw = np.asarray([weights[name] for name in table.attributes], dtype=float)
        if not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError("Attribute weights must be finite and nonnegative")
        raw = np.maximum(raw, MATCHING_EPS)
    return raw / raw.max() / np.sum(raw / raw.max())


def read_attribute_weights(path: Path | None, table: PrototypeTable) -> tuple[dict | None, dict]:
    """Read an attribute->weight map or the project's saved per_attribute accuracy report."""
    if path is None:
        return None, {"policy": "uniform", "source": None}
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if isinstance(payload, dict) and "per_attribute" in payload:
        per_attribute = payload["per_attribute"]
        if not isinstance(per_attribute, dict) or set(per_attribute) != set(table.attributes):
            raise ValueError("Validation report must cover exactly the input attribute names")
        weights = {name: per_attribute[name]["accuracy"] for name in table.attributes}
        if any(not np.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in weights.values()):
            raise ValueError("Validation accuracies must be in [0,1]")
        policy = "validation_accuracy"
    else:
        weights, policy = payload, "explicit"
    normalized_attribute_weights(table, weights)
    return weights, {"policy": policy, "source": str(source),
                     "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "raw_weights": weights}


def pairwise_prototype_distances(
    table: PrototypeTable, *, metric: str = "ordinal_probability",
    attribute_weights: Mapping | None = None,
) -> pd.DataFrame:
    """Matching-compatible distances between deterministic prototypes on [0,1].

    ordinal_probability is the expected absolute ordinal error when the source
    prototype's distribution is a point mass. It reduces EXACTLY to weighted L1;
    it does not simulate a trained classifier's uncertain probabilities.
    euclidean uses the matcher's weighted RMS normalization, not unscaled L2.
    Cosine is optional; two zero vectors use distance 0, one zero vector uses 1.
    """
    if metric not in METRIC_LABELS:
        raise ValueError(f"Unknown distance metric: {metric}")
    values = table.values
    if (values.shape != (len(table.activities), len(table.attributes)) or
            values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1 or
            not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()):
        raise ValueError("Expected a finite normalized C-by-K matrix, C>=2, K>=1")
    weights = normalized_attribute_weights(table, attribute_weights)
    error = np.abs(values[:, None, :] - values[None, :, :])
    if metric in ("ordinal_probability", "ordinal_l1", "manhattan"):
        distances = np.sum(error * weights, axis=2)
    elif metric == "euclidean":
        distances = np.sqrt(np.sum(error**2 * weights, axis=2))
    elif metric == "hamming":
        distances = np.sum((error != 0) * weights, axis=2)
    else:
        weighted = values * np.sqrt(weights)
        norms = np.linalg.norm(weighted, axis=1)
        denominator = norms[:, None] * norms[None, :]
        similarity = np.divide(weighted @ weighted.T, denominator,
                               out=np.zeros_like(denominator), where=denominator > 0)
        distances = 1 - similarity
        both_zero = (norms[:, None] == 0) & (norms[None, :] == 0)
        distances[both_zero] = 0
    distances = np.clip(distances, 0, 1)
    np.fill_diagonal(distances, 0)
    return pd.DataFrame(distances, index=table.activities, columns=table.activities)


def plot_blue_heatmap(
    frame: pd.DataFrame, stem: Path, *, colorbar_label: str,
    title: str | None = None, coverage: bool = False,
    formats: tuple[str, ...] = ("pdf", "svg", "png"), dpi: int = 600,
) -> None:
    """Paper figure: fixed [0,1] Blues scale, numeric labels, vector font output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import textwrap

    rows, columns = frame.shape
    width = max(6.8, columns * (1.18 if coverage else 0.58) + 3.5)
    height = max(5.0, rows * (0.48 if coverage else 0.58) + 2.4)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }):
        fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
        norm = Normalize(vmin=0.0, vmax=1.0)
        cmap = plt.get_cmap("Blues")
        # Vector cells in PDF/SVG, rather than embedding a raster image.
        mesh = ax.pcolormesh(np.arange(columns + 1) - 0.5, np.arange(rows + 1) - 0.5,
                             frame.to_numpy(), cmap=cmap, norm=norm, shading="flat",
                             rasterized=False, antialiased=False)
        ax.set_xlim(-0.5, columns - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_aspect("auto" if coverage else "equal")
        ax.set_xticks(np.arange(columns), [str(x) for x in frame.columns],
                      rotation=0 if coverage else 45, ha="center" if coverage else "right",
                      rotation_mode="anchor")
        labels = [str(x).replace("_", " ").capitalize() if coverage else
                  textwrap.fill(str(x), width=29, break_long_words=False) for x in frame.index]
        ax.set_yticks(np.arange(rows), labels)
        ax.tick_params(axis="both", which="both", length=0, pad=7)
        ax.set_xticks(np.arange(-0.5, columns, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(rows):
            for j in range(columns):
                value = float(frame.iloc[i, j])
                red, green, blue, _ = cmap(norm(value))
                lightness = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=9 if coverage else 8,
                        color="white" if lightness < 0.52 else "#102a43")
        colorbar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.035,
                               ticks=np.linspace(0, 1, 6))
        colorbar.set_label(colorbar_label, labelpad=10)
        colorbar.outline.set_visible(False)
        colorbar.solids.set_rasterized(False)
        if title:
            ax.set_title(title, pad=15)
        if not coverage:
            ax.set_xlabel("Activity", labelpad=10)
            ax.set_ylabel("Activity", labelpad=10)
        for extension in formats:
            fig.savefig(stem.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)


def result_output(path):
    """Keep this explicitly authorized analysis's outputs on the system disk."""
    output = Path(path).expanduser().resolve()
    root = Path('/home/wanzhen/attribute_prototype_analysis/results').resolve()
    if not output.is_relative_to(root):
        raise ValueError('Output must be under /home/wanzhen/attribute_prototype_analysis/results')
    parent = output
    while not parent.exists():
        parent = parent.parent
    if parent.stat().st_dev != Path('/').stat().st_dev:
        raise ValueError('Analysis outputs must reside on the system disk')
    output.mkdir(parents=True, exist_ok=True)
    return output


def run(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.dataset):
        raise ValueError("Dataset identifier must use letters, digits, underscore or hyphen")
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")
    source = (args.vectors if args.vectors else
              args.input_root / f"configs/attributes/activity_attribute_vectors_initial_{args.dataset}_v1_5.json")
    source = source.expanduser().resolve()
    level_map = json.loads(args.level_map_json) if args.level_map_json else DEFAULT_LEVEL_MAP
    table = load_prototypes(source, level_map=level_map, value_range=tuple(args.value_range))
    weights, weighting = read_attribute_weights(args.weights_json, table)
    matrix = pairwise_prototype_distances(table, metric=args.metric, attribute_weights=weights)
    output = result_output(args.output_dir or
                           Path("/home/wanzhen/attribute_prototype_analysis/results/distance_heatmaps") / args.dataset)
    cache = result_output(output / "_cache")
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(cache)
    os.environ["TMPDIR"] = str(cache)
    matrix.to_csv(output / "distance_matrix.csv", index_label="activity")
    pd.DataFrame(table.values, index=table.activities, columns=table.attributes).to_csv(
        output / "normalized_prototypes.csv", index_label="activity")
    left, right = np.triu_indices(len(table.activities), k=1)
    pair_values = matrix.to_numpy()[left, right]
    pd.DataFrame({"activity_i": [table.activities[i] for i in left],
                  "activity_j": [table.activities[j] for j in right],
                  "distance": pair_values, "metric": args.metric}).to_csv(output / "distance_pairs.csv", index=False)
    plot_blue_heatmap(matrix, output / "prototype_distance_heatmap",
                      colorbar_label=METRIC_LABELS[args.metric] +
                      ("\n(validation accuracy weighted)" if weighting["policy"] == "validation_accuracy" else
                       "\n(explicit attribute weights)" if weights is not None else "\n(uniform weights)"),
                      title=args.dataset.upper() if args.titles else None,
                      formats=tuple(args.formats), dpi=args.dpi)
    import matplotlib
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": args.dataset,
        "input_path": str(source), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_dir": str(output), "class_scope": "all input activities, no split",
        "class_count": len(table.activities), "attribute_count": len(table.attributes),
        "total_pairs": len(pair_values), "activities": table.activities, "attributes": table.attributes,
        "metric": args.metric, "formula": METRIC_FORMULAS[args.metric],
        "prototype_distribution": "point_mass; no sample predictions or empirical probabilities used",
        "weighting": weighting,
        "normalized_attribute_weights": dict(zip(table.attributes, normalized_attribute_weights(table, weights))),
        "ordinal_probability_equivalence": "For point-mass prototypes, ordinal_probability equals weighted ordinal L1",
        "matching_scope": "prototype-to-prototype geometry; not sample-to-prototype classifier scores",
        "cosine_zero_vector_convention": "both zero: 0; one zero: 1" if args.metric == "cosine" else None,
        "normalized_encoding": "z=(raw-declared_min)/(declared_max-declared_min), fixed across datasets",
        "level_map": level_map, "value_range": args.value_range,
        "colormap": "Blues", "color_limits": [0, 1], "formats": args.formats, "dpi": args.dpi,
        "mean_pairwise_distance": float(pair_values.mean()),
        "minimum_off_diagonal_distance": float(pair_values.min()),
        "maximum_off_diagonal_distance": float(pair_values.max()),
        "code_path": str(Path(__file__).resolve()),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
    }
    (output / "distance_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{args.dataset}: {len(table.activities)} activities, {len(table.attributes)} attributes, {len(pair_values)} pairs")
    print(f"Metric: {args.metric}; weights: {weighting['policy']}; mean distance: {pair_values.mean():.8f}; Blues: [0,1]")
    print(f"Outputs: {output}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mhealth", help="Dataset identifier (default: mhealth)")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        help="Batch mode; write one subdirectory per dataset")
    parser.add_argument("--metric", choices=tuple(METRIC_LABELS), default="ordinal_probability",
                        help="Default: matcher ordinal_probability specialized to point-mass prototypes")
    parser.add_argument("--weights-json", type=Path,
                        help="Explicit attribute weights or a saved per_attribute validation-accuracy report")
    parser.add_argument("--comparison-metrics", nargs="*", choices=tuple(METRIC_LABELS), default=[],
                        help="Batch-only: also export comparisons/<metric>/<dataset>")
    parser.add_argument("--vectors", type=Path, help="Custom JSON/CSV; otherwise load dataset A1 v1.5")
    parser.add_argument("--input-root", type=Path, default=Path("/data/wanzhen/projects/myproject_new"))
    parser.add_argument("--output-dir", type=Path, help="Single dataset folder, or the parent folder in --datasets mode")
    parser.add_argument("--level-map-json", help='Explicit custom mapping, e.g. {"Low":0,"High":1}')
    parser.add_argument("--value-range", type=float, nargs=2, default=[0, 2])
    parser.add_argument("--formats", nargs="+", choices=("pdf", "svg", "png"), default=["pdf", "svg", "png"])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--titles", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.datasets:
        if args.vectors or args.weights_json:
            parser.error("Use --dataset for a custom vector/weight file; batch inputs and weights must not be mixed")
        if len(set(args.datasets)) != len(args.datasets):
            parser.error("--datasets must not contain duplicates")
        output = result_output(args.output_dir or
                               Path("/home/wanzhen/attribute_prototype_analysis/results/distance_heatmaps"))
        metrics = list(dict.fromkeys([args.metric, *args.comparison_metrics]))
        summaries = []
        for metric in metrics:
            for dataset in args.datasets:
                kwargs = vars(args) | {"dataset": dataset, "metric": metric,
                    "output_dir": output / dataset if metric == args.metric else output / "comparisons" / metric / dataset}
                directory = run(argparse.Namespace(**kwargs))
                metadata = json.loads((directory / "distance_run_metadata.json").read_text())
                summaries.append({key: metadata[key] for key in (
                    "dataset", "metric", "class_count", "attribute_count", "total_pairs", "mean_pairwise_distance",
                    "minimum_off_diagonal_distance", "maximum_off_diagonal_distance", "output_dir")})
        pd.DataFrame(summaries).to_csv(output / "distance_summary.csv", index=False)
        (output / "batch_run_metadata.json").write_text(json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(), "datasets": args.datasets,
            "primary_metric": args.metric, "comparison_metrics": metrics[1:], "weighting": "uniform",
            "class_scope": "all activities, latest initial nine-attribute A1 v1.5, no split",
            "color_limits": [0, 1], "output_dir": str(output),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        if args.comparison_metrics:
            parser.error("--comparison-metrics requires --datasets")
        run(args)


if __name__ == "__main__":
    main()
