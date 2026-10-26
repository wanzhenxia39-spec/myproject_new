#!/usr/bin/env python3
"""Standalone activity-prototype distance calculation and paper heatmap export.

Low/Medium/High has fixed 0/1/2 encoding: d(i,j)=sum_k |p_ik-p_jk|/(2K).
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
    values = table.values
    if (values.shape != (len(table.activities), len(table.attributes)) or
            values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1 or
            not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()):
        raise ValueError("Expected a finite normalized C-by-K matrix, C>=2, K>=1")
    distances = np.abs(values[:, None, :] - values[None, :, :]).mean(axis=2)
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
    matrix = pairwise_ordinal_distances(table)
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
                  "normalized_ordinal_l1": pair_values}).to_csv(output / "distance_pairs.csv", index=False)
    plot_blue_heatmap(matrix, output / "prototype_distance_heatmap",
                      colorbar_label="Normalized ordinal L1 distance",
                      title=args.dataset.upper() if args.titles else None,
                      formats=tuple(args.formats), dpi=args.dpi)
    import matplotlib
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": args.dataset,
        "input_path": str(source), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_dir": str(output), "class_scope": "all input activities, no split",
        "class_count": len(table.activities), "attribute_count": len(table.attributes),
        "total_pairs": len(pair_values), "activities": table.activities, "attributes": table.attributes,
        "formula": "mean_k(abs(raw_ik-raw_jk)/(declared_max-declared_min))",
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
    print(f"Mean pairwise distance: {pair_values.mean():.8f}; fixed Blues scale: [0,1]")
    print(f"Outputs: {output}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mhealth", help="Dataset identifier (default: mhealth)")
    parser.add_argument("--vectors", type=Path, help="Custom JSON/CSV; otherwise load dataset A1 v1.5")
    parser.add_argument("--input-root", type=Path, default=Path("/data/wanzhen/projects/myproject_new"))
    parser.add_argument("--output-dir", type=Path, help="System-disk result folder; default: results/distance_heatmaps/<dataset>")
    parser.add_argument("--level-map-json", help='Explicit custom mapping, e.g. {"Low":0,"High":1}')
    parser.add_argument("--value-range", type=float, nargs=2, default=[0, 2])
    parser.add_argument("--formats", nargs="+", choices=("pdf", "svg", "png"), default=["pdf", "svg", "png"])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--titles", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
