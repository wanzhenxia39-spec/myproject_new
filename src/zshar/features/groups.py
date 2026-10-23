"""Exact feature partition and stable, attribute-specific column selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from zshar.paths import PROJECT_ROOT

DEFAULT_GROUPS = PROJECT_ROOT / "configs/features/feature_groups.json"


def load_groups(path: str | Path = DEFAULT_GROUPS) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    templates = {}
    if set(config["groups"]) != {f"G{i}" for i in range(1, 15)}:
        raise ValueError("Expected exactly G1 through G14")
    for gid, group in config["groups"].items():
        for template in group["feature_templates"]:
            if not template.startswith("{prefix}_") or template in templates:
                raise ValueError(f"Invalid or duplicate feature template: {template}")
            templates[template] = gid
    return config


def expand_feature_groups(dataset: str, config=None, *, prefix_overrides=None) -> dict:
    """Expand templates using the dataset's actual prefixes and available modalities.

    prefix_overrides maps a configured prefix to a custom extractor prefix. It
    renames the expected schema only; it does not rename existing feature arrays.
    """
    config = config if config is not None else load_groups()
    sensors = config.get("dataset_sensors", {}).get(dataset)
    if not sensors:
        raise ValueError(f"No feature sensor configuration for dataset {dataset!r}")
    overrides = prefix_overrides or {}
    if set(overrides) - set(sensors):
        raise ValueError("Prefix overrides contain unknown sensor prefixes")
    prefixes = [overrides.get(prefix, prefix) for prefix in sensors]
    if any(not isinstance(p, str) or not p.strip() for p in prefixes) or len(set(prefixes)) != len(prefixes):
        raise ValueError("Resolved prefixes must be nonempty, unique strings")
    groups = {gid: [] for gid in config["groups"]}
    for prefix, sensor in sensors.items():
        modalities = set(sensor["modalities"])
        if not modalities or not modalities <= {"acc", "gyro"}:
            raise ValueError(f"Unsupported sensor modalities for {dataset}.{prefix}")
        for gid, group in config["groups"].items():
            for template in group["feature_templates"]:
                suffix = template.removeprefix("{prefix}_")
                # The extractor emits all physical features only for acc+gyro pairs.
                required = ({"acc"} if suffix.startswith("acc_") and suffix != "acc_gyro_ratio"
                            else {"gyro"} if suffix.startswith("gyro_") else {"acc", "gyro"})
                if required <= modalities:
                    groups[gid].append(template.format(prefix=overrides.get(prefix, prefix)))
    return {"dataset": dataset, "prefixes": prefixes, "group_version": config["version"],
            "group_to_features": groups,
            "feature_names": [name for names in groups.values() for name in names]}


def resolve_feature_groups(feature_names, config=None, *, dataset=None) -> dict:
    config = config if config is not None else load_groups()
    names = list(map(str, feature_names))
    if not names or len(names) != len(set(names)):
        raise ValueError("Feature names must be nonempty and unique")
    owners = {}
    prefixes = set()
    group_features = {gid: [] for gid in config["groups"]}
    templates = [(gid, t.removeprefix("{prefix}"))
                 for gid, group in config["groups"].items()
                 for t in group["feature_templates"]]
    for name in names:
        matches = [(gid, name[:-len(suffix)]) for gid, suffix in templates
                   if name.endswith(suffix) and len(name) > len(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Feature must belong to exactly one group: {name}, matches={matches}")
        gid, prefix = matches[0]
        owners[name] = gid
        prefixes.add(prefix)
        group_features[gid].append(name)
    if dataset is None:
        expected = {t.format(prefix=p) for p in prefixes
                    for group in config["groups"].values() for t in group["feature_templates"]}
    else:
        expected = set(expand_feature_groups(dataset, config)["feature_names"])
        unexpected = set(names) - expected
        if unexpected:
            raise ValueError(f"Unexpected features for {dataset}: {sorted(unexpected)}")
    # Missing modalities may leave partial groups on other datasets; report this explicitly.
    return {"feature_names": names, "feature_to_group": owners,
            "group_to_features": group_features, "prefixes": sorted(prefixes),
            "missing_features": sorted(expected - set(names)),
            "group_version": config["version"]}


def resolve_attribute_features(feature_names, *, ontology=None, include_auxiliary=True, dataset=None) -> dict:
    if ontology is None:
        raise ValueError(
            "Attribute-feature resolution requires an explicit bound attribute plan; "
            "the semantics-only A1 ontology contains no feature-group mapping"
        )
    partition = resolve_feature_groups(feature_names, dataset=dataset)
    groups = partition["group_to_features"]
    names = partition["feature_names"]
    resolved = {}
    for attr in ontology["attributes"]:
        selected_groups = attr["primary_groups"] + (
            attr["auxiliary_groups"] if include_auxiliary else [])
        for gid in selected_groups:
            if gid not in groups or not groups[gid]:
                raise ValueError(f"No features for {attr['name']} supporting group {gid}")
        indices = [i for i, name in enumerate(names)
                   if partition["feature_to_group"][name] in selected_groups]
        resolved[attr["name"]] = {
            "primary_groups": attr["primary_groups"],
            "auxiliary_groups": attr["auxiliary_groups"],
            "selected_groups": selected_groups,
            "feature_names": [names[i] for i in indices], "indices": indices,
            "dimension": len(indices),
        }
    binding_version = ontology.get("schema_version", ontology.get("version", "unspecified"))
    return {**partition, "attribute_schema_version": binding_version,
            "include_auxiliary": include_auxiliary, "attributes": resolved}


def main():
    import numpy as np
    from zshar.datasets import load_dataset_spec
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pamap2")
    parser.add_argument("--attribute-bindings", type=Path,
                        help="Explicit bound A2 plan containing primary/auxiliary groups")
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--templates-only", action="store_true",
                        help="Expand dataset prefixes without reading existing feature files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.templates_only:
        result = expand_feature_groups(args.dataset)
    else:
        spec = load_dataset_spec(args.dataset)
        names = np.load(spec.feature_names_path, allow_pickle=True)
        if args.attribute_bindings:
            bindings = json.loads(args.attribute_bindings.read_text(encoding="utf-8"))
            result = resolve_attribute_features(
                names, ontology=bindings, include_auxiliary=not args.primary_only,
                dataset=args.dataset,
            )
        else:
            result = resolve_feature_groups(names, dataset=args.dataset)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.dataset}: {len(result['feature_names'])} features, "
          f"{len(result.get('missing_features', []))} missing templates")
    for name, entry in result.get("attributes", {}).items():
        print(f"  {name}: {entry['dimension']} columns ({', '.join(entry['selected_groups'])})")


if __name__ == "__main__":
    main()
