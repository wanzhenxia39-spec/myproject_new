"""Structured feature tags and A2 feature-side selection (no model training)."""
from __future__ import annotations

import re

from zshar.features.groups import load_groups, resolve_feature_groups


def feature_catalog(feature_names, *, dataset, config=None):
    config = config or load_groups()
    partition = resolve_feature_groups(feature_names, config, dataset=dataset)
    sensors = config["dataset_sensors"][dataset]
    catalog = {}
    for name in partition["feature_names"]:
        prefix = next(p for p in sorted(sensors, key=len, reverse=True) if name.startswith(p + "_"))
        suffix = name[len(prefix) + 1:]
        match = re.fullmatch(r"(acc|gyro)_([xyz]|mag)_(.+)", suffix)
        if match:
            modality, axis, statistic = match.groups()
            modalities = [modality]
            axes = list("xyz") if axis == "mag" or "grav_angle" in statistic else [axis]
        else:
            physical_axis = re.fullmatch(r"(ACC|GYRO)_([XYZ])_(DF|ENERGY_phys)", suffix)
            if physical_axis:
                modality, axis, statistic = physical_axis.groups()
                modalities, axes = [modality.lower()], [axis.lower()]
            else:
                statistic = suffix
                modalities = (["gyro"] if suffix in {"ARE", "ARATG"} else
                              ["acc", "gyro"] if suffix == "acc_gyro_ratio" else ["acc"])
                # These axes follow PhysicalFeatureConfig's current defaults.
                axes = ["x"] if suffix in {"AVG", "ARATG"} else list("yz") if suffix == "AVH" else list("xyz")
        catalog[name] = {"group": partition["feature_to_group"][name], "position": prefix,
                         "body_position": sensors[prefix]["body_position"],
                         "modalities": modalities, "axes": axes, "statistic": statistic}
    return catalog


def resolve_plan_features(plan, feature_names, *, dataset, config=None):
    """Compile A2 group bindings and optional selectors into exact ordered columns.

    Empty selectors mean unrestricted. A derived feature is retained only when
    ALL of its signal modalities/axes satisfy the corresponding selector.
    """
    config = config or load_groups()
    names = list(map(str, feature_names))
    catalog = feature_catalog(names, dataset=dataset, config=config)
    resolved = {}
    for attr in plan["attributes"]:
        primary, auxiliary = attr.get("primary_groups"), attr.get("auxiliary_groups", [])
        if not isinstance(primary, list) or not primary or not isinstance(auxiliary, list):
            raise ValueError(f"{attr['name']}: at least one primary group is required")
        selected = primary + auxiliary
        if any(not isinstance(g, str) or g not in config["groups"] for g in selected):
            raise ValueError(f"{attr['name']}: unknown feature group; use G1-G14")
        if len(set(selected)) != len(selected):
            raise ValueError(f"{attr['name']}: duplicate primary/auxiliary groups")
        evidence = attr.get("group_evidence", {})
        if not isinstance(evidence, dict) or set(evidence) != set(selected) or any(
            not isinstance(value, str) or not value.strip() for value in evidence.values()
        ):
            raise ValueError(f"{attr['name']}: give evidence for every bound group")
        filters = attr.get("feature_selectors", {})
        if not isinstance(filters, dict) or set(filters) - {"positions", "modalities", "axes"}:
            raise ValueError("Unknown feature selector")
        allowed = {"positions": set(config["dataset_sensors"][dataset]),
                   "modalities": {"acc", "gyro"}, "axes": set("xyz")}
        for field, choices in allowed.items():
            values = filters.get(field, [])
            if not isinstance(values, list) or any(not isinstance(v, str) or v not in choices for v in values):
                raise ValueError(f"Invalid {field} selector for {attr['name']}")
            if len(set(values)) != len(values):
                raise ValueError(f"Duplicate {field} selector")
        indices = []
        for i, name in enumerate(names):
            tags = catalog[name]
            if tags["group"] not in selected:
                continue
            if filters.get("positions") and tags["position"] not in filters["positions"]:
                continue
            if any(filters.get(field) and not set(tags[field]) <= set(filters[field])
                   for field in ("modalities", "axes")):
                continue
            indices.append(i)
        present = {catalog[names[i]]["group"] for i in indices}
        if set(selected) - present:
            raise ValueError(
                f"{attr['name']}: no available selected features for "
                f"{sorted(set(selected)-present)} under selectors {filters}"
            )
        if attr["name"] in resolved:
            raise ValueError("Duplicate A2 attribute name")
        resolved[attr["name"]] = {"primary_groups": primary, "auxiliary_groups": auxiliary,
                                  "feature_selectors": filters, "indices": indices,
                                  "feature_names": [names[i] for i in indices], "dimension": len(indices)}
    return {"dataset": dataset, "group_version": config["version"],
            "feature_names": names, "feature_metadata": catalog, "attributes": resolved}
