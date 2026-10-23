"""Shared ordinal-level helpers used by metrics and prototype matching."""

from __future__ import annotations

from collections.abc import Iterable


THREE_LEVEL_ORDER = ("Low", "Medium", "High")
FIVE_LEVEL_ORDER = ("Very Low", "Low", "Medium", "High", "Very High")


def resolve_ordinal_values(labels: Iterable[object]) -> dict[str, float]:
    """Return normalized ordinal values for a supported observed label schema.

    ``Low``/``High`` is treated as a two-level schema.  The traditional
    Low/Medium/High schema keeps its historical 0/0.5/1 encoding.  Presence of
    either five-level endpoint selects the 0/0.25/0.5/0.75/1 encoding.
    Unknown labels retain the legacy deterministic fallback: lexical order
    with unit spacing.
    """

    observed = {str(label) for label in labels if label is not None}
    if not observed:
        return {}

    five = set(FIVE_LEVEL_ORDER)
    if observed.issubset(five):
        if observed & {"Very Low", "Very High"}:
            order = FIVE_LEVEL_ORDER
        elif observed.issubset({"Low", "High"}):
            order = ("Low", "High")
        else:
            order = THREE_LEVEL_ORDER
        denominator = max(1, len(order) - 1)
        return {label: index / denominator for index, label in enumerate(order)}

    ordered = sorted(observed)
    return {label: float(index) for index, label in enumerate(ordered)}


def ordered_ordinal_labels(labels: Iterable[object]) -> list[str]:
    """Return observed labels in their resolved ordinal order."""

    observed = {str(label) for label in labels if label is not None}
    values = resolve_ordinal_values(observed)
    return sorted(observed, key=lambda label: (values[label], label))
