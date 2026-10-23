"""Single source of truth for the initial nine-attribute ontology."""
from __future__ import annotations

import json
from pathlib import Path
from zshar.paths import PROJECT_ROOT

DEFAULT_ONTOLOGY = PROJECT_ROOT / "configs/attributes/initial_attributes.json"


def load_ontology(path: str | Path = DEFAULT_ONTOLOGY) -> dict:
    ontology = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(ontology) != {"version", "attributes"}:
        raise ValueError("Ontology must contain only version and attributes")
    attrs = ontology["attributes"]
    names = [a["name"] for a in attrs]
    if not names or len(names) != len(set(names)):
        raise ValueError("Ontology must have nonempty, unique attribute names")
    for attr in attrs:
        if set(attr) != {"name", "definition", "levels"}:
            raise ValueError(
                f"Ontology attribute {attr.get('name')!r} must contain only name, definition, and levels"
            )
        if not isinstance(attr["definition"], str) or not attr["definition"].strip():
            raise ValueError(f"Missing definition for {attr['name']}")
        if set(attr["levels"]) != {"Low", "Medium", "High"}:
            raise ValueError(f"Invalid level definitions for {attr['name']}")
        if any(not isinstance(value, str) or not value.strip() for value in attr["levels"].values()):
            raise ValueError(f"Empty level meaning for {attr['name']}")
    return ontology


def initial_attribute_names() -> list[str]:
    return [attr["name"] for attr in load_ontology()["attributes"]]
