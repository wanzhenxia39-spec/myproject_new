"""Known aliases for the checksum-verified October 2026 workspace relocation.

Only path spellings are normalized; file hashes, seeds and model settings retain
their exact comparison semantics. No saved experiment files are rewritten here.
"""
from pathlib import Path


MIGRATED_ROOTS = ("myproject_new", "myproject", "newagent", "LLMbasedZeroshotHARAgent",
                  "windows", "runtime", "gpt_run.py")
ALIASES = tuple((f"/data/wanzhen/projects/{name}", f"/home/wanzhen/{name}")
                for name in MIGRATED_ROOTS)


def normalize_migrated_paths(value):
    if isinstance(value, dict):
        return {normalize_migrated_paths(key): normalize_migrated_paths(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_migrated_paths(item) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        for physical, logical in ALIASES:
            if value == physical or value.startswith(physical + "/"):
                return logical + value[len(physical):]
    return value


def stable_migrated_path_text(text):
    """Retain original path labels in deterministic GAP CSV fingerprints."""
    for physical, logical in ALIASES:
        text = text.replace(physical + "/", logical + "/")
    return text
