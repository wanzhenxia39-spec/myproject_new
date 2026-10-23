import importlib.util
import json
from pathlib import Path

from zshar.workspace_paths import normalize_migrated_paths, stable_migrated_path_text


def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/refinement_xxapi_common.py"
    spec = importlib.util.spec_from_file_location("migration_refinement_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_known_migrated_paths_are_equivalent():
    old = {"/home/wanzhen/myproject_new/vector.json": "original-hash",
           "input": "/home/wanzhen/windows/data.npy"}
    new = {"/data/wanzhen/projects/myproject_new/vector.json": "original-hash",
           "input": "/data/wanzhen/projects/windows/data.npy"}
    assert normalize_migrated_paths(old) == normalize_migrated_paths(new)
    assert normalize_migrated_paths("/data/wanzhen/LLMbasedZeroshotHARAgent/old") == "/data/wanzhen/LLMbasedZeroshotHARAgent/old"
    assert normalize_migrated_paths("/data/wanzhen/projects/unrelated/a") == "/data/wanzhen/projects/unrelated/a"
    assert normalize_migrated_paths("/data/wanzhen/projects/myproject_new_extra/a") == "/data/wanzhen/projects/myproject_new_extra/a"


def test_refinement_resume_accepts_relocation_but_rejects_modified_files():
    module = launcher()
    old = {"seed": 42, "model": "gpt-6.1-sol", "launcher_sha256": "old",
           "input_files": {"/home/wanzhen/myproject_new/a.json": "same"}}
    new = {**old, "launcher_sha256": "new",
           "input_files": {"/data/wanzhen/projects/myproject_new/a.json": "same"}}
    assert module.configuration_matches(old, new)
    assert not module.configuration_matches(old, {**new, "input_files": {"/data/wanzhen/projects/myproject_new/a.json": "changed"}})
    assert not module.configuration_matches(old, {**new, "seed": 43})


def test_gap_path_labels_keep_identical_checkpoint_bytes():
    old = 'a1_json,a1_sha256,D_kNN\n/home/wanzhen/myproject_new/a.json,same,0.2\n'
    new = old.replace("/home/wanzhen/myproject_new", "/data/wanzhen/projects/myproject_new")
    assert stable_migrated_path_text(new) == old
    assert stable_migrated_path_text(new.replace("same", "changed")) != old
