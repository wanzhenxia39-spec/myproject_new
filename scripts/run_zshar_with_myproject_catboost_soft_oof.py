#!/usr/bin/env python3
"""Run the myproject_new CLI with the requested myproject CatBoost soft-OOF class."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys


NEW_SRC = Path("/home/wanzhen/myproject_new/src")
REQUESTED_BACKEND = Path(
    "/home/wanzhen/myproject/src/zshar/models/catboost_soft_oof_backend.py"
)
if str(NEW_SRC) not in sys.path:
    sys.path.insert(0, str(NEW_SRC))


def install_requested_backend() -> type:
    """Load the requested source file in the active zshar.models package."""
    module_name = "zshar.models._requested_myproject_catboost_soft_oof_backend"
    spec = importlib.util.spec_from_file_location(module_name, REQUESTED_BACKEND)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load requested backend: {REQUESTED_BACKEND}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    backend_class = module.CatBoostSoftOOFModelManager
    actual = Path(inspect.getfile(backend_class)).resolve()
    if actual != REQUESTED_BACKEND.resolve():
        raise RuntimeError(f"Unexpected CatBoost soft-OOF source: {actual}")

    import zshar.models as models
    import zshar.models.factory as factory

    models.CatBoostSoftOOFModelManager = backend_class
    factory.CatBoostSoftOOFModelManager = backend_class
    print(f"Verified requested CatBoost soft-OOF backend: {actual}", file=sys.stderr)
    return backend_class


def main() -> None:
    install_requested_backend()
    from zshar.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
