from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def jakube_ext() -> Generator[object, None, None]:
    """Import the built extension module, searching in-tree artifacts if needed."""

    try:
        module = importlib.import_module("jakube_ext")
        yield module
        return
    except ModuleNotFoundError:
        pass

    project_root = Path(__file__).resolve().parents[1]
    for suffix in ("so", "dylib", "pyd"):
        for artifact in project_root.glob(f"build/**/*jakube_ext*.{suffix}"):
            spec = importlib.util.spec_from_file_location("jakube_ext", artifact)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                yield module
                return

    pytest.skip("jakube_ext extension not built", allow_module_level=True)
