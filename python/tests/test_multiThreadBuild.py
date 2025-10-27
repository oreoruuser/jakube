from __future__ import annotations

import inspect
from typing import Iterable

import numpy as np
import pytest

from jakube import EuclideanIndex


def _prepare_index(data: Iterable[Iterable[float]], *, seed: int = 2025) -> EuclideanIndex:
    vectors = [list(map(float, vector)) for vector in data]
    if not vectors:
        raise ValueError("expected at least one vector")

    index = EuclideanIndex(len(vectors[0]))
    if hasattr(index, "set_seed"):
        index.set_seed(seed)
    if hasattr(index, "verbose"):
        index.verbose(False)

    for item_id, vector in enumerate(vectors):
        index.add_item(int(item_id), vector)

    return index


def _build_with_threads(n_jobs: int) -> None:
    rng = np.random.default_rng(1337 + n_jobs)
    data = rng.normal(size=(2000, 10)).astype(np.float32).tolist()
    index = _prepare_index(data, seed=9000 + n_jobs)
    trees = 31

    signature = inspect.signature(index.build)
    if "n_jobs" in signature.parameters:
        result = index.build(trees, n_jobs=n_jobs)
        if result is not None:
            assert bool(result)
        assert index.n_trees() >= trees
        return

    if n_jobs == 1:
        index.build(trees)
        assert index.n_trees() >= trees
        return

    with pytest.raises(TypeError):
        index.build(trees, n_jobs=n_jobs)


def test_one_thread() -> None:
    _build_with_threads(1)


def test_two_threads() -> None:
    _build_with_threads(2)


def test_four_threads() -> None:
    _build_with_threads(4)


def test_eight_threads() -> None:
    _build_with_threads(8)
