from __future__ import annotations

import random
from typing import Iterable, Sequence

import numpy as np
import pytest

from jakube import EuclideanIndex


def _build_index(
    vectors: Iterable[Sequence[float]], *, trees: int = 20, seed: int = 1337
) -> EuclideanIndex:
    vectors = [list(map(float, vector)) for vector in vectors]
    if not vectors:
        raise ValueError("expected at least one vector")

    index = EuclideanIndex(len(vectors[0]))
    if hasattr(index, "set_seed"):
        index.set_seed(seed)
    if hasattr(index, "verbose"):
        index.verbose(False)

    for item_id, vector in enumerate(vectors):
        index.add_item(int(item_id), vector)

    index.build(trees)
    return index


def test_get_item_repeated_access() -> None:
    dim = 10
    rng = random.Random(42)
    vector = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    index = _build_index([vector])

    expected = list(index.get_item(0))
    for _ in range(500):
        assert list(index.get_item(0)) == expected


def test_get_many_neighbors_requests() -> None:
    dim = 10
    rng = random.Random(1337)
    vectors = [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(50)]
    index = _build_index(vectors, trees=25)

    ids, dists = index.get_nns_by_item(0, index.n_items())
    assert list(ids)[0] == 0
    assert len(ids) == len(dists)
    assert all(distance >= 0.0 for distance in dists)


def test_repeated_build_and_unbuild() -> None:
    rng = np.random.default_rng(2025)
    vectors = rng.normal(size=(100, 12)).astype(np.float32).tolist()
    index = _build_index(vectors, trees=10)

    for _ in range(5):
        index.unbuild()
        index.build(12)

    assert index.n_items() == len(vectors)


def test_include_distances_repeated_queries() -> None:
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(2000, 6)).astype(np.float32).tolist()
    index = _build_index(vectors, trees=15, seed=99)

    query = rng.normal(size=6).astype(np.float32).tolist()
    for _ in range(2000):
        ids, distances = index.get_nns_by_vector(query, 5)
        assert len(ids) == len(distances)
        assert all(distance >= 0.0 for distance in distances)
        assert all(isinstance(distance, float) for distance in distances)


@pytest.mark.parametrize("dimension", [0, -1])
def test_invalid_dimension_raises(dimension: int) -> None:
    with pytest.raises(Exception):
        EuclideanIndex(dimension)
