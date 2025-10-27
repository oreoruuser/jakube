from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import pytest

from jakube import EuclideanIndex


def _build_index(
    n_points: int,
    *,
    dim: int = 10,
    trees: int = 10,
    seed: int = 1337,
    vectors: Iterable[Iterable[float]] | None = None,
) -> EuclideanIndex:
    index = EuclideanIndex(dim)
    if hasattr(index, "set_seed"):
        index.set_seed(seed)
    if hasattr(index, "verbose"):
        index.verbose(False)

    if vectors is None:
        rng = np.random.default_rng(seed)
        vectors = rng.normal(size=(n_points, dim)).astype(np.float32).tolist()
    else:
        vectors = [list(map(float, vector)) for vector in vectors]

    for item_id, vector in enumerate(vectors):
        index.add_item(int(item_id), vector)

    index.build(trees)
    return index


def test_numpy_vectors(n_points: int = 1000, n_trees: int = 10) -> None:
    dim = 10
    rng = np.random.default_rng(2025)
    data = []
    for _ in range(n_points):
        array = rng.normal(size=dim)
        array = array.astype(
            random.choice([np.float64, np.float32, np.uint8, np.int16])
        )
        data.append(array.tolist())

    index = _build_index(n_points, dim=dim, trees=n_trees, vectors=data)
    assert index.n_items() == n_points


def test_tuple_input(n_points: int = 1000, n_trees: int = 10) -> None:
    dim = 10
    data = [tuple(random.gauss(0.0, 1.0) for _ in range(dim)) for _ in range(n_points)]
    index = _build_index(n_points, dim=dim, trees=n_trees, vectors=data)
    assert index.n_items() == n_points


def test_wrong_length_vectors(n_points: int = 1000, n_trees: int = 10) -> None:
    dim = 10
    index = EuclideanIndex(dim)
    index.set_seed(7)
    index.verbose(False)
    index.add_item(0, [random.gauss(0.0, 1.0) for _ in range(dim)])

    with pytest.raises(Exception):
        index.add_item(1, [random.gauss(0.0, 1.0) for _ in range(dim + 5)])
    with pytest.raises(Exception):
        index.add_item(2, [])

    for item_id in range(1, n_points):
        index.add_item(item_id + 1, [random.gauss(0.0, 1.0) for _ in range(dim)])

    index.build(n_trees)


def test_range_errors(n_points: int = 1000, n_trees: int = 10) -> None:
    dim = 10
    index = EuclideanIndex(dim)
    index.set_seed(11)
    index.verbose(False)
    for item_id in range(n_points):
        index.add_item(item_id, [random.gauss(0.0, 1.0) for _ in range(dim)])

    index.build(n_trees)

    distance = index.get_distance(0, n_points)
    assert isinstance(distance, float)

    ids, distances = index.get_nns_by_item(n_points, 1)
    assert len(ids) == len(distances) == 1
    assert isinstance(ids[0], int)
    assert isinstance(distances[0], float)

    vector = index.get_item(n_points)
    assert isinstance(vector, list)
    assert len(vector) == dim
    assert all(isinstance(value, float) for value in vector)


def test_missing_len() -> None:
    class FakeCollection:
        pass

    index = EuclideanIndex(10)
    with pytest.raises(RuntimeError) as excinfo:
        index.add_item(1, FakeCollection())
    assert "bad_cast" in str(excinfo.value)


def test_missing_getitem() -> None:
    class FakeCollection:
        def __len__(self) -> int:
            return 5

    index = EuclideanIndex(5)
    with pytest.raises(RuntimeError) as excinfo:
        index.add_item(1, FakeCollection())
    assert "bad_cast" in str(excinfo.value)


def test_short_collection() -> None:
    class FakeCollection:
        def __len__(self) -> int:
            return 3

        def __getitem__(self, _: int) -> float:
            raise IndexError

    index = EuclideanIndex(3)
    with pytest.raises(Exception):
        index.add_item(1, FakeCollection())


def test_non_float_values() -> None:
    array_strings = ["1", "2", "3"]

    index = EuclideanIndex(3)
    with pytest.raises(RuntimeError) as excinfo:
        index.add_item(1, array_strings)
    assert "bad_cast" in str(excinfo.value)
