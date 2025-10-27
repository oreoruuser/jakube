from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pytest

from jakube import HammingIndex


def _hamming_distance(a: Sequence[int], b: Sequence[int]) -> float:
    a_vec = np.asarray(a, dtype=np.uint8)
    b_vec = np.asarray(b, dtype=np.uint8)
    return float(np.sum(np.bitwise_xor(a_vec, b_vec)))


def _build_index(
    vectors: Iterable[Sequence[int]], *, seed: int = 1337, trees: int = 10
) -> Tuple[HammingIndex, list[list[int]]]:
    vectors = [list(map(int, vector)) for vector in vectors]
    if not vectors:
        raise ValueError("Expected at least one vector to build the index.")

    index = HammingIndex(len(vectors[0]))
    if hasattr(index, "set_seed"):
        index.set_seed(seed)
    if hasattr(index, "verbose"):
        index.verbose(False)

    for item_id, vector in enumerate(vectors):
        index.add_item(int(item_id), vector)

    index.build(trees)
    return index, vectors


def test_basic_conversion() -> None:
    rng = np.random.default_rng(123)
    vectors = (
        rng.binomial(1, 0.5, size=100).tolist(),
        rng.binomial(1, 0.5, size=100).tolist(),
    )
    index, stored = _build_index(vectors, seed=42, trees=20)

    for item_id, original in enumerate(stored):
        recovered = index.get_item(item_id)
        np.testing.assert_allclose(recovered, original, rtol=0.0, atol=0.0)
        assert index.get_distance(item_id, item_id) == pytest.approx(0.0, abs=1e-8)

    expected = _hamming_distance(stored[0], stored[1])
    assert index.get_distance(0, 1) == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert index.get_distance(1, 0) == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_basic_neighbors_and_distances() -> None:
    rng = np.random.default_rng(321)
    vectors = (
        rng.binomial(1, 0.5, size=100).tolist(),
        rng.binomial(1, 0.5, size=100).tolist(),
    )
    index, stored = _build_index(vectors, seed=55, trees=25)

    ids, distances = index.get_nns_by_item(0, 2)
    assert list(ids) == [0, 1]
    assert float(distances[0]) == pytest.approx(0.0, abs=1e-8)
    assert float(distances[1]) == pytest.approx(
        _hamming_distance(stored[0], stored[1]), rel=1e-6, abs=1e-6
    )

    ids, distances = index.get_nns_by_item(1, 2)
    assert list(ids) == [1, 0]
    assert float(distances[0]) == pytest.approx(0.0, abs=1e-8)
    assert float(distances[1]) == pytest.approx(
        _hamming_distance(stored[0], stored[1]), rel=1e-6, abs=1e-6
    )


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(987)
    vectors = (
        rng.binomial(1, 0.5, size=100).tolist(),
        rng.binomial(1, 0.5, size=100).tolist(),
        rng.binomial(1, 0.5, size=100).tolist(),
    )
    index, stored = _build_index(vectors, seed=77, trees=30)

    path = tmp_path / "hamming.index"
    index.save(str(path))

    restored = HammingIndex(len(stored[0]))
    restored.load(str(path))

    assert restored.n_items() == len(stored)
    np.testing.assert_allclose(restored.get_item(1), stored[1], rtol=0.0, atol=0.0)

    ids_original, _ = index.get_nns_by_item(0, len(stored))
    ids_restored, _ = restored.get_nns_by_item(0, len(stored))
    assert list(ids_restored) == list(ids_original)


def test_many_vectors_distance_bounds() -> None:
    rng = np.random.default_rng(2025)
    dim = 10
    vectors = [rng.binomial(1, 0.5, size=dim).tolist() for _ in range(5000)]
    index, _ = _build_index(vectors, seed=88, trees=35)

    ids, distances = index.get_nns_by_vector([0] * dim, 100)
    assert min(distances) >= 0.0
    assert max(distances) <= dim

    sample_dists: list[float] = []
    for _ in range(200):
        query = rng.binomial(1, 0.5, size=dim).tolist()
        _, dists = index.get_nns_by_vector(query, 1, search_k=1000)
        sample_dists.append(float(dists[0]))

    avg_dist = float(np.mean(sample_dists))
    assert avg_dist <= dim * 0.45


def test_distance_consistency_random_subset() -> None:
    rng = np.random.default_rng(11)
    n_items, dim = 256, 32
    data = [rng.binomial(1, 0.5, size=dim).tolist() for _ in range(n_items)]
    index, _ = _build_index(data, seed=5, trees=40)

    sampler = random.Random(404)
    for anchor in sampler.sample(range(n_items), 25):
        ids, distances = index.get_nns_by_item(anchor, min(50, n_items))
        for neighbor, distance in zip(ids, distances):
            expected = _hamming_distance(data[anchor], data[neighbor])
            assert float(distance) == pytest.approx(expected, rel=1e-6, abs=1e-6)
            assert index.get_distance(anchor, neighbor) == pytest.approx(
                expected, rel=1e-6, abs=1e-6
            )
