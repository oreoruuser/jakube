from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pytest

from jakube import AngularIndex


def _angular_distance(a: Sequence[float], b: Sequence[float]) -> float:
	a_vec = np.asarray(a, dtype=np.float32)
	b_vec = np.asarray(b, dtype=np.float32)
	dot = float(np.dot(a_vec, b_vec))
	norm = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
	if norm == 0.0:
		return math.sqrt(2.0)
	cosine = max(-1.0, min(1.0, dot / norm))
	return math.sqrt(max(0.0, 2.0 - 2.0 * cosine))


def _build_index(vectors: Iterable[Sequence[float]], *, seed: int = 1337, trees: int = 10) -> Tuple[AngularIndex, list[Sequence[float]]]:
	vectors = [list(map(float, vector)) for vector in vectors]
	index = AngularIndex(len(vectors[0]))
	index.set_seed(seed)
	index.verbose(False)
	for item_id, vector in enumerate(vectors):
		index.add_item(int(item_id), vector)
	index.build(trees)
	return index, vectors


def test_get_nns_by_vector_returns_expected_ids() -> None:
	vectors = ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
	index, _ = _build_index(vectors, seed=42, trees=20)

	queries = ([3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [2.0, 0.0, 1.0])
	expected = ([2, 1, 0], [0, 1, 2], [2, 0, 1])

	for query, expected_ids in zip(queries, expected):
		ids, _ = index.get_nns_by_vector(list(map(float, query)), len(expected_ids))
		assert list(ids) == list(expected_ids)


def test_get_nns_by_item_returns_expected_ids() -> None:
	vectors = ([2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0])
	index, _ = _build_index(vectors, seed=7, trees=20)

	expected = (
		(0, [0, 1, 2]),
		(1, [1, 0, 2]),
		(2, ([2, 0, 1], [2, 1, 0])),
	)

	item_id, ids = expected[0]
	result_ids, _ = index.get_nns_by_item(item_id, 3)
	assert list(result_ids) == ids

	item_id, ids = expected[1]
	result_ids, _ = index.get_nns_by_item(item_id, 3)
	assert list(result_ids) == ids

	item_id, options = expected[2]
	result_ids, _ = index.get_nns_by_item(item_id, 3)
	assert list(result_ids) in [list(option) for option in options]


def test_get_distance_matches_expected_formula() -> None:
	vectors = ([0.0, 1.0], [1.0, 1.0], [1000.0, 0.0], [10.0, 0.0], [97.0, 0.0], [42.0, 42.0])
	index, vectors = _build_index(vectors, seed=11, trees=5)

	pairs = [(0, 1), (2, 3), (4, 5)]
	for a, b in pairs:
		expected = _angular_distance(vectors[a], vectors[b])
		assert index.get_distance(a, b) == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_get_distance_with_zero_vector() -> None:
	vectors = ([1.0, 0.0], [0.0, 0.0])
	index, _ = _build_index(vectors, seed=21, trees=5)

	expected = math.sqrt(2.0)
	assert index.get_distance(0, 1) == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_neighbors_include_distances_sorted() -> None:
	vectors = ([1.0, 0.0], [0.8, 0.2], [0.0, 1.0])
	index, vectors = _build_index(vectors, seed=101, trees=10)

	ids, distances = index.get_nns_by_item(0, 3)
	ids = list(ids)
	distances = [float(d) for d in distances]

	assert distances == sorted(distances)
	for neighbor_id, distance in zip(ids, distances):
		expected = _angular_distance(vectors[0], vectors[neighbor_id])
		assert distance == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
	vectors = ([1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9])
	index, stored = _build_index(vectors, seed=2024, trees=15)

	path = tmp_path / "angular.index"
	index.save(str(path))

	restored = AngularIndex(len(vectors[0]))
	restored.load(str(path))

	assert restored.n_items() == 3
	np.testing.assert_allclose(restored.get_item(1), stored[1], rtol=1e-6, atol=1e-6)

	query = [0.9, 0.1, 0.0]
	ids_original, _ = index.get_nns_by_vector(query, 3)
	ids_restored, _ = restored.get_nns_by_vector(query, 3)
	assert list(ids_restored) == list(ids_original)


def test_close_pairs_are_mutual_neighbors() -> None:
	rng = random.Random(1337)
	dim = 10
	vectors = []
	for j in range(0, 200, 2):
		base = np.array([rng.gauss(0.0, 1.0) for _ in range(dim)], dtype=np.float32)
		noise1 = np.array([rng.gauss(0.0, 1e-2) for _ in range(dim)], dtype=np.float32)
		noise2 = np.array([rng.gauss(0.0, 1e-2) for _ in range(dim)], dtype=np.float32)
		scale1 = 1.0 + rng.random()
		scale2 = 1.0 + rng.random()
		vectors.append(list((scale1 * base + noise1).astype(float)))
		vectors.append(list((scale2 * base + noise2).astype(float)))

	index, _ = _build_index(vectors, seed=99, trees=30)

	for j in range(0, len(vectors), 2):
		ids, _ = index.get_nns_by_item(j, 2)
		assert set(ids) == {j, j + 1}
		ids, _ = index.get_nns_by_item(j + 1, 2)
		assert set(ids) == {j, j + 1}
