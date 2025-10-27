from __future__ import annotations

import random
from typing import Iterable, Sequence, Tuple

import numpy as np
import pytest

from jakube import ManhattanIndex


def _manhattan_distance(a: Sequence[float], b: Sequence[float]) -> float:
	a_vec = np.asarray(a, dtype=np.float32)
	b_vec = np.asarray(b, dtype=np.float32)
	return float(np.sum(np.abs(a_vec - b_vec)))


def _build_index(
	vectors: Iterable[Sequence[float]], *, seed: int = 1337, trees: int = 10
) -> Tuple[ManhattanIndex, list[list[float]]]:
	vectors = [list(map(float, vector)) for vector in vectors]
	if not vectors:
		raise ValueError("Expected at least one vector to build the index.")

	index = ManhattanIndex(len(vectors[0]))
	if hasattr(index, "set_seed"):
		index.set_seed(seed)
	if hasattr(index, "verbose"):
		index.verbose(False)

	for item_id, vector in enumerate(vectors):
		index.add_item(int(item_id), vector)

	index.build(trees)
	return index, vectors


def test_get_nns_by_vector_returns_expected_ids() -> None:
	vectors = ([2.0, 2.0], [3.0, 2.0], [3.0, 3.0])
	index, _ = _build_index(vectors, seed=42, trees=20)

	queries = ([4.0, 4.0], [1.0, 1.0], [5.0, 3.0])
	expected = ([2, 1, 0], [0, 1, 2], [2, 1, 0])

	for query, expected_ids in zip(queries, expected):
		ids, _ = index.get_nns_by_vector(list(map(float, query)), len(expected_ids))
		assert list(ids) == list(expected_ids)


def test_get_nns_by_item_returns_expected_ids() -> None:
	vectors = ([2.0, 2.0], [3.0, 2.0], [3.0, 3.0])
	index, _ = _build_index(vectors, seed=7, trees=20)

	ids, _ = index.get_nns_by_item(0, 3)
	assert list(ids) == [0, 1, 2]

	ids, _ = index.get_nns_by_item(2, 3)
	assert list(ids) == [2, 1, 0]


def test_get_distance_matches_l1_norm() -> None:
	vectors = ([0.0, 1.0], [1.0, 1.0], [0.0, 0.0])
	index, stored = _build_index(vectors, seed=99, trees=5)

	expected = _manhattan_distance(stored[0], stored[1])
	assert index.get_distance(0, 1) == pytest.approx(expected, rel=1e-6, abs=1e-6)

	expected = _manhattan_distance(stored[1], stored[2])
	assert index.get_distance(1, 2) == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_large_index_recovers_close_pairs() -> None:
	dim = 10
	index = ManhattanIndex(dim)
	index.set_seed(1234)
	index.verbose(False)

	rng = random.Random(2025)
	n_items = 2000
	for item_id in range(0, n_items, 2):
		base = [rng.gauss(0.0, 1.0) for _ in range(dim)]
		x = [base_val + rng.gauss(0.0, 1e-2) for base_val in base]
		y = [base_val + rng.gauss(0.0, 1e-2) for base_val in base]
		index.add_item(item_id, x)
		index.add_item(item_id + 1, y)

	index.build(15)

	for item_id in range(0, n_items, 2):
		ids, _ = index.get_nns_by_item(item_id, 2)
		assert list(ids) == [item_id, item_id + 1]
		ids, _ = index.get_nns_by_item(item_id + 1, 2)
		assert list(ids) == [item_id + 1, item_id]


def _precision(
	n: int,
	*,
	n_trees: int = 10,
	n_points: int = 2000,
	n_rounds: int = 3,
	search_k: int = 100000,
) -> float:
	rng = random.Random(1337)
	total = 0.0
	dim = 10

	for round_idx in range(n_rounds):
		data = []
		index = ManhattanIndex(dim)
		index.set_seed(8000 + round_idx)
		index.verbose(False)
		for item_id in range(n_points):
			vector = [rng.gauss(0.0, 1.0) for _ in range(dim)]
			norm = sum(value * value for value in vector) ** 0.5 or 1.0
			scaled = [(value / norm) + item_id for value in vector]
			data.append(scaled)
			index.add_item(item_id, scaled)
		index.build(n_trees)

		ids, _ = index.get_nns_by_vector([0.0] * dim, n, search_k)
		ordered = list(ids)
		assert ordered == sorted(ordered)
		matches = sum(1 for candidate in ordered if candidate < n)
		total += matches

	return total / float(n * n_rounds)


@pytest.mark.parametrize("n, threshold", [(1, 0.98), (10, 0.95), (100, 0.92), (1000, 0.88)])
def test_precision_thresholds(n: int, threshold: float) -> None:
	assert _precision(n) >= threshold


def test_get_nns_with_distances() -> None:
	vectors = ([0.0, 0.0, 2.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0])
	index, stored = _build_index(vectors, seed=55, trees=20)

	ids, distances = index.get_nns_by_item(0, 3)
	assert list(ids) == [0, 1, 2]
	expected = [_manhattan_distance(stored[0], stored[idx]) for idx in ids]
	for distance, expected_distance in zip(distances, expected):
		assert float(distance) == pytest.approx(expected_distance, rel=1e-6, abs=1e-6)

	ids, distances = index.get_nns_by_vector([2.0, 2.0, 1.0], 3)
	assert list(ids) == [1, 2, 0]
	expected = [_manhattan_distance([2.0, 2.0, 1.0], stored[idx]) for idx in ids]
	for distance, expected_distance in zip(distances, expected):
		assert float(distance) == pytest.approx(expected_distance, rel=1e-6, abs=1e-6)


def test_include_dists() -> None:
	dim = 40
	rng = np.random.default_rng(2024)
	vector = rng.normal(size=dim).astype(np.float32)
	vectors = (vector.tolist(), (-vector).tolist())
	index, stored = _build_index(vectors, seed=64, trees=15)

	ids, distances = index.get_nns_by_item(0, 2)
	assert list(ids) == [0, 1]

	expected_self = _manhattan_distance(stored[0], stored[0])
	expected_opposite = _manhattan_distance(stored[0], stored[1])
	assert float(distances[0]) == pytest.approx(expected_self, rel=1e-6, abs=1e-6)
	assert float(distances[1]) == pytest.approx(expected_opposite, rel=1e-6, abs=1e-6)


def test_distance_consistency() -> None:
	rng = np.random.default_rng(7)
	n_items, dim = 500, 3
	data = rng.normal(size=(n_items, dim)).astype(np.float32)

	index = ManhattanIndex(dim)
	index.set_seed(17)
	index.verbose(False)
	for item_id, vector in enumerate(data.tolist()):
		index.add_item(int(item_id), vector)
	index.build(20)

	sampler = random.Random(404)
	for anchor in sampler.sample(range(n_items), 50):
		ids, distances = index.get_nns_by_item(anchor, min(100, n_items))
		for neighbor, distance in zip(ids, distances):
			expected = _manhattan_distance(data[anchor], data[neighbor])
			assert float(distance) == pytest.approx(expected, rel=1e-6, abs=1e-6)
			assert index.get_distance(anchor, neighbor) == pytest.approx(expected, rel=1e-6, abs=1e-6)
