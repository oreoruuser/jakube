from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pytest

from jakube import DotProductIndex


def _dot_distance(a: Sequence[float], b: Sequence[float]) -> float:
	a_vec = np.asarray(a, dtype=np.float32)
	b_vec = np.asarray(b, dtype=np.float32)
	return -float(np.dot(a_vec, b_vec))


def _build_index(
	vectors: Iterable[Sequence[float]], *, seed: int = 1337, trees: int = 10
) -> Tuple[DotProductIndex, list[list[float]]]:
	vectors = [list(map(float, vector)) for vector in vectors]
	if not vectors:
		raise ValueError("Expected at least one vector to build the index.")

	index = DotProductIndex(len(vectors[0]))
	if hasattr(index, "set_seed"):
		index.set_seed(seed)
	if hasattr(index, "verbose"):
		index.verbose(False)

	for item_id, vector in enumerate(vectors):
		index.add_item(int(item_id), vector)

	index.build(trees)
	return index, vectors


def _expected_top_k(dataset: Sequence[Sequence[float]], query: Sequence[float], k: int) -> list[int]:
	candidates: list[Tuple[float, int]] = []
	for item_id, vector in enumerate(dataset):
		candidates.append((_dot_distance(vector, query), item_id))
	candidates.sort()
	return [item for _, item in candidates[:k]]


def _recall(retrieved: Sequence[int], relevant: Sequence[int]) -> float:
	target = set(relevant)
	if not target:
		return 0.0
	return len(target.intersection(retrieved)) / float(len(target))


def test_get_nns_by_vector_returns_expected_ids() -> None:
	vectors = ([2.0, 2.0], [3.0, 2.0], [3.0, 3.0])
	index, _ = _build_index(vectors, seed=42, trees=20)

	queries = ([4.0, 4.0], [1.0, 1.0], [4.0, 2.0])
	expected = ([2, 1, 0], [2, 1, 0], [2, 1, 0])

	for query, expected_ids in zip(queries, expected):
		ids, _ = index.get_nns_by_vector(list(map(float, query)), len(expected_ids))
		assert list(ids) == list(expected_ids)


def test_get_nns_by_item_returns_expected_ids() -> None:
	vectors = ([2.0, 2.0], [3.0, 2.0], [3.0, 3.0])
	index, _ = _build_index(vectors, seed=7, trees=20)

	ids, _ = index.get_nns_by_item(0, 3)
	assert list(ids) == [2, 1, 0]

	ids, _ = index.get_nns_by_item(2, 3)
	assert list(ids) == [2, 1, 0]


def test_get_distance_matches_negative_dot_product() -> None:
	vectors = ([0.0, 1.0], [1.0, 1.0], [0.0, 0.0])
	index, stored = _build_index(vectors, seed=99, trees=5)

	expected = _dot_distance(stored[0], stored[1])
	assert index.get_distance(0, 1) == pytest.approx(expected, rel=1e-6, abs=1e-6)

	expected = _dot_distance(stored[1], stored[2])
	assert index.get_distance(1, 2) == pytest.approx(expected, rel=1e-6, abs=1e-6)


def _recall_at(
	n: int,
	*,
	n_trees: int = 10,
	n_points: int = 1000,
	n_rounds: int = 5,
	search_k: int = 100000,
) -> float:
	rng = random.Random(1234)
	total = 0.0
	dim = 10

	for _ in range(n_rounds):
		data = np.array(
			[[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(n_points)],
			dtype=np.float32,
		)

		expected_results = [
			_expected_top_k(data.tolist(), data[i], n) for i in range(n_points)
		]

		index = DotProductIndex(dim)
		index.set_seed(4321)
		index.verbose(False)
		for item_id, vector in enumerate(data.tolist()):
			index.add_item(int(item_id), vector)
		index.build(n_trees)

		for item_id, vector in enumerate(data.tolist()):
			ids, _ = index.get_nns_by_vector(vector, n, search_k)
			total += _recall(list(ids), expected_results[item_id])

	return total / float(n_rounds * n_points)


def test_recall_at_10() -> None:
	assert _recall_at(10) >= 0.65


def test_recall_at_100() -> None:
	assert _recall_at(100) >= 0.95


def test_recall_at_1000() -> None:
	assert _recall_at(1000) >= 0.99


def test_recall_at_1000_fewer_trees() -> None:
	assert _recall_at(1000, n_trees=4) >= 0.99


def test_get_nns_with_distances() -> None:
	vectors = ([0.0, 0.0, 2.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0])
	index, stored = _build_index(vectors, seed=55, trees=20)

	ids, distances = index.get_nns_by_item(0, 3)
	assert list(ids) == [0, 1, 2]
	expected = [_dot_distance(stored[0], stored[idx]) for idx in ids]
	for distance, expected_distance in zip(distances, expected):
		assert float(distance) == pytest.approx(expected_distance, rel=1e-6, abs=1e-6)

	ids, distances = index.get_nns_by_vector([2.0, 2.0, 2.0], 3)
	assert list(ids) == [0, 1, 2]
	expected = [_dot_distance([2.0, 2.0, 2.0], stored[idx]) for idx in ids]
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

	expected_self = _dot_distance(stored[0], stored[0])
	expected_opposite = _dot_distance(stored[0], stored[1])
	assert float(distances[0]) == pytest.approx(expected_self, rel=1e-6, abs=1e-6)
	assert float(distances[1]) == pytest.approx(expected_opposite, rel=1e-6, abs=1e-6)


def test_distance_consistency() -> None:
	rng = np.random.default_rng(7)
	n_items, dim = 500, 3
	data = rng.normal(size=(n_items, dim)).astype(np.float32)

	index = DotProductIndex(dim)
	index.set_seed(17)
	index.verbose(False)
	for item_id, vector in enumerate(data.tolist()):
		index.add_item(int(item_id), vector)
	index.build(20)

	sampler = random.Random(404)
	for anchor in sampler.sample(range(n_items), 50):
		ids, distances = index.get_nns_by_item(anchor, min(100, n_items))
		for neighbor, distance in zip(ids, distances):
			expected = _dot_distance(data[anchor], data[neighbor])
			assert float(distance) == pytest.approx(expected, rel=1e-6, abs=1e-6)
			assert index.get_distance(anchor, neighbor) == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
	vectors = ([1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9])
	index, stored = _build_index(vectors, seed=2025, trees=25)

	path = tmp_path / "dot.index"
	index.save(str(path))

	restored = DotProductIndex(len(vectors[0]))
	restored.load(str(path))

	assert restored.n_items() == len(vectors)
	np.testing.assert_allclose(restored.get_item(1), stored[1], rtol=1e-6, atol=1e-6)

	query = [0.9, 0.1, 0.0]
	original_ids, _ = index.get_nns_by_vector(query, 3)
	restored_ids, _ = restored.get_nns_by_vector(query, 3)
	assert list(restored_ids) == list(original_ids)
