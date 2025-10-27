from __future__ import annotations

import random
from typing import Iterable, Sequence, Tuple

import numpy as np

from jakube import AngularIndex


def _build_index(
	pairs: Iterable[Tuple[int, Sequence[float]]], *, dim: int, trees: int = 10, seed: int = 1337
) -> AngularIndex:
	index = AngularIndex(dim)
	index.set_seed(seed)
	index.verbose(False)
	for item_id, vector in pairs:
		index.add_item(int(item_id), list(map(float, vector)))
	index.build(trees)
	return index


def test_random_holes() -> None:
	dim = 10
	rng = np.random.default_rng(2025)
	valid_indices = set(random.sample(range(2000), 1000))

	items = [(item_id, rng.normal(size=dim)) for item_id in sorted(valid_indices)]
	index = _build_index(items, dim=dim, trees=25, seed=42)

	for item_id in valid_indices:
		ids, _ = index.get_nns_by_item(int(item_id), 100)
		assert all(int(neighbor) in valid_indices for neighbor in ids)

	for _ in range(200):
		query = rng.normal(size=dim)
		ids, _ = index.get_nns_by_vector(query.tolist(), 100)
		assert all(int(neighbor) in valid_indices for neighbor in ids)


def _test_holes_base(count: int, *, dim: int = 100, base_id: int = 100000) -> None:
	rng = np.random.default_rng(1337)
	items = [
		(base_id + offset, rng.normal(size=dim).tolist())
		for offset in range(count)
	]
	index = _build_index(items, dim=dim, trees=40, seed=256)

	ids, _ = index.get_nns_by_item(int(base_id), count)
	expected = {base_id + offset for offset in range(count)}
	assert set(map(int, ids)) == expected


def test_root_one_child() -> None:
	# Regression for https://github.com/spotify/annoy/issues/223
	_test_holes_base(1)


def test_root_two_children() -> None:
	_test_holes_base(2)


def test_root_some_children() -> None:
	# Regression for https://github.com/spotify/annoy/issues/295
	_test_holes_base(10)


def test_root_many_children() -> None:
	_test_holes_base(256)
