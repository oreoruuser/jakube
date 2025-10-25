from __future__ import annotations

import pytest


def _make_dataset(dims: int, n_items: int):
    data = []
    for i in range(n_items):
        vector = [float(i) + j * 0.1 for j in range(dims)]
        data.append(vector)
    return data


def test_seed_determinism(jakube_ext):
    dims = 6
    data = _make_dataset(dims, 100)
    query = [float(j) for j in range(dims)]

    first = jakube_ext.EuclideanIndex(dims)
    first.set_seed(1337)
    for idx, vector in enumerate(data):
        first.add_item(idx, vector)
    first.build(12)
    ids_first, _ = first.get_nns_by_vector(query, 10)

    second = jakube_ext.EuclideanIndex(dims)
    second.set_seed(1337)
    for idx, vector in enumerate(data):
        second.add_item(idx, vector)
    second.build(12)
    ids_second, _ = second.get_nns_by_vector(query, 10)

    assert ids_first == ids_second


def test_seed_changes_results(jakube_ext):
    dims = 6
    data = _make_dataset(dims, 100)

    def build_index(seed: int | None):
        index = jakube_ext.EuclideanIndex(dims)
        if seed is not None:
            index.set_seed(seed)
        for idx, vector in enumerate(data):
            index.add_item(idx, vector)
        index.build(6)
        return index

    baseline = build_index(1234)
    baseline_results = []
    for query_idx in range(10):
        ids, _ = baseline.get_nns_by_vector(data[query_idx], 10, search_k=15)
        baseline_results.append(ids)

    different_order_found = False
    for offset in range(1, 21):
        variant = build_index(1234 + offset)
        for query_idx in range(10):
            ids_variant, _ = variant.get_nns_by_vector(
                data[query_idx], 10, search_k=15
            )
            if ids_variant != baseline_results[query_idx]:
                different_order_found = True
                break
        if different_order_found:
            break

    if not different_order_found:
        pytest.fail(
            "Different seeds produced identical neighbour orderings across attempts"
        )
