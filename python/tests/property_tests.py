from __future__ import annotations

from typing import List, Tuple

import pytest
from hypothesis import given, settings, strategies as st


@st.composite
def euclidean_dataset(draw) -> Tuple[int, List[List[float]], int, int]:
    dims = draw(st.integers(min_value=2, max_value=6))
    num_items = draw(st.integers(min_value=5, max_value=25))
    vector_strategy = st.lists(
        st.integers(min_value=-20, max_value=20), min_size=dims, max_size=dims
    )
    dataset = draw(st.lists(vector_strategy, min_size=num_items, max_size=num_items))
    query_index = draw(st.integers(min_value=0, max_value=num_items - 1))
    k = draw(st.integers(min_value=1, max_value=min(10, num_items)))
    return dims, [[float(value) for value in vec] for vec in dataset], query_index, k


@st.composite
def hamming_dataset(draw) -> Tuple[int, List[List[int]], int, int]:
    dims = draw(st.integers(min_value=1, max_value=4))
    num_items = draw(st.integers(min_value=4, max_value=32))
    vector_strategy = st.lists(
        st.integers(min_value=0, max_value=2**16 - 1), min_size=dims, max_size=dims
    )
    dataset = draw(st.lists(vector_strategy, min_size=num_items, max_size=num_items))
    query_index = draw(st.integers(min_value=0, max_value=num_items - 1))
    k = draw(st.integers(min_value=1, max_value=min(8, num_items)))
    return dims, dataset, query_index, k


@settings(max_examples=50)
@given(euclidean_dataset())
def test_euclidean_property_nearest_item(jakube_ext, data):
    dims, dataset, query_index, k = data
    index = jakube_ext.EuclideanIndex(dims)
    index.set_seed(123)
    for item_id, vector in enumerate(dataset):
        index.add_item(item_id, vector)

    index.build(min(len(dataset) * 2, 40))

    query = dataset[query_index]
    ids_vector, distances_vector = index.get_nns_by_vector(
        query, k, search_k=max(k * dims * 8, 50)
    )
    ids_item, distances_item = index.get_nns_by_item(query_index, k, search_k=500)

    assert ids_vector[0] == query_index
    assert ids_vector == ids_item
    assert pytest.approx(distances_vector, rel=1e-6) == distances_item
    assert all(a <= b + 1e-7 for a, b in zip(distances_vector, distances_vector[1:]))


@settings(max_examples=50)
@given(hamming_dataset())
def test_hamming_property_nearest_item(jakube_ext, data):
    dims, dataset, query_index, k = data
    index = jakube_ext.HammingIndex(dims)
    for item_id, vector in enumerate(dataset):
        index.add_item(item_id, vector)

    index.build(min(len(dataset) * 2, 64))

    query = dataset[query_index]
    ids_vector, _ = index.get_nns_by_vector(query, k, search_k=max(64, k * 16))
    ids_item, _ = index.get_nns_by_item(query_index, k, search_k=max(64, k * 16))

    assert ids_vector[0] == query_index
    assert ids_vector == ids_item