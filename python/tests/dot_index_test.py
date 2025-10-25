from __future__ import annotations

import random

import pytest


def test_dot_product_index_basic(jakube_ext):
    index = jakube_ext.DotProductIndex(2)
    index.add_item(0, [1.0, 1.0])
    index.add_item(1, [2.0, 0.0])
    index.add_item(2, [-1.0, 0.0])
    index.build(10)

    assert index.get_distance(0, 1) == -2.0
    ids, distances = index.get_nns_by_vector([3.0, 0.0], 3)
    assert ids == [1, 0, 2]
    assert distances == [-6.0, -3.0, 3.0]


@pytest.mark.parametrize("dims", [4, 10])
@pytest.mark.parametrize("seed", [13, 101])
def test_dot_product_rank_matches_dot_score(jakube_ext, dims: int, seed: int):
    rng = random.Random(seed)
    index = jakube_ext.DotProductIndex(dims)
    vectors = []
    for item_id in range(50):
        vec = [rng.uniform(-1.5, 1.5) for _ in range(dims)]
        vectors.append(vec)
        index.add_item(item_id, vec)

    index.set_seed(seed)
    index.build(20)

    query = [rng.uniform(-1.5, 1.5) for _ in range(dims)]
    ids, distances = index.get_nns_by_vector(query, 10, search_k=300)

    dot_scores = sorted(
        ((-sum(q * v for q, v in zip(query, vec)), idx) for idx, vec in enumerate(vectors)),
        key=lambda item: item[0],
    )[:10]
    expected_ids = [idx for _, idx in dot_scores]

    assert ids == expected_ids
    assert all(first <= second + 1e-7 for first, second in zip(distances, distances[1:]))


def test_dot_product_unbuild_and_rebuild(jakube_ext):
    index = jakube_ext.DotProductIndex(3)
    vectors = {
        0: [1.0, 0.0, 0.0],
        1: [0.0, 1.0, 0.0],
        2: [0.0, 0.0, 1.0],
    }
    for item_id, vec in vectors.items():
        index.add_item(item_id, vec)

    index.build(6)
    assert index.n_trees() == 6

    index.unbuild()
    assert index.n_trees() == 0

    index.build(4)
    ids, _ = index.get_nns_by_vector([1.0, 0.0, 0.0], 3, search_k=30)
    assert ids[0] == 0
