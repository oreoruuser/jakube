from __future__ import annotations

import math
import random

import pytest


def test_euclidean_workflow(jakube_ext):
    index = jakube_ext.EuclideanIndex(2)
    index.add_item(0, [1.0, 1.0])
    index.add_item(1, [2.0, 2.0])
    index.add_item(2, [10.0, 10.0])
    index.add_item(3, [1.0, 2.0])

    assert index.n_items() == 4
    index.build(10)
    assert index.n_trees() == 10

    ids, distances = index.get_nns_by_vector([0.0, 0.0], 4)
    assert ids == [0, 3, 1, 2]
    assert math.isclose(distances[0], math.sqrt(2.0), rel_tol=1e-6)


@pytest.mark.parametrize("dims", [3, 9])
@pytest.mark.parametrize("seed", [0, 1234])
def test_euclidean_query_rank_matches_distance(jakube_ext, dims: int, seed: int):
    rng = random.Random(seed)
    index = jakube_ext.EuclideanIndex(dims)
    vectors = []
    for item_id in range(60):
        vec = [rng.uniform(-1.0, 2.0) for _ in range(dims)]
        vectors.append(vec)
        index.add_item(item_id, vec)

    index.set_seed(seed)
    index.build(25)

    query = [rng.uniform(-1.0, 2.0) for _ in range(dims)]
    ids, distances = index.get_nns_by_vector(query, 15, search_k=400)

    expected = sorted(
        ((math.dist(query, vec), idx) for idx, vec in enumerate(vectors)),
        key=lambda item: item[0],
    )[:15]
    expected_ids = [idx for _, idx in expected]

    assert ids == expected_ids
    assert all(first <= second + 1e-7 for first, second in zip(distances, distances[1:]))


def test_euclidean_get_nns_by_item_matches_vector(jakube_ext):
    rng = random.Random(99)
    dims = 5
    index = jakube_ext.EuclideanIndex(dims)
    for item_id in range(20):
        index.add_item(item_id, [rng.random() for _ in range(dims)])

    index.build(12)
    ids_vector, distances_vector = index.get_nns_by_vector(
        index.get_item(7), 6, search_k=120
    )
    ids_item, distances_item = index.get_nns_by_item(7, 6, search_k=120)

    assert ids_vector == ids_item
    assert pytest.approx(distances_vector, rel=1e-6) == distances_item
