from __future__ import annotations

import random

import pytest


def test_manhattan_workflow(jakube_ext):
    index = jakube_ext.ManhattanIndex(2)
    index.add_item(0, [1.0, 1.0])
    index.add_item(1, [2.0, 2.0])
    index.add_item(2, [10.0, 10.0])
    index.add_item(3, [1.0, 2.0])

    index.build(8)

    assert index.get_distance(0, 1) == pytest.approx(2.0)
    assert index.get_distance(0, 3) == pytest.approx(1.0)
    assert index.get_distance(0, 2) == pytest.approx(18.0)

    ids, distances = index.get_nns_by_vector([0.0, 0.0], 4)
    assert ids == [0, 3, 1, 2]
    assert distances == pytest.approx([2.0, 3.0, 4.0, 20.0])


@pytest.mark.parametrize("dims", [3, 7])
def test_manhattan_rank_matches_l1_distance(jakube_ext, dims: int):
    rng = random.Random(dims)
    index = jakube_ext.ManhattanIndex(dims)
    vectors = []
    for item_id in range(50):
        vec = [rng.uniform(-2.0, 2.0) for _ in range(dims)]
        vectors.append(vec)
        index.add_item(item_id, vec)

    index.set_seed(2024)
    index.build(18)

    query = [rng.uniform(-2.0, 2.0) for _ in range(dims)]
    ids, distances = index.get_nns_by_vector(query, 12, search_k=240)

    expected = sorted(
        (
            (sum(abs(q - v) for q, v in zip(query, vec)), idx)
            for idx, vec in enumerate(vectors)
        ),
        key=lambda item: item[0],
    )[:12]
    expected_ids = [idx for _, idx in expected]

    assert ids == expected_ids
    assert all(first <= second + 1e-7 for first, second in zip(distances, distances[1:]))
