from __future__ import annotations

import math
import random

import pytest


def test_angular_index_workflow(jakube_ext):
    index = jakube_ext.AngularIndex(2)
    index.add_item(0, [1.0, 0.0])
    index.add_item(1, [0.0, 1.0])
    index.add_item(2, [-1.0, 0.0])
    index.build(10)

    assert math.isclose(index.get_distance(0, 1), math.sqrt(2.0), rel_tol=1e-6)
    assert math.isclose(index.get_distance(0, 2), 2.0, rel_tol=1e-6)

    ids, _ = index.get_nns_by_vector([0.9, 0.1], 3)
    assert ids == [0, 1, 2]


@pytest.mark.parametrize("dims", [3, 8])
@pytest.mark.parametrize("seed", [7, 1337])
def test_angular_distance_monotonicity(jakube_ext, dims: int, seed: int):
    rng = random.Random(seed)
    index = jakube_ext.AngularIndex(dims)

    vectors = []
    for item_id in range(40):
        vec = [rng.uniform(-1.0, 1.0) for _ in range(dims)]
        vectors.append(vec)
        index.add_item(item_id, vec)

    index.set_seed(seed)
    index.build(20)

    query = [rng.uniform(-1.0, 1.0) for _ in range(dims)]
    ids, distances = index.get_nns_by_vector(query, 10, search_k=200)

    assert len(ids) == 10
    assert all(first <= second + 1e-7 for first, second in zip(distances, distances[1:]))

    # Cross-check distance ordering against direct angular distance computation.
    def angular_distance(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return math.inf
        cos_theta = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
        return math.sqrt(max(0.0, 2.0 - 2.0 * cos_theta))

    truth_ranking = sorted(
        ((angular_distance(query, vectors[i]), i) for i in range(len(vectors))),
        key=lambda item: item[0],
    )

    top_k = truth_ranking[:10]
    truth_distances = {candidate: dist for dist, candidate in top_k}
    allowed_set = {candidate for _, candidate in truth_ranking[:20]}

    hits = sum(1 for candidate in ids if candidate in allowed_set)
    assert hits >= 8  # Expect strong recall but tolerate a couple swaps

    tolerance = 1e-2
    for candidate, returned_distance in zip(ids, distances):
        computed = truth_distances.get(candidate)
        if computed is None:
            computed = angular_distance(query, vectors[candidate])
        assert returned_distance == pytest.approx(computed, rel=1e-6, abs=tolerance)
