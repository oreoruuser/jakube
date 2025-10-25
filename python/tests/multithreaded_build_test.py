from __future__ import annotations

import random

import pytest


@pytest.mark.parametrize("n_threads", [1, 2, 4, 8])
def test_build_with_explicit_thread_count(jakube_ext, n_threads: int):
    index = jakube_ext.EuclideanIndex(3)
    for item in range(6):
        index.add_item(item, [float(item), float(item + 1), float(item + 2)])

    index.build(12, n_threads=n_threads)
    assert index.n_trees() == 12

    ids, _ = index.get_nns_by_vector([0.0, 0.0, 0.0], 3)
    assert ids[0] == 0


def test_multithreaded_build_matches_single_threaded_order(jakube_ext):
    rng = random.Random(321)
    dims = 6
    data = [[rng.uniform(-1.0, 1.0) for _ in range(dims)] for _ in range(80)]

    single = jakube_ext.EuclideanIndex(dims)
    multi = jakube_ext.EuclideanIndex(dims)

    single.set_seed(777)
    multi.set_seed(777)
    for idx, vector in enumerate(data):
        single.add_item(idx, vector)
        multi.add_item(idx, vector)

    single.build(24, n_threads=1)
    multi.build(24, n_threads=8)

    query = [rng.uniform(-1.0, 1.0) for _ in range(dims)]
    ids_single, _ = single.get_nns_by_vector(query, 20, search_k=400)
    ids_multi, _ = multi.get_nns_by_vector(query, 20, search_k=400)

    assert ids_single == ids_multi
