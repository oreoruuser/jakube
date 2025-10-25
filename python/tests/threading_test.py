from __future__ import annotations

import concurrent.futures
import random


def test_concurrent_queries(jakube_ext):
    index = jakube_ext.EuclideanIndex(4)
    for item in range(20):
        index.add_item(item, [float(item + offset) for offset in range(4)])

    index.build(12)

    def run_query(item_id: int) -> int:
        ids, _ = index.get_nns_by_item(item_id, 3)
        return ids[0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_query, range(10)))

    assert results == list(range(10))


def test_mixed_parallel_queries(jakube_ext):
    dims = 6
    rng = random.Random(404)
    index = jakube_ext.EuclideanIndex(dims)
    dataset = []
    for item in range(64):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(dims)]
        dataset.append(vector)
        index.add_item(item, vector)

    index.build(18)

    def vector_query(idx: int) -> int:
        ids, _ = index.get_nns_by_vector(dataset[idx], 5, search_k=120)
        return ids[0]

    def item_query(idx: int) -> int:
        ids, _ = index.get_nns_by_item(idx, 5, search_k=120)
        return ids[0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        mixed = list(executor.map(vector_query, range(32))) + list(
            executor.map(item_query, range(32, 64))
        )

    assert mixed[:32] == list(range(32))
    assert mixed[32:] == list(range(32, 64))


def test_parallel_index_builds(jakube_ext):
    dims = 5

    def build_and_query(seed: int) -> list[int]:
        rng = random.Random(seed)
        index = jakube_ext.EuclideanIndex(dims)
        for item in range(30):
            index.add_item(item, [rng.uniform(-2.0, 2.0) for _ in range(dims)])
        index.set_seed(seed)
        index.build(16)
        ids, _ = index.get_nns_by_vector([0.0] * dims, 5, search_k=80)
        return ids

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(build_and_query, range(10, 14)))

    assert len({tuple(result) for result in results}) == len(results)
