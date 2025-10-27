from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from jakube import EuclideanIndex


def test_threads_can_query_shared_index() -> None:
    n_items, dim = 2000, 10
    rng = np.random.default_rng(9001)
    data = rng.normal(size=(n_items, dim)).astype(np.float32)

    index = EuclideanIndex(dim)
    index.set_seed(2025)
    index.verbose(False)
    for item_id, vector in enumerate(data.tolist()):
        index.add_item(int(item_id), vector)
    index.build(20)

    def query(_: int) -> None:
        ids, distances = index.get_nns_by_item(1, 100)
        assert len(ids) == len(distances)
        assert ids[0] == 1

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(query, range(200)))
