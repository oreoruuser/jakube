from __future__ import annotations

import numpy as np

from jakube import AngularIndex


def test_seeding_produces_deterministic_results() -> None:
    dim = 10
    rng = np.random.default_rng(2024)
    data = rng.random((1000, dim)).astype(np.float32)
    queries = rng.random((50, dim)).astype(np.float32)

    indexes = []
    for _ in range(2):
        index = AngularIndex(dim)
        index.set_seed(42)
        index.verbose(False)
        for item_id, vector in enumerate(data.tolist()):
            index.add_item(int(item_id), vector)
        index.build(20)
        indexes.append(index)

    for vector in queries.tolist():
        ids_a, _ = indexes[0].get_nns_by_vector(vector, 100)
        ids_b, _ = indexes[1].get_nns_by_vector(vector, 100)
        assert list(ids_a) == list(ids_b)
