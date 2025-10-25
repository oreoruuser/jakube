from __future__ import annotations

import math


def test_example_usage(jakube_ext):
    index = jakube_ext.EuclideanIndex(3)
    for i in range(5):
        index.add_item(i, [float(i), float(i + 1), float(i + 2)])

    assert index.n_items() == 5
    index.build(10)
    assert index.n_trees() == 10

    ids, distances = index.get_nns_by_vector([0.0, 0.0, 0.0], 3, search_k=90)
    assert ids[0] == 0
    assert math.isclose(distances[0], math.sqrt(5.0), rel_tol=1e-6)

    ids_item, _ = index.get_nns_by_item(3, 3, search_k=90)
    assert ids_item[0] == 3
