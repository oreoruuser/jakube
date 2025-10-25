from __future__ import annotations

import math


def test_index_with_holes(jakube_ext):
    index = jakube_ext.EuclideanIndex(2)
    index.add_item(0, [1.0, 1.0])
    index.add_item(5, [5.0, 5.0])
    index.add_item(10, [10.0, 10.0])

    assert index.n_items() == 11

    index.build(8)
    assert index.n_trees() == 8

    assert index.get_item(5) == [5.0, 5.0]

    ids, distances = index.get_nns_by_vector([0.0, 0.0], 3)
    assert ids == [0, 5, 10]
    assert math.isclose(distances[0], math.sqrt(2.0), rel_tol=1e-6)
    assert math.isclose(distances[1], math.sqrt(50.0), rel_tol=1e-6)
    assert math.isclose(distances[2], math.sqrt(200.0), rel_tol=1e-6)

    index.unbuild()
    assert index.n_trees() == 0
