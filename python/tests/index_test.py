from __future__ import annotations

import pytest


def test_index_lifecycle(jakube_ext):
    index = jakube_ext.EuclideanIndex(2)
    assert index.dims() == 2

    index.add_item(0, [1.0, 1.0])
    assert index.n_items() == 1

    index.build(6)
    assert index.n_trees() == 6

    ids, _ = index.get_nns_by_item(0, 1)
    assert ids == [0]

    index.unload()
    assert index.n_items() == 0
    assert index.n_trees() == 0

    reused = jakube_ext.EuclideanIndex(2)
    reused.verbose(True)
    reused.add_item(0, [10.0, 10.0])
    reused.add_item(1, [11.0, 11.0])
    assert reused.n_items() == 2

    reused.build(5)
    assert reused.n_trees() == 5

    ids, _ = reused.get_nns_by_vector([9.0, 9.0], 1)
    assert ids[0] == 0


def test_index_build_twice_errors(jakube_ext):
    index = jakube_ext.EuclideanIndex(3)
    index.add_item(0, [0.0, 0.0, 0.0])
    index.add_item(1, [1.0, 1.0, 1.0])

    index.build(4)
    with pytest.raises(ValueError):
        index.build(4)
