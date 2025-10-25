from __future__ import annotations

import array

import pytest


def test_sequence_type_coercion(jakube_ext):
    index = jakube_ext.EuclideanIndex(3)
    index.add_item(0, (1, 2, 3))
    index.add_item(1, [4.0, 5.0, 6.0])
    index.add_item(2, array.array("f", [7.0, 8.0, 9.0]))

    index.build(6)

    assert index.get_item(0) == [1.0, 2.0, 3.0]

    ids, _ = index.get_nns_by_vector([0.0, 0.0, 0.0], 3)
    assert ids[0] == 0


def test_memoryview_and_tuple_input(jakube_ext):
    index = jakube_ext.EuclideanIndex(4)
    index.add_item(0, memoryview(array.array("f", [1.0, 2.0, 3.0, 4.0])))
    index.add_item(1, tuple(float(i) for i in range(4)))

    index.build(8)

    assert index.get_item(1) == [0.0, 1.0, 2.0, 3.0]


def test_dimension_mismatch_rejected(jakube_ext):
    index = jakube_ext.EuclideanIndex(2)
    with pytest.raises(ValueError):
        index.add_item(0, [1.0])


def test_hamming_requires_ints(jakube_ext):
    index = jakube_ext.HammingIndex(1)
    with pytest.raises((TypeError, RuntimeError)):
        index.add_item(0, [1.5])
