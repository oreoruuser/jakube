from __future__ import annotations

import math

import pytest


def test_duplicate_ids_overwrite_vector(jakube_ext):
    index = jakube_ext.EuclideanIndex(3)
    index.add_item(5, [1.0, 1.0, 1.0])
    index.add_item(5, [9.0, 9.0, 9.0])

    index.build(4)
    assert index.get_item(5) == [9.0, 9.0, 9.0]


def test_unload_after_save_restores_empty_state(jakube_ext, tmp_path):
    index = jakube_ext.EuclideanIndex(2)
    for value in range(3):
        index.add_item(value, [float(value), float(value)])
    index.build(6)
    index.save(str(tmp_path / "persisted.jak"))

    index.unload()
    assert index.n_items() == 0
    assert index.n_trees() == 0


def test_invalid_dimension_raises(jakube_ext):
    index = jakube_ext.EuclideanIndex(3)
    with pytest.raises(ValueError):
        index.add_item(0, [1.0, 2.0])


def test_add_after_load_rejected(jakube_ext, tmp_path):
    index = jakube_ext.EuclideanIndex(2)
    index.add_item(0, [0.0, 0.0])
    index.build(4)
    path = tmp_path / "read_only.jak"
    index.save(str(path))

    reader = jakube_ext.EuclideanIndex(2)
    reader.load(str(path))
    with pytest.raises(ValueError):
        reader.add_item(1, [1.0, 1.0])


def test_distance_symmetry(jakube_ext):
    index = jakube_ext.EuclideanIndex(2)
    index.add_item(0, [0.0, 0.0])
    index.add_item(1, [1.0, 1.0])
    index.add_item(2, [2.0, 2.0])
    index.build(4)

    assert math.isclose(index.get_distance(0, 1), index.get_distance(1, 0), rel_tol=1e-6)
    assert math.isclose(index.get_distance(1, 2), index.get_distance(2, 1), rel_tol=1e-6)