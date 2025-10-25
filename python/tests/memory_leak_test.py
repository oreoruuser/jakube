from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "index_name,dims,items",
    [
        ("EuclideanIndex", 4, 40),
        ("AngularIndex", 5, 40),
        ("ManhattanIndex", 4, 40),
        ("DotProductIndex", 4, 40),
        ("HammingIndex", 2, 40),
    ],
)
def test_repeated_construction_and_teardown(jakube_ext, index_name: str, dims: int, items: int):
    index_cls = getattr(jakube_ext, index_name)

    for cycle in range(6):
        index = index_cls(dims)
        for item in range(items):
            if index_name == "HammingIndex":
                vector = [item & 0xFFFFFFFFFFFFFFFF for _ in range(dims)]
            else:
                vector = [float(item + offset) for offset in range(dims)]
            index.add_item(item, vector)

        index.build(8)
        k = min(5, items)
        ids, _ = index.get_nns_by_item(0, k)
        assert len(ids) == k
        assert all(0 <= candidate < items for candidate in ids)
        index.unload()

    assert True
