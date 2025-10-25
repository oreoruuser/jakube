from __future__ import annotations

from typing import Sequence

import pytest


@pytest.mark.parametrize(
    "index_name,vectors,query",
    [
        ("EuclideanIndex", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], [0.1, 0.0]),
        ("AngularIndex", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [0.9, 0.1, 0.0]),
        ("ManhattanIndex", [[0.0, 0.0], [5.0, 5.0], [3.0, 4.0]], [0.1, 0.1]),
        ("DotProductIndex", [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], [0.75, 0.25]),
        ("HammingIndex", [[0x0, 0x0], [0xF, 0x0], [0x0, 0xF]], [0x0, 0x0]),
    ],
)
def test_save_load_roundtrip(
    jakube_ext,
    tmp_path,
    index_name: str,
    vectors: list[Sequence[float | int]],
    query: Sequence[float | int],
):
    index_cls = getattr(jakube_ext, index_name)
    index = index_cls(len(vectors[0]))
    for item_id, vector in enumerate(vectors):
        index.add_item(item_id, list(vector))

    index.build(10)
    baseline_ids, baseline_distances = index.get_nns_by_vector(query, len(vectors), search_k=200)

    path = tmp_path / f"{index_name}.jakube"
    index.save(str(path))
    assert path.exists()

    index.unload()

    loaded = index_cls(len(vectors[0]))
    loaded.load(str(path))
    assert loaded.n_items() == len(vectors)
    assert loaded.n_trees() == 10

    ids, distances = loaded.get_nns_by_vector(query, len(vectors), search_k=200)
    assert ids == baseline_ids
    if index_name == "HammingIndex":
        assert distances == baseline_distances
    else:
        assert pytest.approx(distances, rel=1e-6) == baseline_distances

    loaded.unload()


def test_on_disk_build_creates_mmap_artifact(jakube_ext, tmp_path):
    index = jakube_ext.AngularIndex(3)
    path = tmp_path / "angular.mmap"

    index.add_item(0, [1.0, 0.0, 0.0])
    index.add_item(1, [0.0, 1.0, 0.0])

    index.on_disk_build(str(path))
    index.build(6)

    assert path.exists()
    assert index.n_items() == 2
    assert index.n_trees() == 6

    index.unload()
