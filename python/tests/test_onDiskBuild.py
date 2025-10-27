from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pytest

from jakube import EuclideanIndex


def add_items(index: EuclideanIndex, vectors: Iterable[Sequence[float]]) -> None:
    for item_id, vector in enumerate(vectors):
        index.add_item(int(item_id), list(map(float, vector)))


def check_nns(index: EuclideanIndex) -> None:
    ids, _ = index.get_nns_by_vector([4.0, 4.0], 3)
    assert list(ids) == [2, 1, 0]
    ids, _ = index.get_nns_by_vector([1.0, 1.0], 3)
    assert list(ids) == [0, 1, 2]
    ids, _ = index.get_nns_by_vector([4.0, 2.0], 3)
    assert list(ids) == [1, 2, 0]


@pytest.fixture(scope="module", autouse=True)
def remove_existing_file() -> None:
    path = Path("on_disk.ann")
    if path.exists():
        path.unlink()


def test_on_disk_roundtrip(tmp_path: Path) -> None:
    vectors = ([2.0, 2.0], [3.0, 2.0], [3.0, 3.0])
    path = tmp_path / "on_disk.index"

    index = EuclideanIndex(2)
    index.verbose(False)
    add_items(index, vectors)
    index.build(15)
    check_nns(index)
    index.save(str(path))

    loaded = EuclideanIndex(2)
    loaded.load(str(path))
    check_nns(loaded)

    reloaded = EuclideanIndex(2)
    reloaded.load(str(path))
    check_nns(reloaded)

    path.unlink()
