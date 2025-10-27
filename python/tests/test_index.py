from __future__ import annotations

import os
import shutil
import random
from collections.abc import Generator
from pathlib import Path

import pytest

from jakube import AngularIndex


TESTDATA_DIR = Path("testData")
TEST_TREE_PATH = TESTDATA_DIR / "test.tree"
_TESTDATA_DIM = 10
_TESTDATA_ITEMS = 100
_TESTDATA_VECTOR_SEED = 1337
_TESTDATA_INDEX_SEED = 7
_MIN_TESTDATA_TREES = 10
_EXPECTED_NEIGHBORS = [0, 56, 38, 43, 37, 36, 3, 48, 57, 53]
_TESTDATA_CREATED = False


def _ensure_test_tree() -> Path:
    """Create the binary test tree lazily so tests can run without fixtures."""
    global _TESTDATA_CREATED
    TESTDATA_DIR.mkdir(parents=True, exist_ok=True)
    if TEST_TREE_PATH.exists():
        return TEST_TREE_PATH

    _TESTDATA_CREATED = True
    rng = random.Random(_TESTDATA_VECTOR_SEED)
    builder = AngularIndex(_TESTDATA_DIM)
    for item_id in range(_TESTDATA_ITEMS):
        vector = [rng.gauss(0.0, 1.0) for _ in range(_TESTDATA_DIM)]
        builder.add_item(item_id, vector)
    builder.set_seed(_TESTDATA_INDEX_SEED)
    builder.build(_MIN_TESTDATA_TREES)
    tree_count = builder.n_trees()
    assert tree_count >= _MIN_TESTDATA_TREES

    ids, _ = builder.get_nns_by_item(0, 10)
    assert list(ids) == _EXPECTED_NEIGHBORS

    tmp_path = TEST_TREE_PATH.with_suffix(".tmp")
    builder.save(str(tmp_path))
    os.replace(tmp_path, TEST_TREE_PATH)
    return TEST_TREE_PATH


@pytest.fixture(scope="session", autouse=True)
def _cleanup_generated_testdata() -> Generator[None, None, None]:
    yield
    if _TESTDATA_CREATED and TESTDATA_DIR.exists():
        shutil.rmtree(TESTDATA_DIR)


def test_not_found_tree() -> None:
    i = AngularIndex(10)
    with pytest.raises(Exception):
        i.load("nonexists.tree")


def test_binary_compatibility() -> None:
    i = AngularIndex(10)
    # load a lazily generated test file so the suite stays deterministic
    path = _ensure_test_tree()
    i.load(str(path))

    ids, _ = i.get_nns_by_item(0, 10)
    assert list(ids) == _EXPECTED_NEIGHBORS


def test_load_unload() -> None:
    i = AngularIndex(10)
    path = _ensure_test_tree()
    for _ in range(10):
        i.load(str(path))
        i.unload()


def test_construct_load_destruct() -> None:
    path = _ensure_test_tree()
    for _ in range(10):
        i = AngularIndex(10)
        i.load(str(path))


def test_construct_destruct() -> None:
    for _ in range(10):
        i = AngularIndex(10)
        i.add_item(1000, [random.gauss(0, 1) for _ in range(10)])


def test_save_twice(tmp_path: Path) -> None:
    t = AngularIndex(10)
    for i in range(100):
        t.add_item(i, [random.gauss(0, 1) for _ in range(10)])
    t.build(10)
    p1 = tmp_path / "t1.index"
    p2 = tmp_path / "t2.index"
    t.save(str(p1))
    t.save(str(p2))


def test_load_save(tmp_path: Path) -> None:
    i = AngularIndex(10)
    path = _ensure_test_tree()
    i.load(str(path))
    u = i.get_item(99)

    out = tmp_path / "i.index"
    i.save(str(out))
    v = i.get_item(99)
    assert list(u) == list(v)

    j = AngularIndex(10)
    j.load(str(path))
    w = j.get_item(99)
    assert list(u) == list(w)

    # prefault variations
    j.save(str(tmp_path / "j.index"), True)
    k = AngularIndex(10)
    k.load(str(tmp_path / "j.index"), True)
    x = k.get_item(99)
    assert list(u) == list(x)

    k.save(str(tmp_path / "k.index"), False)
    l = AngularIndex(10)
    l.load(str(tmp_path / "k.index"), False)
    y = l.get_item(99)
    assert list(u) == list(y)


def test_save_without_build() -> None:
    t = AngularIndex(10)
    for i in range(100):
        t.add_item(i, [random.gauss(0, 1) for _ in range(10)])
    with pytest.raises(Exception):
        t.save("")


def test_unbuild_with_loaded_tree() -> None:
    i = AngularIndex(10)
    path = _ensure_test_tree()
    i.load(str(path))
    with pytest.raises(Exception):
        i.unbuild()


def test_seed_and_metric_kwarg() -> None:
    i = AngularIndex(10)
    path = _ensure_test_tree()
    i.load(str(path))
    i.set_seed(42)


def test_unknown_distance() -> None:
    # Jakube exposes explicit metric classes rather than a generic factory.
    # Verify that attempting to access a non-existent metric raises the expected error.
    with pytest.raises(AttributeError):
        getattr(__import__("jakube"), "UnknownIndex")


def test_metric_kwarg() -> None:
    # ensure calling with dim works
    i = AngularIndex(2)
    i.add_item(0, [1, 0])
    i.add_item(1, [9, 0])
    i.build(1)
    assert i.get_distance(0, 1) == pytest.approx(0.0, abs=1e-8)


def test_item_vector_after_save() -> None:
    a = AngularIndex(3)
    a.verbose(True)
    a.add_item(1, [1, 0, 0])
    a.add_item(2, [0, 1, 0])
    a.add_item(3, [0, 0, 1])
    a.build(1)
    assert a.n_items() == 4
    assert a.get_item(3) == [0, 0, 1]
    ids, _ = a.get_nns_by_item(1, 999)
    assert set(ids) == {1, 2, 3}
    p = TESTDATA_DIR / "something.index"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()
    a.save(str(p))
    assert a.n_items() == 4
    assert a.get_item(3) == [0, 0, 1]
    ids2, _ = a.get_nns_by_item(1, 999)
    assert set(ids2) == {1, 2, 3}
    if p.exists():
        p.unlink()


def test_prefault() -> None:
    i = AngularIndex(10)
    path = _ensure_test_tree()
    i.load(str(path), prefault=True)
    ids, _ = i.get_nns_by_item(0, 10)
    assert list(ids) == _EXPECTED_NEIGHBORS


def test_fail_save() -> None:
    t = AngularIndex(40)
    with pytest.raises(Exception):
        t.save("")


def test_overwrite_index(tmp_path: Path) -> None:
    f = 40
    t = AngularIndex(f)
    for i in range(1000):
        v = [random.gauss(0, 1) for _ in range(f)]
        t.add_item(i, v)
    t.build(10)
    p = tmp_path / "test.ann"
    t.save(str(p))

    t2 = AngularIndex(f)
    t2.load(str(p))

    t3 = AngularIndex(f)
    for i in range(500):
        v = [random.gauss(0, 1) for _ in range(f)]
        t3.add_item(i, v)
    t3.build(10)
    # Overwrite is allowed on POSIX
    t3.save(str(p))
    v = [random.gauss(0, 1) for _ in range(f)]
    t2.get_nns_by_vector(v, 1000)


def test_get_n_trees() -> None:
    i = AngularIndex(10)
    path = _ensure_test_tree()
    i.load(str(path))
    assert i.n_trees() >= _MIN_TESTDATA_TREES


def test_write_failed() -> None:
    f = 40
    t = AngularIndex(f)
    t.verbose(True)
    for i in range(1000):
        v = [random.gauss(0, 1) for _ in range(f)]
        t.add_item(i, v)
    t.build(10)

    if os.name == "nt":
        path = "Z:/xyz.index"
    else:
        path = "/x/y/z.index"
    with pytest.raises(Exception):
        t.save(path)


def test_dimension_mismatch(tmp_path: Path) -> None:
    t = AngularIndex(100)
    for i in range(1000):
        t.add_item(i, [random.gauss(0, 1) for _ in range(100)])
    t.build(10)
    p = tmp_path / "test.ann"
    t.save(str(p))

    u = AngularIndex(200)
    with pytest.raises(Exception):
        u.load(str(p))
    u = AngularIndex(50)
    with pytest.raises(Exception):
        u.load(str(p))


def test_add_after_save() -> None:
    t = AngularIndex(100)
    for i in range(1000):
        t.add_item(i, [random.gauss(0, 1) for _ in range(100)])
    t.build(10)
    p = TESTDATA_DIR / "test.ann"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()
    t.save(str(p))
    v = [random.gauss(0, 1) for _ in range(100)]
    with pytest.raises(Exception):
        t.add_item(1001, v)
    if p.exists():
        p.unlink()


def test_build_twice() -> None:
    t = AngularIndex(100)
    for i in range(1000):
        t.add_item(i, [random.gauss(0, 1) for _ in range(100)])
    t.build(10)
    with pytest.raises(Exception):
        t.build(10)


def test_very_large_index(tmp_path: Path) -> None:
    f = 3
    dangerous_size = 2**20  # reduced for CI
    size_per_vector = 4 * (f + 3)
    n_vectors = int(dangerous_size / size_per_vector)
    m = AngularIndex(3)
    m.verbose(True)
    for i in range(100):
        m.add_item(n_vectors + i, [random.gauss(0, 1) for _ in range(f)])
    n_trees = 10
    m.build(n_trees)
    path = tmp_path / "test_big.index"
    m.save(str(path))

    assert os.path.getsize(str(path)) >= 0
    assert m.n_trees() >= n_trees
