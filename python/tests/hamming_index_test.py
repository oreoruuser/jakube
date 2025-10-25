from __future__ import annotations


def test_hamming_distance_basic(jakube_ext):
    index = jakube_ext.HammingIndex(1)
    index.add_item(0, [0x1])
    index.add_item(1, [0xF])
    index.build(8)

    assert index.get_distance(0, 1) == 3
    ids, distances = index.get_nns_by_vector([0x0], 2)
    assert ids == [0, 1]
    assert distances == [1, 4]


def test_hamming_multiple_words(jakube_ext):
    index = jakube_ext.HammingIndex(2)
    vectors = {
        0: [0xFFFFFFFFFFFFFFFF, 0x0],
        1: [0x0, 0xFFFFFFFFFFFFFFFF],
        2: [0xAAAAAAAAAAAAAAAA, 0x5555555555555555],
        3: [0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF],
    }
    for item_id, vec in vectors.items():
        index.add_item(item_id, vec)

    index.build(16)

    ids, distances = index.get_nns_by_vector([0xFFFFFFFFFFFFFFFF, 0x0], 4, search_k=128)
    assert ids[0] == 0
    assert distances[0] == 0
    assert sorted(ids) == [0, 1, 2, 3]


def test_hamming_get_item_roundtrip(jakube_ext):
    index = jakube_ext.HammingIndex(1)
    index.add_item(5, [0x1234])
    index.build(4)

    assert index.get_item(5) == [0x1234]
    ids, _ = index.get_nns_by_item(5, 1)
    assert ids == [5]
