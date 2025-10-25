from __future__ import annotations

import math
import random
from typing import Callable, Iterable, Sequence

import pytest


def _generate_float_dataset(rng: random.Random, num_items: int, dims: int, *, centered: bool = False) -> list[list[float]]:
    scale = 1.0 if not centered else 2.0
    offset = 0.0 if not centered else -1.0
    return [[rng.random() * scale + offset for _ in range(dims)] for _ in range(num_items)]


def _euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.dist(a, b)


def _manhattan_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def _angular_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return math.inf
    cos_theta = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
    return math.sqrt(max(0.0, 2.0 - 2.0 * cos_theta))


def _dot_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return -sum(x * y for x, y in zip(a, b))


def _recall(
    index,
    dataset: list[list[float]],
    queries: Iterable[int],
    k: int,
    scorer: Callable[[Sequence[float], Sequence[float]], float],
    search_k: int,
) -> float:
    query_list = list(queries)
    hits = 0
    for idx in query_list:
        query = dataset[idx]
        truth = {
            candidate_idx
            for _, candidate_idx in sorted(
                ((scorer(query, candidate), cid) for cid, candidate in enumerate(dataset)),
                key=lambda item: item[0],
            )[:k]
        }
        ids, _ = index.get_nns_by_vector(query, k, search_k=search_k)
        hits += sum(1 for item in ids if item in truth)
    return hits / (k * len(query_list))


@pytest.mark.parametrize(
    "index_name,dims,num_items,k,search_multiplier,scorer,threshold,centered",
    [
        ("EuclideanIndex", 12, 320, 15, 40, _euclidean_distance, 0.85, False),
        ("AngularIndex", 10, 320, 12, 45, _angular_distance, 0.80, True),
        ("ManhattanIndex", 10, 320, 12, 45, _manhattan_distance, 0.80, False),
        ("DotProductIndex", 12, 320, 12, 60, _dot_distance, 0.75, True),
    ],
)
def test_accuracy_recall_multi_metric(
    jakube_ext,
    index_name: str,
    dims: int,
    num_items: int,
    k: int,
    search_multiplier: int,
    scorer: Callable[[Sequence[float], Sequence[float]], float],
    threshold: float,
    centered: bool,
):
    rng = random.Random(42)
    data = _generate_float_dataset(rng, num_items, dims, centered=centered)

    index_cls = getattr(jakube_ext, index_name)
    index = index_cls(dims)
    index.set_seed(1337)
    for item_id, vector in enumerate(data):
        index.add_item(item_id, vector)

    index.build(30)

    query_indices = list(range(min(25, num_items // 4)))
    search_k = max(k * search_multiplier, k + 1)
    recall = _recall(index, data, query_indices, k, scorer, search_k)
    assert recall >= threshold


def _popcount(value: int) -> int:
    return value.bit_count()


def _hamming_distance(a: Sequence[int], b: Sequence[int]) -> int:
    return sum(_popcount(x ^ y) for x, y in zip(a, b))


def test_accuracy_recall_hamming(jakube_ext):
    dims = 2
    num_items = 256
    rng = random.Random(99)
    data = [[rng.getrandbits(64) for _ in range(dims)] for _ in range(num_items)]

    index = jakube_ext.HammingIndex(dims)
    index.set_seed(2718)
    for item_id, vector in enumerate(data):
        index.add_item(item_id, vector)

    index.build(32)

    k = 16
    search_k = k * 32
    hits = 0
    queries = list(range(32))
    for idx in queries:
        query = data[idx]
        truth = {
            candidate_idx
            for _, candidate_idx in sorted(
                ((
                    _hamming_distance(query, candidate),
                    cid,
                ) for cid, candidate in enumerate(data)),
                key=lambda item: item[0],
            )[:k]
        }
        ids, _ = index.get_nns_by_vector(query, k, search_k=search_k)
        hits += sum(1 for item in ids if item in truth)

    recall = hits / (k * len(queries))
    assert recall >= 0.75
