from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from jakube import AngularIndex, DotProductIndex


Dataset = Dict[str, object]


def _angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / denom))
    return np.sqrt(max(0.0, 2.0 - 2.0 * cosine))


def _expected_neighbors(vectors: np.ndarray, query: np.ndarray, metric: str, top_k: int) -> list[int]:
    distances: list[Tuple[float, int]] = []
    for item_id, vector in enumerate(vectors):
        if metric == "angular":
            dist = _angular_distance(vector, query)
        elif metric == "dot":
            dist = -float(np.dot(vector, query))
        else:
            dist = float(np.linalg.norm(vector - query))
        distances.append((dist, item_id))
    distances.sort()
    return [item for _, item in distances[:top_k]]


def _recall(truth: Sequence[int], result: Sequence[int]) -> float:
    overlap = len(set(truth).intersection(result[: len(truth)]))
    return overlap / len(truth)


DATASETS: Dict[str, Dataset] = {
    "glove_25": {
        "index_cls": AngularIndex,
        "metric": "angular",
        "dim": 3,
        "train": np.array(
            [
                [1.0, 0.0, 0.0],
                [0.95, 0.05, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        ),
        "queries": np.array(
            [
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "threshold": 69.0,
    },
    "nytimes_16": {
        "index_cls": AngularIndex,
        "metric": "angular",
        "dim": 4,
        "train": np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.5, 0.5, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "queries": np.array(
            [
                [0.6, 0.4, 0.0, 0.0],
                [0.0, 0.2, 0.8, 0.0],
                [0.5, 0.5, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "threshold": 80.0,
    },
    "lastfm_dot": {
        "index_cls": DotProductIndex,
        "metric": "dot",
        "dim": 3,
        "train": np.array(
            [
                [1.0, 1.0, 0.0],
                [2.0, 0.5, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.5, 1.5],
                [1.1, 0.9, 0.0],
            ],
            dtype=np.float32,
        ),
        "queries": np.array(
            [
                [3.0, 1.0, 0.0],
                [0.5, 2.5, 2.5],
            ],
            dtype=np.float32,
        ),
        "threshold": 60.0,
    },
    "lastfm_angular": {
        "index_cls": AngularIndex,
        "metric": "angular",
        "dim": 3,
        "train": np.array(
            [
                [1.0, 0.0, 0.0, 10.0],
                [0.7, 0.7, 0.0, 20.0],
                [0.0, 1.0, 0.0, 30.0],
                [0.0, 0.0, 1.0, 40.0],
                [0.8, 0.2, 0.0, 50.0],
            ],
            dtype=np.float32,
        ),
        "queries": np.array(
            [
                [0.85, 0.15, 0.0, 5.0],
                [0.05, 0.95, 0.0, 7.0],
            ],
            dtype=np.float32,
        ),
        "threshold": 60.0,
    },
}


def _run_case(name: str) -> float:
    spec = DATASETS[name]
    dim = int(spec["dim"])
    train = np.asarray(spec["train"], dtype=np.float32)
    queries = np.asarray(spec["queries"], dtype=np.float32)
    index_cls = spec["index_cls"]
    metric = str(spec["metric"])
    index = index_cls(dim)
    if hasattr(index, "set_seed"):
        index.set_seed(1337)
    if hasattr(index, "verbose"):
        index.verbose(False)
    for item_id, vector in enumerate(train[:, :dim]):
        index.add_item(int(item_id), vector.tolist())
    index.build(10)

    top_k = min(10, train.shape[0])
    recalls = []
    for vector in queries[:, :dim]:
        expected = _expected_neighbors(train[:, :dim], vector, metric, top_k)
        result_ids, _ = index.get_nns_by_vector(vector.tolist(), top_k, 10000)
        recalls.append(_recall(expected, list(result_ids)))

    accuracy = float(np.mean(recalls) * 100.0)
    return accuracy


def test_glove_25() -> None:
    accuracy = _run_case("glove_25")
    assert accuracy >= float(DATASETS["glove_25"]["threshold"]) - 1.0


def test_nytimes_16() -> None:
    accuracy = _run_case("nytimes_16")
    assert accuracy >= float(DATASETS["nytimes_16"]["threshold"]) - 1.0


def test_lastfm_dot() -> None:
    accuracy = _run_case("lastfm_dot")
    assert accuracy >= float(DATASETS["lastfm_dot"]["threshold"]) - 1.0


def test_lastfm_angular() -> None:
    accuracy = _run_case("lastfm_angular")
    assert accuracy >= float(DATASETS["lastfm_angular"]["threshold"]) - 1.0
