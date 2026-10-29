import numpy as np

from annindex.data import random_data
from annindex.exact import exact_knn
from annindex.metrics import recall_at_k


def slow_knn(data, query, k, metric):
    if metric == "cosine":
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        query = query / np.linalg.norm(query)
    return np.argsort(np.linalg.norm(data - query, axis=1))[:k]


def test_matches_naive_search():
    data, queries = random_data(500, 16), random_data(20, 16, seed=1)
    for metric in ("euclidean", "cosine"):
        found = exact_knn(data, queries, 5, metric)
        for q, row in zip(queries, found):
            assert list(row) == list(slow_knn(data, q, 5, metric))


def test_recall_of_perfect_and_empty_results():
    truth = np.array([[0, 1, 2], [3, 4, 5]])
    assert recall_at_k(truth, truth, 3) == 1.0
    assert recall_at_k(truth + 100, truth, 3) == 0.0
