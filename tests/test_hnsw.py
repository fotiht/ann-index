import numpy as np

from annindex.data import random_data
from annindex.exact import exact_knn
from annindex.hnsw import HNSW
from annindex.metrics import recall_at_k


def test_finds_stored_points_exactly():
    data = random_data(300, 8)
    index = HNSW(M=8, efConstruction=50)
    index.add(data)
    assert list(index.search(data[:50], 1)[:, 0]) == list(range(50))


def test_recall_close_to_brute_force():
    data, queries = random_data(2000, 16), random_data(100, 16, seed=1)
    for metric in ("euclidean", "cosine"):
        index = HNSW(M=12, efConstruction=100, metric=metric)
        index.add(data)
        truth = exact_knn(data, queries, 10, metric)
        assert recall_at_k(index.search(queries, 10, ef=100), truth, 10) > 0.9


def test_links_respect_limits_and_are_valid():
    index = HNSW(M=6, efConstruction=40)
    index.add(random_data(500, 8))
    for l, layer in enumerate(index.layers):
        limit = index.maxLinks0 if l == 0 else index.M
        for node, links in layer.items():
            assert len(links) <= limit
            assert node not in links and all(n in layer for n in links)
