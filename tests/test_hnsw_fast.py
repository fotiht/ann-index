import numpy as np

from annindex.data import random_data
from annindex.exact import exact_knn
from annindex.hnsw_fast import HNSWFast
from annindex.metrics import recall_at_k


def test_finds_stored_points_exactly():
    data = random_data(300, 8)
    index = HNSWFast(M=8, efConstruction=50)
    index.add(data)
    assert list(index.search(data[:50], 1)[:, 0]) == list(range(50))


def test_recall_close_to_brute_force():
    data, queries = random_data(2000, 16), random_data(100, 16, seed=1)
    for metric in ("euclidean", "cosine"):
        index = HNSWFast(M=12, efConstruction=100, metric=metric)
        index.add(data)
        truth = exact_knn(data, queries, 10, metric)
        assert recall_at_k(index.search(queries, 10, ef=100), truth, 10) > 0.9


def test_links_respect_limits_and_are_valid():
    index = HNSWFast(M=6, efConstruction=40)
    index.add(random_data(500, 8))
    for layer in range(index.counts.shape[0]):
        limit = 2 * index.M if layer == 0 else index.M
        for node in range(500):
            links = index.links[layer, node, :index.counts[layer, node]]
            assert len(links) <= limit
            assert node not in links and len(set(links)) == len(links)
            assert all(0 <= n < 500 for n in links)


def test_threads_give_the_same_answers():
    data, queries = random_data(2000, 16), random_data(100, 16, seed=1)
    index = HNSWFast(M=12, efConstruction=100)
    index.add(data)
    assert (index.search(queries, 10, ef=50) == index.search(queries, 10, ef=50, threads=3)).all()


def test_threaded_build_keeps_recall_and_valid_links():
    data, queries = random_data(4000, 16), random_data(100, 16, seed=1)
    index = HNSWFast(M=12, efConstruction=100)
    index.add(data, threads=3)
    truth = exact_knn(data, queries, 10)
    assert recall_at_k(index.search(queries, 10, ef=100), truth, 10) > 0.9
    for layer in range(index.counts.shape[0]):
        limit = 2 * index.M if layer == 0 else index.M
        for node in range(len(data)):
            links = index.links[layer, node, :index.counts[layer, node]]
            assert len(links) <= limit and node not in links and len(set(links)) == len(links)


def test_build_does_not_depend_on_thread_count():
    data = random_data(3000, 16)
    two, four = HNSWFast(M=12, efConstruction=100), HNSWFast(M=12, efConstruction=100)
    two.add(data, threads=2)
    four.add(data, threads=4)
    assert (two.links == four.links).all() and (two.counts == four.counts).all()
