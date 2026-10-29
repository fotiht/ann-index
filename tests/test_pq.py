import numpy as np

from annindex.data import random_data
from annindex.exact import exact_knn
from annindex.hnsw_fast import HNSWFast
from annindex.hnsw_pq import CompressedHNSW
from annindex.metrics import recall_at_k
from annindex.pq import PQFlatIndex, ProductQuantizer


def test_codes_are_small_and_tables_match_decoded_distances():
    data = random_data(2000, 16)
    pq = ProductQuantizer(pieces=4, centroids=32, iterations=5).fit(data)
    codes = pq.encode(data)
    assert codes.shape == (2000, 4) and codes.dtype == np.uint8
    query = random_data(1, 16, seed=1)
    table = pq.distanceTables(query)[0]
    decoded = np.hstack([pq.codebooks[p][codes[:, p]] for p in range(4)])
    expected = ((decoded - query) ** 2).sum(axis=1)
    approx = table[np.arange(4), codes].sum(axis=1)
    assert np.allclose(approx, expected, atol=1e-3)


def test_pq_flat_finds_most_neighbours():
    data, queries = random_data(3000, 16), random_data(50, 16, seed=1)
    index = PQFlatIndex(pieces=8, centroids=64, trainSize=3000)
    index.add(data)
    truth = exact_knn(data, queries, 10)
    assert recall_at_k(index.search(queries, 10), truth, 10) > 0.3
    assert index.memoryBytes() < data.nbytes / 2


def test_hnsw_search_on_codes_with_and_without_rerank():
    data, queries = random_data(3000, 16), random_data(50, 16, seed=1)
    index = HNSWFast(M=12, efConstruction=100)
    index.add(data)
    compressed = CompressedHNSW(index, ProductQuantizer(pieces=8, centroids=64, iterations=5).fit(data))
    truth = exact_knn(data, queries, 10)
    plain = recall_at_k(compressed.search(queries, 10, ef=100), truth, 10)
    reranked = recall_at_k(compressed.search(queries, 10, ef=100, rerank=True), truth, 10)
    assert plain > 0.3 and reranked > plain
    assert (compressed.search(queries, 10, ef=100, threads=3) == compressed.search(queries, 10, ef=100)).all()
    assert compressed.memoryBytes() < index.memoryBytes()
