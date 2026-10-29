"""Benchmark an index: build time, queries per second, recall@k, memory.

An index is any object with add(data), search(queries, k) -> indices and memoryBytes().
Run `python bench.py` to benchmark the exact baseline on random data, or
pass --base and --query .fvecs files (for example SIFT) to use real data.
"""
import argparse
import time

import numpy as np

from annindex.data import random_data, read_fvecs
from annindex.exact import exact_knn
from annindex.metrics import recall_at_k


class ExactIndex:
    def __init__(self, metric="euclidean"):
        self.metric = metric

    def add(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    def memoryBytes(self):
        return self.data.nbytes

    def search(self, queries, k):
        return exact_knn(self.data, queries, k, self.metric)


def benchmark(index, data, queries, truth, k):
    start = time.perf_counter()
    index.add(data)
    buildSeconds = time.perf_counter() - start

    start = time.perf_counter()
    found = index.search(queries, k)
    querySeconds = time.perf_counter() - start
    return {
        "build_s": buildSeconds,
        "qps": len(queries) / querySeconds,
        "recall": recall_at_k(found, truth, k),
        "memory_MB": index.memoryBytes() / 1e6,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--query")
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", default="euclidean")
    args = parser.parse_args()

    if args.base:
        data = read_fvecs(args.base, args.n)
        queries = read_fvecs(args.query, 1000)
    else:
        data = random_data(args.n, 128)
        queries = random_data(1000, 128, seed=1)

    truth = exact_knn(data, queries, args.k, args.metric)
    print(benchmark(ExactIndex(args.metric), data, queries, truth, args.k))
