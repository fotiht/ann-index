"""Recall vs queries per second for HNSW over several ef values, next to exact search."""
import argparse

from annindex.data import random_data, read_fvecs
from annindex.exact import exact_knn
from annindex.hnsw import HNSW
from bench import ExactIndex, benchmark


class HNSWAtEf:
    def __init__(self, index, ef):
        self.index, self.ef = index, ef

    def add(self, data):
        pass  # the index is already built

    def memoryBytes(self):
        return self.index.memoryBytes()

    def search(self, queries, k):
        return self.index.search(queries, k, self.ef)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--query")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--efc", type=int, default=100)
    args = parser.parse_args()

    if args.base:
        data, queries = read_fvecs(args.base, args.n), read_fvecs(args.query, 500)
    else:
        data, queries = random_data(args.n, 128), random_data(500, 128, seed=1)
    truth = exact_knn(data, queries, args.k)

    exact = benchmark(ExactIndex(), data, queries, truth, args.k)
    print(f"exact          recall {exact['recall']:.3f}  qps {exact['qps']:.0f}")

    index = HNSW(M=args.M, efConstruction=args.efc)
    index.add(data)
    for ef in (10, 20, 50, 100, 200, 400):
        r = benchmark(HNSWAtEf(index, ef), data, queries, truth, args.k)
        print(f"hnsw ef={ef:<4d}  recall {r['recall']:.3f}  qps {r['qps']:.0f}")
