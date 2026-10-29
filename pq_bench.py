"""HNSW with product quantization on SIFT: recall, speed and memory.

Rows: raw vectors, distances from PQ codes only, and PQ codes followed by an exact re-score of
the ef candidates (that last variant still needs the raw vectors, for example on disk).
"""
import argparse
import time

import numpy as np

from annindex.data import read_fvecs
from annindex.exact import exact_knn
from annindex.hnsw_fast import HNSWFast
from annindex.hnsw_pq import CompressedHNSW
from annindex.metrics import recall_at_k
from annindex.pq import PQFlatIndex, ProductQuantizer


def measure(search, queries, truth, k):
    search(queries[:8])  # compile before timing
    start = time.perf_counter()
    found = search(queries)
    return recall_at_k(found, truth, k), len(queries) / (time.perf_counter() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/sift/sift_base.fvecs")
    parser.add_argument("--query", default="data/sift/sift_query.fvecs")
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--pieces", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--threads", type=int, default=4, help="threads for building only")
    args = parser.parse_args()

    data, queries = read_fvecs(args.base, args.n), read_fvecs(args.query, 1000)
    truth = exact_knn(data, queries, args.k)
    print(f"n = {args.n}, raw vectors {data.nbytes / 1e6:.1f} MB")

    print("\nPQ scan over all codes (no graph)")
    for pieces in args.pieces:
        flat = PQFlatIndex(pieces)
        flat.add(data)
        recall, qps = measure(lambda q: flat.search(q, args.k), queries, truth, args.k)
        print(f"  {pieces} pieces: recall {recall:.3f}, {qps:.0f} qps, {flat.memoryBytes() / 1e6:.1f} MB")

    index = HNSWFast(M=16, efConstruction=100)
    index.add(data, args.threads)
    print(f"\nHNSW graph: raw index {index.memoryBytes() / 1e6:.1f} MB")
    sample = np.random.default_rng(0).choice(len(data), 20_000, replace=False)
    for pieces in args.pieces:
        compressed = CompressedHNSW(index, ProductQuantizer(pieces).fit(data[sample]))
        print(f"\nHNSW with {pieces} pieces: codes + graph {compressed.memoryBytes() / 1e6:.1f} MB")
        print("  ef    raw: recall, qps      codes only: recall, qps    codes + rerank: recall, qps")
        for ef in (50, 100, 200, 400):
            raw = measure(lambda q: index.search(q, args.k, ef), queries, truth, args.k)
            codes = measure(lambda q: compressed.search(q, args.k, ef), queries, truth, args.k)
            rerank = measure(lambda q: compressed.search(q, args.k, ef, rerank=True), queries, truth, args.k)
            print(f"  {ef:<4d}  {raw[0]:.3f}, {raw[1]:>6.0f}          {codes[0]:.3f}, {codes[1]:>6.0f}"
                  f"            {rerank[0]:.3f}, {rerank[1]:>6.0f}", flush=True)
