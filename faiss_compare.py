"""Compare our HNSW (Numba) with FAISS IndexHNSWFlat on the same SIFT data and parameters.

Both run on one thread. Memory for FAISS is the size of the serialized index.
"""
import argparse
import time

import faiss
import numpy as np

from annindex.data import read_fvecs
from annindex.exact import exact_knn
from annindex.hnsw_fast import HNSWFast
from annindex.metrics import recall_at_k

EF_VALUES = (10, 20, 50, 100, 200, 400)


def queriesPerSecond(search, queries):
    start = time.perf_counter()
    result = search(queries)
    return len(queries) / (time.perf_counter() - start), result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/sift/sift_base.fvecs")
    parser.add_argument("--query", default="data/sift/sift_query.fvecs")
    parser.add_argument("--truth", default="data/sift/sift_groundtruth.ivecs")
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--efc", type=int, default=100)
    args = parser.parse_args()

    faiss.omp_set_num_threads(1)
    data, queries = read_fvecs(args.base, args.n), read_fvecs(args.query, 1000)
    if args.n == 1_000_000:  # SIFT ships the exact neighbours for the full set
        truth = np.fromfile(args.truth, dtype=np.int32).reshape(-1, 101)[:1000, 1:]
    else:
        truth = exact_knn(data, queries, args.k)

    ours = HNSWFast(M=args.M, efConstruction=args.efc)
    start = time.perf_counter()
    ours.add(data)
    oursBuild = time.perf_counter() - start
    ours.search(queries[:1], args.k)  # compile the search code before timing it

    theirs = faiss.IndexHNSWFlat(data.shape[1], args.M)
    theirs.hnsw.efConstruction = args.efc
    start = time.perf_counter()
    theirs.add(data)
    theirsBuild = time.perf_counter() - start

    print(f"n = {args.n}, M = {args.M}, efConstruction = {args.efc}, one thread")
    print(f"build: ours {oursBuild:.1f} s, {ours.memoryBytes() / 1e6:.1f} MB | "
          f"FAISS {theirsBuild:.1f} s, {faiss.serialize_index(theirs).nbytes / 1e6:.1f} MB")
    print("ef    ours recall / qps       FAISS recall / qps")
    for ef in EF_VALUES:
        oursQps, found = queriesPerSecond(lambda q: ours.search(q, args.k, ef), queries)
        oursRecall = recall_at_k(found, truth, args.k)
        theirs.hnsw.efSearch = ef
        theirsQps, (_, found) = queriesPerSecond(lambda q: theirs.search(q, args.k), queries)
        theirsRecall = recall_at_k(found, truth, args.k)
        print(f"{ef:<5d} {oursRecall:.3f} / {oursQps:>7.0f}        {theirsRecall:.3f} / {theirsQps:>7.0f}", flush=True)
