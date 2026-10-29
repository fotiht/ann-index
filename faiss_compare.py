"""Compare our HNSW (Numba) with FAISS IndexHNSWFlat on the same data and parameters.
Datasets: SIFT (Euclidean) and GloVe (cosine, which FAISS handles here by normalizing the vectors).

Both build and search on --threads threads. Memory for FAISS is the size of the serialized index.
"""
import argparse
import time

import faiss
import numpy as np

from annindex.data import read_ann_benchmarks, read_fvecs
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
    parser.add_argument("--dataset", choices=["sift", "glove"], default="sift")
    parser.add_argument("--base", default="data/sift/sift_base.fvecs")
    parser.add_argument("--query", default="data/sift/sift_query.fvecs")
    parser.add_argument("--truth", default="data/sift/sift_groundtruth.ivecs")
    parser.add_argument("--glove", default="data/glove-100-angular.hdf5")
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--efc", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1, help="threads used for building and searching, for both indexes")
    parser.add_argument("--efs", type=int, nargs="+", default=EF_VALUES, help="ef values to sweep")
    args = parser.parse_args()

    faiss.omp_set_num_threads(args.threads)
    metric = "euclidean" if args.dataset == "sift" else "cosine"
    if args.dataset == "sift":
        data, queries = read_fvecs(args.base, args.n), read_fvecs(args.query, 1000)
        shipped = np.fromfile(args.truth, dtype=np.int32).reshape(-1, 101)[:1000, 1:]
        wholeSet = args.n == 1_000_000
    else:
        data, queries, shipped = read_ann_benchmarks(args.glove, args.n)
        wholeSet = args.n >= 1_183_514
    # the shipped neighbours are for the full base set, so a subset needs its own
    truth = shipped[:, :args.k] if wholeSet else exact_knn(data, queries, args.k, metric)

    ours = HNSWFast(M=args.M, efConstruction=args.efc, metric=metric)
    start = time.perf_counter()
    ours.add(data, args.threads)
    oursBuild = time.perf_counter() - start
    ours.search(queries[:8], args.k, 50, args.threads)  # compile the search code before timing it

    # cosine ranking equals Euclidean ranking on unit vectors, so FAISS gets normalized data
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    theirData, theirQueries = (data, queries) if metric == "euclidean" else (unit(data), unit(queries))
    theirs = faiss.IndexHNSWFlat(data.shape[1], args.M)
    theirs.hnsw.efConstruction = args.efc
    start = time.perf_counter()
    theirs.add(theirData)
    theirsBuild = time.perf_counter() - start

    print(f"{args.dataset}, n = {len(data)}, M = {args.M}, efConstruction = {args.efc}, {args.threads} thread(s)")
    print(f"build: ours {oursBuild:.1f} s, {ours.memoryBytes() / 1e6:.1f} MB | "
          f"FAISS {theirsBuild:.1f} s, {faiss.serialize_index(theirs).nbytes / 1e6:.1f} MB")
    print("ef    ours recall / qps       FAISS recall / qps")
    for ef in args.efs:
        oursQps, found = queriesPerSecond(lambda q: ours.search(q, args.k, ef, args.threads), queries)
        oursRecall = recall_at_k(found, truth, args.k)
        theirs.hnsw.efSearch = ef
        theirsQps, (_, found) = queriesPerSecond(lambda q: theirs.search(q, args.k), theirQueries)
        theirsRecall = recall_at_k(found, truth, args.k)
        print(f"{ef:<5d} {oursRecall:.3f} / {oursQps:>7.0f}        {theirsRecall:.3f} / {theirsQps:>7.0f}", flush=True)
