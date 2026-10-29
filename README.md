# ann-index

An approximate nearest neighbour index written from scratch in Python and NumPy. The plan is an HNSW graph index (Malkov and Yashunin, 2016) that is benchmarked against exact search, and later against FAISS, on the SIFT dataset. The goal is an honest picture of how recall, speed and memory trade off.

## Status

Phase 1 (baseline and benchmark harness) is done, and the first HNSW results on SIFT are in. The HNSW code is pure Python for now, so speed work comes next.

## Baseline

Exact brute-force search on the first 100,000 SIFT base vectors (128 dimensions), 1,000 queries, k = 10. The exact search code reproduces the ground truth that ships with SIFT (checked on 100 queries against the full 1,000,000 vectors).

| Method | Recall@10 | Queries per second | Memory |
|---|---|---|---|
| Exact, Euclidean | 1.00 | 811 | 51.2 MB |
| Exact, cosine | 1.00 | 862 | 51.2 MB |

Numbers come from one 4-core server and will vary with the machine. Any index has to beat 811 queries per second at a recall close to 1.0 to be worth using at this size.

## HNSW on SIFT, first results

The first 20,000 SIFT base vectors, 1,000 queries, k = 10, built with M = 16 and efConstruction = 100. Exact search on the same data runs at 5,350 queries per second with recall 1.00.

| ef | Recall@10 | Queries per second |
|---|---|---|
| 10 | 0.878 | 1,931 |
| 20 | 0.958 | 1,350 |
| 50 | 0.995 | 698 |
| 100 | 0.999 | 391 |
| 200 | 1.000 | 221 |
| 400 | 1.000 | 120 |

Recall climbs quickly with ef, and 0.995 is reached at ef = 50. At this size, though, exact search is faster than HNSW. NumPy does the exact search as one large matrix product, while the HNSW search runs in a Python loop. A graph index only pays off once the data is large enough that scanning everything gets expensive, and our Python implementation needs to get faster before the comparison at 100,000 vectors or more is fair.

## Layout

- `annindex/exact.py`: exact k-NN search for Euclidean and cosine distance.
- `annindex/hnsw.py`: the HNSW index.
- `annindex/metrics.py`: recall@k.
- `annindex/data.py`: reads SIFT `.fvecs` files and makes random data.
- `bench.py`: build time, queries per second, recall and memory for any index.
- `sweep.py`: recall against queries per second for several values of `ef`.
- `tests/`: checks against brute force.

## Reproduce

Download `sift.tar.gz` from `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`, then:

```
tar -xzf sift.tar.gz -C data sift/sift_base.fvecs sift/sift_query.fvecs
python bench.py --base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs --n 100000
python -m pytest
```

## Next

1. Speed up the distance calls (Numba or batched NumPy) and re-run the sweep at 100,000 vectors.
2. Compare with FAISS HNSW on the same data. We do not expect to beat it.
3. Report build time and memory for HNSW next to the exact baseline.
