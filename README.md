# ann-index

An approximate nearest neighbour index written from scratch in Python and NumPy. The plan is an HNSW graph index (Malkov and Yashunin, 2016) that is benchmarked against exact search, and later against FAISS, on the SIFT dataset. The goal is an honest picture of how recall, speed and memory trade off.

## Status

Phase 1 (baseline and benchmark harness) is done. An HNSW implementation and its tests exist and pass, but it has not yet been run on SIFT, so there are no SIFT results for it.

## Baseline

Exact brute-force search on the first 100,000 SIFT base vectors (128 dimensions), 1,000 queries, k = 10. The exact search code reproduces the ground truth that ships with SIFT (checked on 100 queries against the full 1,000,000 vectors).

| Method | Recall@10 | Queries per second | Memory |
|---|---|---|---|
| Exact, Euclidean | 1.00 | 811 | 51.2 MB |
| Exact, cosine | 1.00 | 862 | 51.2 MB |

Numbers come from one 4-core server and will vary with the machine. Any index has to beat 811 queries per second at a recall close to 1.0 to be worth using at this size.

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

1. Run `sweep.py` on SIFT and record the recall and speed curve for HNSW.
2. Speed up the distance calls (Numba or batched NumPy).
3. Compare with FAISS HNSW on the same data. We do not expect to beat it.
