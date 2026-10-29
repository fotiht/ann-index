# ann-index

An approximate nearest neighbour index written from scratch in Python and NumPy. The plan is an HNSW graph index (Malkov and Yashunin, 2016) that is benchmarked against exact search, and later against FAISS, on the SIFT dataset. The goal is an honest picture of how recall, speed and memory trade off.

## Status

Phase 2 (speed) is done. The first HNSW version was pure Python and slower than exact search. A Numba version of the same algorithm now beats exact search on the 100,000-vector SIFT subset. Next is the comparison with FAISS.

## Baseline

Exact brute-force search on the first 100,000 SIFT base vectors (128 dimensions), 1,000 queries, k = 10. The exact search code reproduces the ground truth that ships with SIFT (checked on 100 queries against the full 1,000,000 vectors).

| Method | Recall@10 | Queries per second | Memory |
|---|---|---|---|
| Exact, Euclidean | 1.00 | 811 | 51.2 MB |
| Exact, cosine | 1.00 | 862 | 51.2 MB |

Numbers come from one 4-core server and will vary with the machine. Any index has to beat 811 queries per second at a recall close to 1.0 to be worth using at this size.

## HNSW on SIFT, 100,000 vectors

The Numba implementation (`annindex/hnsw_fast.py`) on the first 100,000 SIFT base vectors, 1,000 queries, k = 10, M = 16, efConstruction = 100. It took 81 seconds to build and uses 61.9 MB, against 51.2 MB for the raw vectors. In the same run, exact search reached 880 queries per second.

| ef | Recall@10 | Queries per second | Speed-up over exact |
|---|---|---|---|
| 10 | 0.816 | 6,550 | 7.4x |
| 20 | 0.910 | 5,326 | 6.1x |
| 50 | 0.978 | 2,836 | 3.2x |
| 100 | 0.996 | 1,608 | 1.8x |
| 200 | 1.000 | 971 | 1.1x |
| 400 | 1.000 | 579 | 0.7x |

At a recall of 0.996 the index answers 1.8 times as many queries per second as exact search, and at 0.978 it answers 3.2 times as many. Past ef = 200 the extra recall costs more than exact search saves. The gap should widen on the full million vectors, since exact search slows down in proportion to the data while HNSW search slows down much less. We have not run that yet.

## Speed work

The pure Python version (`annindex/hnsw.py`) gave the same recall at every ef value but ran slowly. On 20,000 vectors at ef = 50 it handled 698 queries per second, against 4,409 for the Numba version (6.3 times faster), with identical recall of 0.995. Exact search was faster than the Python version at that size, because NumPy runs it as one large matrix product while the Python search loops one step at a time. Batching the neighbour selection step in NumPy cut the Python build time roughly in half, but most of the gain came from compiling the search and insert loops with Numba. The Python version stays in the repo as a readable reference.

## Layout

- `annindex/exact.py`: exact k-NN search for Euclidean and cosine distance.
- `annindex/hnsw.py`: the HNSW index in plain Python, kept as a readable reference.
- `annindex/hnsw_fast.py`: the same algorithm with the graph in NumPy arrays and Numba-compiled loops.
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
python sweep.py --base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs --n 100000
python -m pytest
```

## Next

1. Compare with FAISS HNSW on the same data and parameters. We do not expect to beat it.
2. Run the sweep on all 1,000,000 vectors.
3. Optional: product quantization to cut the memory.
