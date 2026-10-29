# ann-index

An approximate nearest neighbour index written from scratch in Python and NumPy. It is an HNSW graph index (Malkov and Yashunin, 2016), benchmarked against exact search and against FAISS on the SIFT dataset. The goal is an honest picture of how recall, speed and memory trade off.

## Status

Phases 1 to 3 are done: an exact baseline, an HNSW index in Python and in Numba, and a comparison with FAISS on SIFT at 100,000 and 1,000,000 vectors. At equal recall our index is about 1.3 to 1.8 times slower than FAISS and takes about 1.8 times as long to build. It is 20 times faster than exact search at 1,000,000 vectors.

## Baseline

Exact brute-force search on the first 100,000 SIFT base vectors (128 dimensions), 1,000 queries, k = 10. The exact search code reproduces the ground truth that ships with SIFT (checked on 100 queries against the full 1,000,000 vectors).

| Method | Recall@10 | Queries per second | Memory |
|---|---|---|---|
| Exact, Euclidean | 1.00 | 811 | 51.2 MB |
| Exact, cosine | 1.00 | 862 | 51.2 MB |

Numbers come from one 4-core server and vary a little from run to run (880 queries per second in a later run). On all 1,000,000 vectors, exact search runs at 88 queries per second.

## Comparison with FAISS

FAISS `IndexHNSWFlat` against our Numba index (`annindex/hnsw_fast.py`), same data and parameters (M = 16, efConstruction = 100), one thread each, 1,000 queries, k = 10. The script is `faiss_compare.py`. FAISS memory is the size of its serialized index.

100,000 vectors. Build: ours 36.7 s and 61.9 MB, FAISS 20.4 s and 65.6 MB. Exact search: about 880 queries per second.

| ef | Ours: recall, queries per second | FAISS: recall, queries per second |
|---|---|---|
| 10 | 0.833, 14,241 | 0.783, 24,284 |
| 20 | 0.921, 8,833 | 0.892, 15,862 |
| 50 | 0.982, 4,622 | 0.974, 7,709 |
| 100 | 0.997, 2,600 | 0.994, 4,359 |
| 200 | 1.000, 1,537 | 0.999, 2,453 |
| 400 | 1.000, 876 | 1.000, 1,270 |

1,000,000 vectors (the full SIFT base set). Build: ours 586 s and 618.5 MB, FAISS 332 s and 656.3 MB. The raw vectors take 512 MB. Exact search: 88 queries per second.

| ef | Ours: recall, queries per second | FAISS: recall, queries per second |
|---|---|---|
| 10 | 0.710, 8,584 | 0.684, 16,303 |
| 20 | 0.836, 6,160 | 0.814, 10,525 |
| 50 | 0.943, 3,224 | 0.929, 5,357 |
| 100 | 0.979, 1,771 | 0.974, 3,161 |
| 200 | 0.993, 1,029 | 0.992, 1,630 |
| 400 | 0.997, 573 | 0.998, 871 |

At the same ef our recall is a little higher. We have not checked why, but the two libraries may not do the same amount of work for the same ef, so the fair comparison is recall against speed. Judged that way, FAISS answers about 1.3 times as many queries per second at 100,000 vectors and about 1.5 times as many at 1,000,000. At 1,000,000 vectors our index reaches recall 0.979 at 1,771 queries per second, 20 times faster than exact search.

A remaining gap to FAISS is expected. FAISS is tuned C++ with hand-vectorized distance code, while ours is Numba code with a plain loop for each distance.

## Speed work

The pure Python version (`annindex/hnsw.py`) gives the same recall as the Numba version at every ef value but runs far slower. On 20,000 vectors at ef = 50 it handled 698 queries per second, against 4,409 for the first Numba version (6.3 times faster). Exact search was faster than the Python version at that size, because NumPy runs it as one large matrix product while the Python search takes one step at a time.

Three changes made the difference, in this order. Batching the neighbour selection step in NumPy roughly halved the Python build time. Compiling the search and insert loops with Numba gave most of the gain. Turning on fast-math for the distance loop, which lets the compiler vectorize it, then nearly doubled the speed again without changing recall (build at 100,000 vectors fell from 81 s to 37 s). The Python version stays in the repo as a readable reference.

## Layout

- `annindex/exact.py`: exact k-NN search for Euclidean and cosine distance.
- `annindex/hnsw.py`: the HNSW index in plain Python, kept as a readable reference.
- `annindex/hnsw_fast.py`: the same algorithm with the graph in NumPy arrays and Numba-compiled loops.
- `annindex/metrics.py`: recall@k.
- `annindex/data.py`: reads SIFT `.fvecs` files and makes random data.
- `bench.py`: build time, queries per second, recall and memory for any index.
- `sweep.py`: recall against queries per second for several values of `ef`.
- `faiss_compare.py`: our index against FAISS HNSW on the same data.
- `tests/`: checks against brute force.

## Reproduce

Download `sift.tar.gz` from `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`, then:

```
tar -xzf sift.tar.gz -C data sift/sift_base.fvecs sift/sift_query.fvecs sift/sift_groundtruth.ivecs
python bench.py --base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs --n 100000
python sweep.py --base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs --n 100000
python faiss_compare.py --n 100000
python -m pytest
```

## Possible next steps

1. Multiple threads for building and searching, or hand-written SIMD distance code, to narrow the gap to FAISS.
2. Product quantization to reduce memory.
3. Other datasets, such as GloVe, where the data is harder than SIFT.
