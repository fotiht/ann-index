# ann-index

An approximate nearest neighbour index written from scratch in Python and NumPy. It is an HNSW graph index (Malkov and Yashunin, 2016), benchmarked against exact search and against FAISS on the SIFT dataset. The goal is an honest picture of how recall, speed and memory trade off.

![Recall against queries per second, SIFT and GloVe, ours and FAISS](results/recall_vs_qps.png)

## Quickstart

No dataset download is needed to try it. This builds the index on 10,000 random vectors and prints recall against queries per second for several values of ef:

```
pip install numpy numba pytest
python sweep.py
python -m pytest
```

Random vectors are an easy case. The SIFT and GloVe results below come from the commands in the Reproduce section.

See [WRITEUP.md](WRITEUP.md) for what the benchmarks show and where this index loses to FAISS.

## Status

Done: an exact baseline, HNSW in Python and in Numba, a comparison with FAISS on SIFT at 100,000 and 1,000,000 vectors, multi-threaded build and search, and product quantization. Compared at equal recall (computed from the logged curves, see the comparison below), on one thread our index runs at 1.05 to 1.19 times the speed of FAISS on SIFT at 100,000 vectors, and within about 10 percent of it either way at 1,000,000 vectors, except at recall 0.997 where FAISS is ahead. With four threads FAISS is ahead. The index is 34 times faster than exact search at 1,000,000 vectors, and product quantization shrinks the counted index (vectors and links in use) from 61.9 MB to 12.5 MB at 100,000 vectors, though the allocated memory shrinks less (see the memory note). On GloVe, a harder dataset, recall is much lower for the same ef. On the full set, at equal recall, FAISS is faster below recall 0.72 and our index is level at about 0.72 and ahead above it.

## Baseline

Exact brute-force search on the first 100,000 SIFT base vectors (128 dimensions), 1,000 queries, k = 10. The exact search code reproduces the ground truth that ships with SIFT (checked on 100 queries against the full 1,000,000 vectors).

| Method | Recall@10 | Queries per second | Memory |
|---|---|---|---|
| Exact, Euclidean | 1.00 | 811 | 51.2 MB |
| Exact, cosine | 1.00 | 862 | 51.2 MB |

Numbers come from one 4-core server and vary a little from run to run (880 queries per second in a later run). On all 1,000,000 vectors, exact search runs at 88 queries per second.

## Comparison with FAISS

FAISS `IndexHNSWFlat` against our Numba index (`annindex/hnsw_fast.py`), same data and parameters (M = 16, efConstruction = 100), one thread each, 1,000 queries, k = 10. The script is `faiss_compare.py`. FAISS memory is the size of its serialized index. Our memory is the vectors plus the links in use (`memoryBytes`). The arrays reserve 2M link slots for every node on every layer, so the allocated size is larger: about 117 MB at 100,000 SIFT vectors and about 1,304 MB at 1,000,000 (5 and 6 layers with seed 0, computed from the code by `equal_recall.py`, not measured). The memory figures below count links in use only, so they are not directly comparable with the FAISS figures, and our allocated index is probably the larger of the two.

100,000 vectors. Build: ours 26.2 s and 61.9 MB, FAISS 20.6 s and 65.6 MB. Exact search: about 880 queries per second.

| ef | Ours: recall, queries per second | FAISS: recall, queries per second |
|---|---|---|
| 10 | 0.833, 23,218 | 0.783, 24,403 |
| 20 | 0.921, 15,046 | 0.892, 15,343 |
| 50 | 0.982, 7,258 | 0.974, 7,748 |
| 100 | 0.997, 4,037 | 0.994, 4,458 |
| 200 | 1.000, 2,396 | 0.999, 2,459 |
| 400 | 1.000, 1,332 | 1.000, 1,285 |

1,000,000 vectors (the full SIFT base set). Build: ours 365 s and 618.5 MB, FAISS 321 s and 656.3 MB. The raw vectors take 512 MB. Exact search: 88 queries per second.

| ef | Ours: recall, queries per second | FAISS: recall, queries per second |
|---|---|---|
| 10 | 0.710, 15,446 | 0.684, 16,957 |
| 20 | 0.836, 10,364 | 0.814, 11,048 |
| 50 | 0.943, 5,301 | 0.929, 5,727 |
| 100 | 0.979, 3,034 | 0.974, 3,197 |
| 200 | 0.993, 1,622 | 0.992, 1,705 |
| 400 | 0.997, 850 | 0.998, 898 |

At the same ef our recall is a little higher. We have not checked why, but the two libraries may not do the same amount of work for the same ef, so the fair comparison is recall against speed. Judged that way, by interpolating each curve to the other's recall (computed from these logs by `equal_recall.py`, not re-measured), our index runs at 1.05 to 1.19 times the speed of FAISS at 100,000 vectors up to recall 0.994, and at 0.98 to 1.09 times at 1,000,000 vectors up to recall 0.993. At recall 0.997 on the 1,000,000 vector set FAISS is ahead (ours at about 0.82 times its speed). With four threads at 100,000 vectors FAISS is ahead below recall 0.99 (ours at 0.79 to 0.96 times its speed). The interpolation is linear between a few logged points, so differences of a few percent are within its error. At 1,000,000 vectors our index reaches recall 0.979 at 3,034 queries per second, 34 times faster than exact search.

On one thread there is no clear speed gap to FAISS on SIFT at equal recall. FAISS is tuned C++ with hand-vectorized distance code, while ours is Numba code with a plain float32 loop for each distance.

## GloVe, a harder dataset

GloVe word vectors (100 dimensions, 1,183,514 base vectors, cosine distance) from ann-benchmarks. Same parameters (M = 16, efConstruction = 100), one thread, 1,000 queries, k = 10. FAISS gets unit-length vectors, since cosine ranking equals Euclidean ranking on those. Run with `python faiss_compare.py --dataset glove --n 100000`; logs are in `results/`.

100,000 vectors. Build: ours 31.2 s and 50.4 MB, FAISS 25.4 s and 54.4 MB. Exact search: 865 queries per second.

| ef | Ours: recall, queries per second | FAISS: recall, queries per second |
|---|---|---|
| 10 | 0.555, 21,411 | 0.542, 22,108 |
| 20 | 0.677, 14,656 | 0.669, 14,237 |
| 50 | 0.803, 7,091 | 0.795, 6,719 |
| 100 | 0.875, 3,933 | 0.865, 3,789 |
| 200 | 0.927, 2,034 | 0.919, 2,017 |
| 400 | 0.967, 1,083 | 0.958, 1,047 |
| 800 | 0.989, 555 | 0.985, 499 |
| 1,600 | 0.997, 287 | 0.994, 215 |
| 3,200 | 1.000, 149 | 0.998, 83 |

All 1,183,514 vectors. Build: ours 499 s and 594.5 MB, FAISS 412 s and 644.2 MB. The raw vectors take 473 MB. Exact search: 60 queries per second.

| ef | Ours: recall, queries per second | FAISS: recall, queries per second |
|---|---|---|
| 10 | 0.464, 11,961 | 0.449, 15,344 |
| 20 | 0.590, 8,829 | 0.574, 10,103 |
| 50 | 0.728, 4,662 | 0.712, 5,155 |
| 100 | 0.804, 2,721 | 0.788, 3,020 |
| 200 | 0.863, 1,556 | 0.849, 1,649 |
| 400 | 0.905, 826 | 0.898, 869 |
| 800 | 0.938, 435 | 0.931, 424 |
| 1,600 | 0.966, 228 | 0.959, 190 |
| 3,200 | 0.983, 117 | 0.975, 76 |

GloVe is much harder than SIFT. On 100,000 vectors ef = 100 gives recall 0.875, where SIFT gave 0.997, and on the full set it takes ef = 400 to pass 0.9. One likely reason is that the word vectors have no clear clusters, so the graph search has to visit more points to find the true neighbours. I have not tested this. Even so, our index at ef = 100 on the full set answers 2,721 queries per second at recall 0.804, which is 45 times faster than exact search.

Against FAISS the two are close. At 100,000 vectors ours is level or ahead from ef = 20 up, and 1.8 times as fast at ef = 3,200 (149 queries per second against 83, recall 1.000 against 0.998). On the full set FAISS is ahead by up to 28 percent at low ef (ef = 10 to 400), and ours is ahead from ef = 800 (435 against 424), reaching 0.983 at 117 queries per second against 0.975 at 76 for FAISS at ef = 3,200. We have not checked why the curves cross. Our recall is also a little higher at the same ef throughout. Exact search matters here too: at 100,000 vectors it handles 865 queries per second, which is faster than either index once recall is above about 0.98 (at ef = 400 ours reaches 0.967 at 1,083, but at ef = 800 it reaches 0.989 at only 555). On the full set exact search handles 60 queries per second, so the graph index is still well ahead there.

## Threads

`HNSWFast.add(data, threads=4)` and `HNSWFast.search(queries, k, ef, threads=4)` use several threads. Searching splits the queries into chunks, one per thread, and returns the same answers as one thread. Building inserts the points in batches of 64. Each batch looks for neighbours in parallel against the graph built so far and is then linked in parallel, so points in the same batch do not see each other. Linking gives exactly the graph that adding the points one after another would give: each new point's own links are written on separate threads, and the links back to existing points are grouped by target and applied in node order. The graph therefore does not depend on the thread count, and `tests/test_hnsw_fast.py` checks that. That cost almost no recall (0.981 against 0.982 at ef = 50), and the first 2,000 points go in one at a time.

Four threads on 100,000 vectors, same parameters as above. Build: ours 10.9 s (26.2 s on one thread), FAISS 5.6 s.

| ef | Ours: recall, queries per second | FAISS: recall, queries per second |
|---|---|---|
| 10 | 0.831, 44,541 | 0.783, 61,270 |
| 20 | 0.918, 32,508 | 0.892, 45,818 |
| 50 | 0.981, 21,578 | 0.974, 27,270 |
| 100 | 0.997, 13,659 | 0.994, 15,855 |
| 200 | 1.000, 6,471 | 0.999, 8,572 |
| 400 | 1.000, 4,308 | 1.000, 4,707 |

Search speed grows about 3 times with four threads for both libraries, so the gap to FAISS stays about the same. Multi-thread timings on this shared server vary by 20 to 30 percent between runs. The build gains less than the search; the first 2,000 points are still inserted on one thread.

## Product quantization

Product quantization (`annindex/pq.py`) splits each 128 dimensional vector into pieces, here 16 or 32, and replaces each piece by the number of its nearest centroid out of 256, found by k-means on 20,000 training vectors. A 512 byte vector becomes 16 or 32 bytes. A search computes a small table of distances from the query to every centroid, and the distance to a stored vector is then a sum of table lookups.

`annindex/hnsw_pq.py` searches the HNSW graph with these distances, so the raw vectors are not needed during the search. Results on 100,000 vectors, one thread. Raw vectors take 51.2 MB and the raw index with its graph takes 61.9 MB.

| Pieces | Memory (codes, codebooks, graph) | Recall from codes alone | Recall with a re-score of the ef candidates |
|---|---|---|---|
| 16 | 12.5 MB | 0.618 | 0.986 at ef = 100 |
| 32 | 14.1 MB | 0.756 | 0.995 at ef = 100 |

Searching through codes alone is limited by the quantization error. Recall stays at 0.62 (16 pieces) and 0.76 (32 pieces) however large ef is, which matches a full scan of the same codes. Re-scoring the candidates with the raw vectors restores recall to 0.986 and 0.995, and it is faster than the raw search at the same ef (4,875 queries per second against 3,689 at ef = 100 with 16 pieces; with 32 pieces the two are level, 4,328 against 4,232) because code lookups are cheaper than 128 dimensional distances. The catch is that the raw vectors are still needed for the re-score, for example on disk, so this saves memory only if they are kept off the main memory. At equal recall the raw search is not slower (raw ef = 50 reaches 0.981 at 5,627 queries per second, the re-score reaches 0.986 at 4,875). The memory figures above also count only the links in use. The link table is still allocated in full (about 66 MB at 100,000 vectors), and the index object keeps the raw vectors in memory even after `moveVectorsToDisk`, so the process does not shrink fivefold as the code stands. Full numbers for each ef value are in `results/pq_100k.log`.

## Speed work

The pure Python version (`annindex/hnsw.py`) gives the same recall as the Numba version at every ef value but runs far slower. On 20,000 vectors at ef = 50 it handled 698 queries per second, against 4,409 for the first Numba version (6.3 times faster). Exact search was faster than the Python version at that size, because NumPy runs it as one large matrix product while the Python search takes one step at a time.

Three changes made the difference, in this order. Batching the neighbour selection step in NumPy roughly halved the Python build time. Compiling the search and insert loops with Numba gave most of the gain. Turning on fast-math for the distance loop, which lets the compiler vectorize it, then nearly doubled the speed again without changing recall (build at 100,000 vectors fell from 81 s to 37 s). A fourth change was summing the squared differences in float32 instead of float64, so that each AVX2 instruction handles eight values instead of four plus a conversion: at 100,000 SIFT vectors and recall 0.982, search went from 4,552 to 7,258 queries per second and the build from 37.9 s to 26.2 s, with the same recall. A micro benchmark of the distance function alone showed about twice the speed. Unrolling the loop by hand with four accumulators was slower than the compiler's own vectorisation. Float32 sums lose some precision, which did not change recall here. The Python version stays in the repo as a readable reference.

## Layout

- `annindex/exact.py`: exact k-NN search for Euclidean and cosine distance.
- `annindex/hnsw.py`: the HNSW index in plain Python, kept as a readable reference.
- `annindex/hnsw_fast.py`: the same algorithm with the graph in NumPy arrays and Numba-compiled loops.
- `annindex/pq.py`: product quantization, plus a scan over the codes.
- `annindex/hnsw_pq.py`: search the HNSW graph through PQ codes.
- `annindex/metrics.py`: recall@k.
- `annindex/data.py`: reads SIFT `.fvecs` files and makes random data.
- `bench.py`: build time, queries per second, recall and memory for any index.
- `sweep.py`: recall against queries per second for several values of `ef`.
- `faiss_compare.py`: our index against FAISS HNSW on the same data.
- `plot_results.py`: recall against queries per second plots from the logs in `results/`.
- `pq_bench.py`: recall, speed and memory with product quantization.
- `equal_recall.py`: speed ratios against FAISS at equal recall and the allocated memory of our index, computed from the logs in `results/` (no benchmark is re-run).
- `tests/`: checks against brute force.

## Reproduce

Download `sift.tar.gz` from `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`, then:

```
tar -xzf sift.tar.gz -C data sift/sift_base.fvecs sift/sift_query.fvecs sift/sift_groundtruth.ivecs
python bench.py --base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs --n 100000
python sweep.py --base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs --n 100000
python faiss_compare.py --n 100000 --threads 4
python pq_bench.py
python plot_results.py
python -m pytest
```

## Possible next steps

1. Measure the disk re-score. `CompressedHNSW.moveVectorsToDisk(path)` now keeps the raw vectors in a memory-mapped file, and a test checks that it returns the same results as the in-memory re-score, but I have not timed it.
