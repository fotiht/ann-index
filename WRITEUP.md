# What the benchmarks show

This is a short summary of the results in the README. Every number here comes from the logs in `results/`, measured on one shared 4-core server, so timings vary by 20 to 30 percent between runs.

## The setup

The index is an HNSW graph (Malkov and Yashunin, 2016) with M = 16 and efConstruction = 100. I compared it with exact search and with FAISS `IndexHNSWFlat` on SIFT (128 dimensions) and GloVe (100 dimensions, cosine distance), at 100,000 vectors and on the full sets of about 1 million.

## Results

- On SIFT at 1,000,000 vectors, ours reaches recall 0.979 at 3,034 queries per second, 34 times faster than exact search. At equal recall, computed from the logs by interpolation and not re-measured, the two are within about 10 percent of each other on one thread at 1,000,000 vectors, and ours is up to 1.19 times faster at 100,000 vectors.
- GloVe is much harder. At 100,000 vectors, ef = 100 gives recall 0.875, where SIFT gave 0.997. On the full set it takes ef = 400 to pass 0.9.
- On the full GloVe set, at equal recall, FAISS is faster below recall 0.72 (by about 23 percent at 0.46), ours is level near 0.72 and ahead above it (1.09 times at 0.9, about 2.2 times at 0.975). At ef = 3,200 ours reaches 0.983 at 117 queries per second against 0.975 at 76.
- Product quantization cuts the counted memory of the 100,000-vector SIFT index (vectors and links in use) from 61.9 MB to 12.5 MB (16 pieces). Allocated memory is larger and shrinks less, see the README. Searching the codes alone caps recall at 0.62. Re-scoring candidates with the raw vectors brings it back to 0.986.
- Four threads speed up search about 1.9 to 3.4 times for ours and 2.5 to 3.7 times for FAISS, so with four threads FAISS is ahead at equal recall (ours at 0.79 to 0.96 times its speed below recall 0.99).

## Where it loses to FAISS, and what I know about why

FAISS is tuned C++ with hand-vectorized distance code. Mine is Numba with a plain float32 loop. On one thread there is no clear speed gap on SIFT at equal recall, but FAISS is ahead with four threads, and the build is slower: 365 s against 321 s at 1,000,000 vectors.

Two things in the data I have not explained. My recall is slightly higher at the same ef, which probably means the two libraries do different amounts of work per ef, so recall against speed is the fair comparison. On the full GloVe set the curves cross near recall 0.72, and I have not found out why.

## What changed the speed most

On 100,000 SIFT vectors the biggest gains came from compiling the loops with Numba, turning on fast-math for the distance loop, and summing in float32 instead of float64 (search 4,552 to 7,258 queries per second at recall 0.982, with the same recall). Unrolling the loop by hand was slower than letting the compiler vectorize it.

## Limits

The PQ re-score needs the raw vectors, so memory is saved only if they are kept on disk. Float32 sums can lose precision. SIFT coordinates are whole numbers, so this test would not show it, and I have not tested data where it could matter. The parallel build inserts points in batches of 64, which cost almost no recall (0.981 against 0.982 at ef = 50).
