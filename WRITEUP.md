# What the benchmarks show

This is a short summary of the results in the README. Every number here comes from the logs in `results/`, measured on one shared 4-core server, so timings vary by 20 to 30 percent between runs.

## The setup

The index is an HNSW graph (Malkov and Yashunin, 2016) with M = 16 and efConstruction = 100. I compared it with exact search and with FAISS `IndexHNSWFlat` on SIFT (128 dimensions) and GloVe (100 dimensions, cosine distance), at 100,000 vectors and on the full sets of about 1 million.

## Results

- On SIFT at 1,000,000 vectors, ours reaches recall 0.979 at 3,034 queries per second, 34 times faster than exact search. At equal recall FAISS is about 1.05 to 1.1 times faster on one thread.
- GloVe is much harder. At 100,000 vectors, ef = 100 gives recall 0.875, where SIFT gave 0.997. On the full set it takes ef = 400 to pass 0.9.
- On the full GloVe set FAISS is up to 28 percent faster at ef up to 400. From ef = 800 ours is ahead, and at ef = 3,200 it reaches 0.983 at 117 queries per second against 0.975 at 76.
- Product quantization cuts the 100,000-vector SIFT index from 61.9 MB to 12.5 MB (16 pieces). Searching the codes alone caps recall at 0.62. Re-scoring candidates with the raw vectors brings it back to 0.986.
- Four threads speed up search about 3 times for both libraries, so the gap to FAISS stays the same.

## Where it loses to FAISS, and what I know about why

FAISS is tuned C++ with hand-vectorized distance code. Mine is Numba with a plain float32 loop. The gap is small (5 to 10 percent on SIFT), and the build is also slower: 365 s against 321 s at 1,000,000 vectors.

Two things in the data I have not explained. My recall is slightly higher at the same ef, which probably means the two libraries do different amounts of work per ef, so recall against speed is the fair comparison. The GloVe curves also cross at ef = 800 on the full set, and I have not found out why.

## What changed the speed most

On 100,000 SIFT vectors the biggest gains came from compiling the loops with Numba, turning on fast-math for the distance loop, and summing in float32 instead of float64 (search 4,552 to 7,258 queries per second at recall 0.982, with the same recall). Unrolling the loop by hand was slower than letting the compiler vectorize it.

## Limits

The PQ re-score needs the raw vectors, so memory is saved only if they are kept on disk. Float32 sums lose some precision, which did not change recall here. The parallel build inserts points in batches of 64, which cost almost no recall (0.981 against 0.982 at ef = 50).
