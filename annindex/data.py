import numpy as np


def read_fvecs(path, limit=None):
    """Read a .fvecs file (SIFT format): each row is an int32 dimension, then floats."""
    dim = np.fromfile(path, dtype=np.int32, count=1)[0]
    count = -1 if limit is None else limit * (dim + 1)
    rows = np.fromfile(path, dtype=np.int32, count=count).reshape(-1, dim + 1)
    return rows[:, 1:].copy().view(np.float32)


def random_data(n, dim, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def read_ann_benchmarks(path, limit=None, queries=1000):
    """Read an ann-benchmarks .hdf5 file (for example GloVe). Returns the base vectors, the
    first `queries` query vectors, and the shipped true neighbours of those queries, which are
    only valid when the whole base set is used."""
    import h5py
    with h5py.File(path, "r") as f:
        base = f["train"][:limit] if limit else f["train"][:]
        return base, f["test"][:queries], f["neighbors"][:queries]
