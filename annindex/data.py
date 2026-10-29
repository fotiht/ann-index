import numpy as np


def read_fvecs(path, limit=None):
    """Read a .fvecs file (SIFT format): each row is an int32 dimension, then floats."""
    dim = np.fromfile(path, dtype=np.int32, count=1)[0]
    count = -1 if limit is None else limit * (dim + 1)
    rows = np.fromfile(path, dtype=np.int32, count=count).reshape(-1, dim + 1)
    return rows[:, 1:].copy().view(np.float32)


def random_data(n, dim, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
