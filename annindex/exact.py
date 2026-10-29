import numpy as np


def exact_knn(data, queries, k, metric="euclidean", batch=256):
    """Brute-force k nearest neighbours. Returns indices, closest first."""
    data = np.asarray(data, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    if metric == "cosine":
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    elif metric != "euclidean":
        raise ValueError(f"unknown metric: {metric}")

    dataNorms = (data ** 2).sum(axis=1)
    result = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), batch):
        chunk = queries[start:start + batch]
        # squared distance up to a per-query constant: |x|^2 - 2 q.x
        dist = dataNorms[None, :] - 2 * chunk @ data.T
        nearest = np.argpartition(dist, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(dist, nearest, axis=1), axis=1)
        result[start:start + batch] = np.take_along_axis(nearest, order, axis=1)
    return result
