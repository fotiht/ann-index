"""Product quantization: each vector is split into m pieces and each piece is replaced by
the index (one byte) of its nearest centroid, so a 128 dimensional float32 vector (512 bytes)
becomes m bytes."""
import numpy as np
from numba import njit


def nearestCentroid(points, centroids):
    """Index of the closest centroid for every row of points."""
    dist = (centroids ** 2).sum(axis=1)[None, :] - 2 * points @ centroids.T
    return dist.argmin(axis=1)


class ProductQuantizer:
    def __init__(self, pieces=16, centroids=256, iterations=15, seed=0):
        self.pieces = pieces
        self.centroidCount = centroids
        self.iterations = iterations
        self.seed = seed

    def fit(self, train):
        train = np.asarray(train, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        self.width = train.shape[1] // self.pieces
        self.codebooks = np.empty((self.pieces, self.centroidCount, self.width), np.float32)
        for p in range(self.pieces):
            part = train[:, p * self.width:(p + 1) * self.width]
            centroids = part[rng.choice(len(part), self.centroidCount, replace=False)].copy()
            for _ in range(self.iterations):
                owner = nearestCentroid(part, centroids)
                for c in range(self.centroidCount):
                    members = part[owner == c]
                    centroids[c] = members.mean(axis=0) if len(members) else part[rng.integers(len(part))]
            self.codebooks[p] = centroids
        return self

    def encode(self, data, batch=50_000):
        data = np.asarray(data, dtype=np.float32)
        codes = np.empty((len(data), self.pieces), np.uint8)
        for start in range(0, len(data), batch):
            chunk = data[start:start + batch]
            for p in range(self.pieces):
                part = chunk[:, p * self.width:(p + 1) * self.width]
                codes[start:start + batch, p] = nearestCentroid(part, self.codebooks[p])
        return codes

    def distanceTables(self, queries):
        """For each query, the squared distance from each of its pieces to every centroid.
        Shape (queries, pieces, centroids)."""
        queries = np.asarray(queries, dtype=np.float32)
        tables = np.empty((len(queries), self.pieces, self.centroidCount), np.float32)
        for p in range(self.pieces):
            part = queries[:, p * self.width:(p + 1) * self.width]
            centroids = self.codebooks[p]
            tables[:, p, :] = ((part ** 2).sum(axis=1)[:, None] + (centroids ** 2).sum(axis=1)[None, :]
                               - 2 * part @ centroids.T)
        return tables

    def memoryBytes(self):
        return self.codebooks.nbytes


@njit(cache=True, fastmath=True)
def scanCodes(codes, table):
    """Approximate squared distance from a query (given by its table) to every code."""
    out = np.empty(codes.shape[0], np.float32)
    for i in range(codes.shape[0]):
        total = np.float32(0.0)
        for p in range(codes.shape[1]):
            total += table[p, codes[i, p]]
        out[i] = total
    return out


class PQFlatIndex:
    """Exact scan over the compressed codes. Small, and as slow as any full scan."""

    def __init__(self, pieces=16, centroids=256, trainSize=20_000, seed=0):
        self.quantizer = ProductQuantizer(pieces, centroids, seed=seed)
        self.trainSize = trainSize
        self.seed = seed

    def add(self, data):
        data = np.asarray(data, dtype=np.float32)
        sample = np.random.default_rng(self.seed).choice(len(data), min(self.trainSize, len(data)), replace=False)
        self.quantizer.fit(data[sample])
        self.codes = self.quantizer.encode(data)

    def search(self, queries, k):
        tables = self.quantizer.distanceTables(queries)
        result = np.empty((len(queries), k), np.int64)
        for row, table in enumerate(tables):
            dist = scanCodes(self.codes, table)
            nearest = np.argpartition(dist, k - 1)[:k]
            result[row] = nearest[np.argsort(dist[nearest])]
        return result

    def memoryBytes(self):
        return self.codes.nbytes + self.quantizer.memoryBytes()
