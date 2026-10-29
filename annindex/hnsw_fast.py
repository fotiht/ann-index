"""The same HNSW algorithm as hnsw.py, with the graph stored in NumPy arrays and the
hot loops compiled with Numba."""
import heapq
import math

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def sqDist(vectors, i, query):
    total = 0.0
    for d in range(query.shape[0]):
        diff = vectors[i, d] - query[d]
        total += diff * diff
    return total


@njit(cache=True)
def searchLayer(vectors, links, counts, layer, query, entry, entryDist, ef, visited, stamp):
    """Best-first search of one layer. Returns up to ef ids and distances, closest first."""
    visited[entry] = stamp
    toVisit = [(entryDist, entry)]  # min-heap: closest first
    best = [(-entryDist, entry)]  # max-heap of the ef closest
    while len(toVisit) > 0:
        dist, current = heapq.heappop(toVisit)
        if len(best) >= ef and dist > -best[0][0]:
            break
        for k in range(counts[layer, current]):
            other = np.int64(links[layer, current, k])
            if visited[other] == stamp:
                continue
            visited[other] = stamp
            otherDist = sqDist(vectors, other, query)
            if len(best) < ef or otherDist < -best[0][0]:
                heapq.heappush(toVisit, (otherDist, other))
                heapq.heappush(best, (-otherDist, other))
                if len(best) > ef:
                    heapq.heappop(best)
    size = len(best)
    ids = np.empty(size, np.int64)
    dists = np.empty(size, np.float64)
    for i in range(size - 1, -1, -1):
        negDist, other = heapq.heappop(best)
        ids[i] = other
        dists[i] = -negDist
    return ids, dists


@njit(cache=True)
def selectNeighbours(vectors, ids, dists, count):
    """Keep a candidate only if it is closer to the node than to any already kept."""
    kept = np.empty(count, np.int64)
    size = 0
    for i in range(len(ids)):  # closest first
        if size == count:
            break
        good = True
        for j in range(size):
            if sqDist(vectors, ids[i], vectors[kept[j]]) <= dists[i]:
                good = False
                break
        if good:
            kept[size] = ids[i]
            size += 1
    # top up with the nearest rejected ones if the heuristic kept too few
    for i in range(len(ids)):
        if size == count:
            break
        used = False
        for j in range(size):
            if kept[j] == ids[i]:
                used = True
                break
        if not used:
            kept[size] = ids[i]
            size += 1
    return kept[:size]


@njit(cache=True)
def link(vectors, links, counts, layer, a, b, limit):
    """Add b to a's links, pruning with the neighbour heuristic if a has too many."""
    size = counts[layer, a]
    if size < limit:
        links[layer, a, size] = b
        counts[layer, a] = size + 1
        return
    ids = np.empty(size + 1, np.int64)
    dists = np.empty(size + 1, np.float64)
    for k in range(size):
        ids[k] = links[layer, a, k]
    ids[size] = b
    for k in range(size + 1):
        dists[k] = sqDist(vectors, ids[k], vectors[a])
    order = np.argsort(dists)
    kept = selectNeighbours(vectors, ids[order], dists[order], limit)
    for k in range(len(kept)):
        links[layer, a, k] = kept[k]
    counts[layer, a] = len(kept)


@njit(cache=True)
def build(vectors, links, counts, levels, M, efConstruction):
    n = vectors.shape[0]
    visited = np.zeros(n, np.int64)
    stamp = 0
    entry = 0
    topLevel = levels[0]
    for node in range(1, n):
        level = levels[node]
        query = vectors[node]
        ep = np.int64(entry)
        epDist = sqDist(vectors, ep, query)
        for l in range(topLevel, level, -1):
            stamp += 1
            ids, dists = searchLayer(vectors, links, counts, l, query, ep, epDist, 1, visited, stamp)
            ep, epDist = ids[0], dists[0]
        for l in range(min(level, topLevel), -1, -1):
            stamp += 1
            ids, dists = searchLayer(vectors, links, counts, l, query, ep, epDist,
                                     efConstruction, visited, stamp)
            limit = 2 * M if l == 0 else M
            for other in selectNeighbours(vectors, ids, dists, M):
                link(vectors, links, counts, l, node, other, limit)
                link(vectors, links, counts, l, other, node, limit)
            ep, epDist = ids[0], dists[0]
        if level > topLevel:
            entry, topLevel = node, level
    return entry, topLevel


@njit(cache=True)
def searchAll(vectors, links, counts, queries, k, ef, entry, topLevel):
    visited = np.zeros(vectors.shape[0], np.int64)
    stamp = 0
    result = np.full((queries.shape[0], k), -1, np.int64)
    for row in range(queries.shape[0]):
        query = queries[row]
        ep = np.int64(entry)
        epDist = sqDist(vectors, ep, query)
        for l in range(topLevel, 0, -1):
            stamp += 1
            ids, dists = searchLayer(vectors, links, counts, l, query, ep, epDist, 1, visited, stamp)
            ep, epDist = ids[0], dists[0]
        stamp += 1
        ids, dists = searchLayer(vectors, links, counts, 0, query, ep, epDist, max(ef, k), visited, stamp)
        for i in range(min(k, len(ids))):
            result[row, i] = ids[i]
    return result


class HNSWFast:
    """HNSW index with the same interface as HNSW in hnsw.py."""

    def __init__(self, M=16, efConstruction=100, metric="euclidean", seed=0):
        self.M = M
        self.efConstruction = efConstruction
        self.metric = metric
        self.seed = seed

    def _prepare(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.metric == "cosine":
            x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        elif self.metric != "euclidean":
            raise ValueError(f"unknown metric: {self.metric}")
        return x

    def add(self, data):
        self.vectors = self._prepare(data)
        rng = np.random.default_rng(self.seed)
        scale = 1 / math.log(self.M)
        levels = (-np.log(1 - rng.random(len(self.vectors))) * scale).astype(np.int64)
        shape = (levels.max() + 1, len(self.vectors))
        self.links = np.full(shape + (2 * self.M,), -1, dtype=np.int32)
        self.counts = np.zeros(shape, dtype=np.int32)
        self.entry, self.topLevel = build(self.vectors, self.links, self.counts, levels,
                                          self.M, self.efConstruction)

    def memoryBytes(self):
        return self.vectors.nbytes + 4 * int(self.counts.sum())  # 4 bytes per stored link id

    def search(self, queries, k, ef=50):
        return searchAll(self.vectors, self.links, self.counts, self._prepare(queries),
                         k, ef, self.entry, self.topLevel)
