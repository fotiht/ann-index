"""The same HNSW algorithm as hnsw.py, with the graph stored in NumPy arrays and the
hot loops compiled with Numba."""
import heapq
import math

import numba
import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def sqDist(vectors, i, query):
    total = np.float32(0.0)  # float32 sums let the compiler use 8-wide vector instructions
    for d in range(query.shape[0]):
        diff = vectors[i, d] - query[d]
        total += diff * diff
    return np.float64(total)


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
def build(vectors, links, counts, levels, M, efConstruction, stop):
    """Insert nodes 1 to stop - 1 one at a time (node 0 is the first entry point)."""
    n = vectors.shape[0]
    visited = np.zeros(n, np.int64)
    stamp = 0
    entry = 0
    topLevel = levels[0]
    for node in range(1, stop):
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


@njit(cache=True, parallel=True)
def findBatch(vectors, links, counts, levels, first, size, entry, topLevel, efConstruction,
              visitedPool, stampBase, candIds, candDists, candCounts):
    """Search the graph for the neighbours of nodes first to first + size - 1, in parallel.
    The graph is only read here, so the threads do not interfere."""
    workers = visitedPool.shape[0]
    bounds = np.linspace(0, size, workers + 1).astype(np.int64)
    for w in prange(workers):
        visited = visitedPool[w]
        stamp = stampBase
        for slot in range(bounds[w], bounds[w + 1]):
            node = first + slot
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
                candCounts[slot, l] = len(ids)
                candIds[slot, l, :len(ids)] = ids
                candDists[slot, l, :len(ids)] = dists
                ep, epDist = ids[0], dists[0]


@njit(cache=True, parallel=True)
def linkBatch(vectors, links, counts, levels, first, size, topLevel, M, candIds, candDists, candCounts):
    """Link the batch into the graph. The result is the same as linking the nodes one after
    another: each new node's own links are written in parallel (no other new node touches
    them), and the links back to existing nodes are grouped by target and applied in node
    order, with different targets on different threads."""
    chosen = np.empty((size, M), np.int64)
    for l in range(topLevel + 1):
        limit = 2 * M if l == 0 else M
        chosenCounts = np.zeros(size, np.int64)
        for slot in prange(size):
            node = first + slot
            if l <= levels[node]:
                found = candCounts[slot, l]
                kept = selectNeighbours(vectors, candIds[slot, l, :found], candDists[slot, l, :found], M)
                chosen[slot, :len(kept)] = kept
                chosenCounts[slot] = len(kept)
                for k in range(len(kept)):
                    links[l, node, k] = kept[k]
                counts[l, node] = len(kept)
        total = chosenCounts.sum()
        keys = np.empty(total, np.int64)
        position = 0
        for slot in range(size):
            for k in range(chosenCounts[slot]):
                keys[position] = chosen[slot, k] * size + slot
                position += 1
        keys.sort()
        starts = [0]
        for i in range(1, total):
            if keys[i] // size != keys[i - 1] // size:
                starts.append(i)
        starts.append(total)
        for g in prange(len(starts) - 1):
            for i in range(starts[g], starts[g + 1]):
                link(vectors, links, counts, l, keys[i] // size, first + keys[i] % size, limit)


def buildBatches(vectors, links, counts, levels, M, efConstruction, workers, batch=64, warmup=2000):
    """Insert the nodes in batches. Each batch finds its neighbours in parallel and is then
    linked in order. Nodes in the same batch do not see each other, which costs very little
    recall when the batch is small next to the graph."""
    n = len(vectors)
    warmup = min(n, warmup)
    entry, topLevel = build(vectors, links, counts, levels, M, efConstruction, warmup)
    visitedPool = np.zeros((workers, n), np.int64)
    candIds = np.empty((batch, links.shape[0], efConstruction), np.int64)
    candDists = np.empty((batch, links.shape[0], efConstruction), np.float64)
    candCounts = np.zeros((batch, links.shape[0]), np.int64)
    for number, first in enumerate(range(warmup, n, batch)):
        size = min(batch, n - first)
        findBatch(vectors, links, counts, levels, first, size, entry, topLevel, efConstruction,
                  visitedPool, number * 100_000, candIds, candDists, candCounts)
        linkBatch(vectors, links, counts, levels, first, size, topLevel, M, candIds, candDists, candCounts)
        for node in range(first, first + size):
            if levels[node] > topLevel:
                entry, topLevel = node, levels[node]
    return entry, topLevel


@njit(cache=True)
def searchRange(vectors, links, counts, queries, start, stop, k, ef, entry, topLevel, result):
    """Search queries[start:stop] and write their neighbours into result."""
    visited = np.zeros(vectors.shape[0], np.int64)
    stamp = 0
    for row in range(start, stop):
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


@njit(cache=True, parallel=True)
def searchParallel(vectors, links, counts, queries, k, ef, entry, topLevel, chunks):
    """Split the queries into chunks and search the chunks on separate threads."""
    result = np.full((queries.shape[0], k), -1, np.int64)
    bounds = np.linspace(0, queries.shape[0], chunks + 1).astype(np.int64)
    for c in prange(chunks):
        searchRange(vectors, links, counts, queries, bounds[c], bounds[c + 1], k, ef,
                    entry, topLevel, result)
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

    def add(self, data, threads=1):
        self.vectors = self._prepare(data)
        rng = np.random.default_rng(self.seed)
        scale = 1 / math.log(self.M)
        levels = (-np.log(1 - rng.random(len(self.vectors))) * scale).astype(np.int64)
        shape = (levels.max() + 1, len(self.vectors))
        self.links = np.full(shape + (2 * self.M,), -1, dtype=np.int32)
        self.counts = np.zeros(shape, dtype=np.int32)
        if threads == 1:
            self.entry, self.topLevel = build(self.vectors, self.links, self.counts, levels,
                                              self.M, self.efConstruction, len(self.vectors))
        else:
            numba.set_num_threads(threads)
            self.entry, self.topLevel = buildBatches(self.vectors, self.links, self.counts, levels,
                                                     self.M, self.efConstruction, threads)

    def memoryBytes(self):
        return self.vectors.nbytes + 4 * int(self.counts.sum())  # 4 bytes per stored link id

    def search(self, queries, k, ef=50, threads=1):
        queries = self._prepare(queries)
        if threads == 1:
            result = np.full((len(queries), k), -1, np.int64)
            searchRange(self.vectors, self.links, self.counts, queries, 0, len(queries), k, ef,
                        self.entry, self.topLevel, result)
            return result
        numba.set_num_threads(threads)
        return searchParallel(self.vectors, self.links, self.counts, queries, k, ef,
                              self.entry, self.topLevel, threads)
