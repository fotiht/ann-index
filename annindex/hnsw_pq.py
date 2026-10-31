"""Search an HNSW graph using product quantization codes instead of the raw vectors.

The graph comes from an HNSWFast index built on the raw vectors. This module repeats the
layer search of hnsw_fast.py with a code-based distance, instead of adding a switch to the
shared search loop, so that the raw-vector search keeps its speed.
"""
import heapq

import numba
import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def codeDist(codes, table, i):
    """Approximate squared distance from the query behind table to the vector with code i."""
    total = 0.0
    for p in range(codes.shape[1]):
        total += table[p, codes[i, p]]
    return total


@njit(cache=True)
def searchLayerCodes(codes, table, links, counts, layer, entry, entryDist, ef, visited, stamp):
    visited[entry] = stamp
    toVisit = [(entryDist, entry)]
    best = [(-entryDist, entry)]
    while len(toVisit) > 0:
        dist, current = heapq.heappop(toVisit)
        if len(best) >= ef and dist > -best[0][0]:
            break
        for k in range(counts[layer, current]):
            other = np.int64(links[layer, current, k])
            if visited[other] == stamp:
                continue
            visited[other] = stamp
            otherDist = codeDist(codes, table, other)
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
def searchRangeCodes(codes, tables, links, counts, start, stop, keep, ef, entry, topLevel, result):
    """Write the ids of the `keep` closest codes for queries start to stop - 1 into result."""
    visited = np.zeros(codes.shape[0], np.int64)
    stamp = 0
    for row in range(start, stop):
        table = tables[row]
        ep = np.int64(entry)
        epDist = codeDist(codes, table, ep)
        for l in range(topLevel, 0, -1):
            stamp += 1
            ids, dists = searchLayerCodes(codes, table, links, counts, l, ep, epDist, 1, visited, stamp)
            ep, epDist = ids[0], dists[0]
        stamp += 1
        ids, dists = searchLayerCodes(codes, table, links, counts, 0, ep, epDist, max(ef, keep), visited, stamp)
        for i in range(min(keep, len(ids))):
            result[row, i] = ids[i]


@njit(cache=True, parallel=True)
def searchParallelCodes(codes, tables, links, counts, keep, ef, entry, topLevel, chunks):
    result = np.full((tables.shape[0], keep), -1, np.int64)
    bounds = np.linspace(0, tables.shape[0], chunks + 1).astype(np.int64)
    for c in prange(chunks):
        searchRangeCodes(codes, tables, links, counts, bounds[c], bounds[c + 1], keep, ef,
                         entry, topLevel, result)
    return result


class CompressedHNSW:
    """An HNSWFast graph searched through PQ codes. With rerank=True the ef candidates are
    re-scored with the raw vectors, which are then still needed (for example on disk)."""

    def __init__(self, index, quantizer):
        self.index = index
        self.quantizer = quantizer
        self.codes = quantizer.encode(index.vectors)
        self.rawVectors = index.vectors

    def moveVectorsToDisk(self, path):
        """Save the raw vectors to a .npy file and read them from disk from now on, one
        memory-mapped row at a time. Only rerank=True searches touch them."""
        np.save(path, self.index.vectors)
        self.rawVectors = np.load(path, mmap_mode="r")

    def memoryBytes(self):
        """The graph, the codes and the codebooks. The raw vectors are not counted."""
        links = 4 * int(self.index.counts.sum())
        return links + self.codes.nbytes + self.quantizer.memoryBytes()

    def search(self, queries, k, ef=50, threads=1, rerank=False):
        queries = self.index._prepare(queries)
        tables = self.quantizer.distanceTables(queries)
        keep = max(ef, k) if rerank else k
        graph = (self.index.links, self.index.counts)
        if threads == 1:
            found = np.full((len(queries), keep), -1, np.int64)
            searchRangeCodes(self.codes, tables, *graph, 0, len(queries), keep, ef,
                             self.index.entry, self.index.topLevel, found)
        else:
            numba.set_num_threads(threads)
            found = searchParallelCodes(self.codes, tables, *graph, keep, ef,
                                        self.index.entry, self.index.topLevel, threads)
        return self._rerank(queries, found, k) if rerank else found

    def _rerank(self, queries, candidates, k):
        result = np.full((len(queries), k), -1, np.int64)
        for row, query in enumerate(queries):
            ids = candidates[row][candidates[row] >= 0]
            diff = self.rawVectors[ids] - query
            order = np.argsort(np.einsum("ij,ij->i", diff, diff))[:k]
            result[row, :len(order)] = ids[order]
        return result
