import heapq
import math

import numpy as np


class HNSW:
    """Hierarchical Navigable Small World index (Malkov and Yashunin, 2016)."""

    def __init__(self, M=16, efConstruction=100, metric="euclidean", seed=0):
        self.M = M
        self.maxLinks0 = 2 * M  # layer 0 may keep more links than upper layers
        self.efConstruction = efConstruction
        self.metric = metric
        self.levelScale = 1 / math.log(M)
        self.rng = np.random.default_rng(seed)
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self.layers = []  # layers[l][node] = list of neighbour ids
        self.entry = None

    def _prepare(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.metric == "cosine":
            x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        elif self.metric != "euclidean":
            raise ValueError(f"unknown metric: {self.metric}")
        return x

    def _dist(self, query, ids):
        diff = self.vectors[ids] - query
        return np.einsum("ij,ij->i", diff, diff)

    def add(self, data):
        self.vectors = self._prepare(data)
        for node in range(len(self.vectors)):
            self._insert(node)

    def _insert(self, node):
        query = self.vectors[node]
        level = int(-math.log(1 - self.rng.random()) * self.levelScale)
        if self.entry is None:
            self.entry = node
            self.layers = [{} for _ in range(level + 1)]
            for layer in self.layers:
                layer[node] = []
            return

        topLevel = len(self.layers) - 1
        while len(self.layers) <= level:
            self.layers.append({})
        for l in range(level + 1):
            self.layers[l][node] = []

        nearest = [self.entry]
        for l in range(topLevel, level, -1):
            nearest = self._searchLayer(query, nearest, 1, l)[:1]
        for l in range(min(level, topLevel), -1, -1):
            found = self._searchLayer(query, nearest, self.efConstruction, l)
            for other in self._selectNeighbours(node, found, self.M):
                self._link(node, other, l)
                self._link(other, node, l)
            nearest = found

        if level > topLevel:
            self.entry = node

    def _searchLayer(self, query, entryPoints, ef, l):
        """Best-first search of one layer. Returns up to ef ids, closest first."""
        graph = self.layers[l]
        visited = set(entryPoints)
        dists = self._dist(query, entryPoints)
        toVisit = list(zip(dists, entryPoints))  # min-heap: closest first
        heapq.heapify(toVisit)
        best = [(-d, i) for d, i in toVisit]  # max-heap of the ef closest
        heapq.heapify(best)
        while toVisit:
            d, current = heapq.heappop(toVisit)
            if d > -best[0][0] and len(best) >= ef:
                break
            fresh = [n for n in graph[current] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for nd, n in zip(self._dist(query, fresh), fresh):
                if len(best) < ef or nd < -best[0][0]:
                    heapq.heappush(toVisit, (nd, n))
                    heapq.heappush(best, (-nd, n))
                    if len(best) > ef:
                        heapq.heappop(best)
        return [i for _, i in sorted((-d, i) for d, i in best)]

    def _selectNeighbours(self, node, candidates, count):
        """Keep a candidate only if it is closer to the node than to any already kept."""
        query = self.vectors[node]
        candidates = [c for c in candidates if c != node]
        kept = []
        for c in candidates:  # closest first
            if len(kept) == count:
                break
            dNode = self._dist(query, [c])[0]
            if not kept or dNode < self._dist(self.vectors[c], kept).min():
                kept.append(c)
        # top up with the nearest rejected ones if the heuristic kept too few
        for c in candidates:
            if len(kept) == count:
                break
            if c not in kept:
                kept.append(c)
        return kept

    def _link(self, a, b, l):
        links = self.layers[l][a]
        links.append(b)
        limit = self.maxLinks0 if l == 0 else self.M
        if len(links) > limit:
            order = np.argsort(self._dist(self.vectors[a], links))
            self.layers[l][a] = self._selectNeighbours(a, [links[i] for i in order], limit)

    def search(self, queries, k, ef=50):
        queries = self._prepare(queries)
        result = np.full((len(queries), k), -1, dtype=np.int64)
        for row, query in enumerate(queries):
            nearest = [self.entry]
            for l in range(len(self.layers) - 1, 0, -1):
                nearest = self._searchLayer(query, nearest, 1, l)
            found = self._searchLayer(query, nearest, max(ef, k), 0)[:k]
            result[row, :len(found)] = found
        return result
