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
        """Squared distances from query to the stored vectors `ids`."""
        return self.sqNorms[ids] - 2 * (self.vectors[ids] @ query) + query @ query

    def add(self, data):
        self.vectors = self._prepare(data)
        self.sqNorms = np.einsum("ij,ij->i", self.vectors, self.vectors)
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
        dists = self._dist(query, entryPoints).tolist()
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
            for nd, n in zip(self._dist(query, fresh).tolist(), fresh):
                if len(best) < ef or nd < -best[0][0]:
                    heapq.heappush(toVisit, (nd, n))
                    heapq.heappush(best, (-nd, n))
                    if len(best) > ef:
                        heapq.heappop(best)
        return [i for _, i in sorted((-d, i) for d, i in best)]

    def _selectNeighbours(self, node, candidates, count):
        """Keep a candidate only if it is closer to the node than to any already kept."""
        candidates = [c for c in candidates if c != node]
        if not candidates:
            return []
        points = self.vectors[candidates]
        toNode = self._dist(self.vectors[node], candidates)
        # squared distances between every pair of candidates, computed in one product
        norms = self.sqNorms[candidates]
        between = norms[:, None] + norms[None, :] - 2 * points @ points.T
        kept = []  # positions in `candidates`, closest first
        for i in range(len(candidates)):  # closest first
            if len(kept) == count:
                break
            if not kept or toNode[i] < between[i, kept].min():
                kept.append(i)
        # top up with the nearest rejected ones if the heuristic kept too few
        if len(kept) < count:
            chosen = set(kept)
            kept += [i for i in range(len(candidates)) if i not in chosen][:count - len(kept)]
        return [candidates[i] for i in kept]

    def _link(self, a, b, l):
        links = self.layers[l][a]
        links.append(b)
        limit = self.maxLinks0 if l == 0 else self.M
        if len(links) > limit:
            order = np.argsort(self._dist(self.vectors[a], links))
            self.layers[l][a] = self._selectNeighbours(a, [links[i] for i in order], limit)

    def memoryBytes(self):
        links = sum(len(l) for layer in self.layers for l in layer.values())
        return self.vectors.nbytes + 8 * links  # 8 bytes per stored link id

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
