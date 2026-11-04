"""Equal-recall speed ratios and allocated memory, computed from the committed ann-index logs.

Run with the ML venv (numpy only needed for the level draw):
    ~/jarvis/.venv-ml/bin/python -I equal-recall.py

Method: for each point on one library's curve, the other library's queries per second at the
same recall is found by linear interpolation in recall between its two neighbouring logged
points. Ratios are ours / FAISS, so above 1 means ours is faster. Only recalls inside the other
curve's logged range are used (no extrapolation). Nothing here is re-measured.
"""
import math, re, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent / "results"

def parse(name):
    rows, head = [], ""
    for line in (R / name).read_text().splitlines():
        if not head:
            head = line
        m = re.match(r"\s*(\d+)\s+([\d.]+) /\s+(\d+)\s+([\d.]+) /\s+(\d+)", line)
        if m:
            ef, ro, qo, rf, qf = m.groups()
            rows.append((int(ef), float(ro), int(qo), float(rf), int(qf)))
    return head, rows

def interp(r, pts):
    pts = sorted(pts)
    if r < pts[0][0] or r > pts[-1][0]:
        return None
    for (r0, q0), (r1, q1) in zip(pts, pts[1:]):
        if r0 <= r <= r1:
            return q0 if r1 == r0 else q0 + (q1 - q0) * (r - r0) / (r1 - r0)
    return None

def ratios(rows):
    ours = [(ro, qo) for _, ro, qo, _, _ in rows]
    fa = [(rf, qf) for _, _, _, rf, qf in rows]
    out = []
    for r, q in ours:                      # at ours' recall, what would FAISS do
        qf = interp(r, fa)
        if qf: out.append((r, q / qf, "ours point"))
    for r, q in fa:                        # at FAISS' recall, what would ours do
        qo = interp(r, ours)
        if qo: out.append((r, qo / q, "FAISS point"))
    return sorted(out)

for name in ["faiss_100k.log", "faiss_100k_4threads.log", "faiss_1m.log", "glove_100k.log", "glove_full.log"]:
    head, rows = parse(name)
    print("\n" + name + ":", head)
    for r, ratio, src in ratios(rows):
        print(f"  recall {r:.3f}  ours/FAISS speed {ratio:.2f}  ({src})")

print("\nAllocated memory of our index (vectors + links + counts), seed 0, M = 16:")
M = 16
for label, n, d in [("SIFT 100k", 100000, 128), ("SIFT 1M", 1000000, 128), ("GloVe 100k", 100000, 100), ("GloVe full", 1183514, 100)]:
    rng = np.random.default_rng(0)
    scale = 1 / math.log(M)
    levels = (-np.log(1 - rng.random(n)) * scale).astype(np.int64)
    layers = int(levels.max()) + 1
    vec, links, counts = n * d * 4, layers * n * 2 * M * 4, layers * n * 4
    print(f"  {label}: {layers} layers, vectors {vec/1e6:.1f} MB, links {links/1e6:.1f} MB, counts {counts/1e6:.1f} MB, total {(vec+links+counts)/1e6:.1f} MB")
