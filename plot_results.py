"""Plot recall against queries per second from the logs in results/ (written by faiss_compare.py)."""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROW = re.compile(r"^(\d+)\s+([\d.]+) /\s*(\d+)\s+([\d.]+) /\s*(\d+)")
RUNS = {
    "SIFT, 100,000 vectors": "faiss_100k.log",
    "SIFT, 1,000,000 vectors": "faiss_1m.log",
    "GloVe, 100,000 vectors": "glove_100k.log",
    "GloVe, 1,183,514 vectors": "glove_full.log",
}


def readRows(results, names):
    """Rows (ef, our recall, our qps, FAISS recall, FAISS qps) from one or more logs, one row per ef."""
    rows = {}
    for name in names.split():
        for line in (results / name).read_text().splitlines():
            match = ROW.match(line)
            if match:
                row = tuple(float(x) for x in match.groups())
                rows[row[0]] = row
    return [rows[ef] for ef in sorted(rows)]


def plot(results, output):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey=False)
    for ax, (title, name) in zip(axes.flat, RUNS.items()):
        rows = readRows(results, name)
        for label, column, style in (("ours (Numba)", 1, "o-"), ("FAISS", 3, "s--")):
            ax.plot([r[column] for r in rows], [r[column + 1] for r in rows], style, label=label)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("recall@10")
        ax.set_ylabel("queries per second (1 thread)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=130)


if __name__ == "__main__":
    plot(Path("results"), sys.argv[1] if len(sys.argv) > 1 else "results/recall_vs_qps.png")
