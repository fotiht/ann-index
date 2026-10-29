import numpy as np


def recall_at_k(found, truth, k):
    """Average fraction of the true k nearest neighbours that were found."""
    hits = sum(len(set(f[:k]) & set(t[:k])) for f, t in zip(found, truth))
    return hits / (len(truth) * k)
