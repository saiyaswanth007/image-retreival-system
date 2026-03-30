"""
Utility: Retrieval Metrics
==========================
Task 9: Evaluation Stage

Provides exact, vectorized operations for computing Precision@K, Recall@K,
and Average Precision (AP@K). Takes boolean 'matches' arrays where True
indicates a relevant retrieved item.
"""

import numpy as np

def precision_at_k(matches: np.ndarray, k: int) -> float:
    """Computes Precision at K."""
    assert len(matches) >= k, f"Cannot compute P@{k} when only {len(matches)} retrieved."
    return float(np.mean(matches[:k]))

def recall_at_k(matches: np.ndarray, k: int, total_relevant: int) -> float:
    """Computes Recall at K."""
    if total_relevant == 0:
        return 0.0
    assert len(matches) >= k, f"Cannot compute R@{k} when only {len(matches)} retrieved."
    return float(np.sum(matches[:k]) / total_relevant)

def average_precision(matches: np.ndarray, k: int, total_relevant: int) -> float:
    """Computes Average Precision up to rank K."""
    if total_relevant == 0:
        return 0.0
        
    assert len(matches) >= k, f"Cannot compute AP@{k} when only {len(matches)} retrieved."
    matches_k = matches[:k]
    
    # Precompute precision at every rank from 1 to K
    # Formula: P@i = cumulative sum of matches up to i, divided by i
    precisions = np.cumsum(matches_k) / np.arange(1, k + 1)
    
    # AP is the sum of Precisions at relevant ranks, exactly normalized by theoretical maximum
    # Standard bounded AP divides by min(total_relevant, k)
    ap = np.sum(precisions * matches_k) / min(total_relevant, k)
    return float(ap)
