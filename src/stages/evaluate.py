"""
Stage: Evaluation
==================
Reads  : outputs/{dataset}/{method}/retrieval/ (indices.npy, meta.json)
         outputs/{dataset}/{method}/features/ (for labels)
Writes : outputs/{dataset}/{method}/evaluate/
           └── metrics.json       ← full metric dictionary

Design rules enforced:
  ✅ RULE 1  reads ONLY from exact previous outputs
  ✅ RULE 2  output dir deleted before run
  ✅ RULE 3  method-namespaced: outputs/{dataset}/{method}/evaluate/
  ✅ RULE 4  relative paths via ROOT
  ✅ RULE 5  deterministic tracking

Usage:
    python src/stages/evaluate.py --dataset cifar10 --method osag
    python src/stages/evaluate.py --dataset cifar10 --method lbp
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# ─── PATHS ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # src/stages/ → project root
sys.path.insert(0, str(ROOT / "src"))

from utils.metrics import precision_at_k, recall_at_k, average_precision


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def load_labels(dataset: str, method: str, split: str):
    """Load labels generated safely during feature extraction."""
    label_path = ROOT / "outputs" / dataset / method / "features" / f"{split}_labels.npy"
    assert label_path.exists(), f"[FAIL] Labels missing: {label_path}"
    return np.load(label_path)


def run_evaluation(dataset: str, method: str):
    retrieval_dir = ROOT / "outputs" / dataset / method / "retrieval"
    out_dir = ROOT / "outputs" / dataset / method / "evaluate"

    # ── RULE 2: delete before run ─────────────────────────────────────────────
    reset_dir(out_dir)

    print(f"\n{'='*58}")
    print(f"Evaluation Stage: {dataset} | method={method}")
    print(f"Output -> outputs/{dataset}/{method}/evaluate/")
    print(f"{'='*58}")

    t_total = time.perf_counter()

    # ── 1. Load configuration and indices ─────────────────────────────────────
    meta_path = retrieval_dir / "meta.json"
    indices_path = retrieval_dir / "indices.npy"
    
    assert meta_path.exists(), f"[FAIL] Retrieval meta.json missing. Run retrieve.py first!"
    assert indices_path.exists(), f"[FAIL] Retrieval indices.npy missing. Run retrieve.py first!"

    with open(meta_path) as f:
        meta = json.load(f)
        
    query_split = meta["query_split"]
    gallery_split = meta["gallery_split"]
    retrieved_k = meta["k"]

    indices = np.load(indices_path)

    print("\n[1/3] Loading ground-truth labels...")
    q_labels = load_labels(dataset, method, query_split)
    g_labels = load_labels(dataset, method, gallery_split)
    
    # Determine the total occurrences of each label in the gallery
    # Used for rigorous Recall and Average Precision bounds
    unique_labels, counts = np.unique(g_labels, return_counts=True)
    label_to_count = dict(zip(unique_labels, counts))

    print(f"\n[2/3] Computing geometric metrics up to K={retrieved_k}...")
    
    k_vals = [1, 5, 10, 50, 100]
    k_vals = [k for k in k_vals if k <= retrieved_k]
    if retrieved_k not in k_vals:
        k_vals.append(retrieved_k)
        
    results = {
        f"mAP@{k}": [] for k in k_vals
    }
    for k in k_vals:
        results[f"Precision@{k}"] = []
        results[f"Recall@{k}"] = []

    N_queries = len(q_labels)
    for i in range(N_queries):
        true_label = q_labels[i]
        retrieved_idx = indices[i]
        
        # Boolean relevance array shape (K,)
        matches = (g_labels[retrieved_idx] == true_label)
        total_relevant = label_to_count.get(true_label, 0)
        
        for k in k_vals:
            results[f"Precision@{k}"].append(precision_at_k(matches, k))
            results[f"Recall@{k}"].append(recall_at_k(matches, k, total_relevant))
            results[f"mAP@{k}"].append(average_precision(matches, k, total_relevant))

    print("\n[3/3] Aggregating Final Scores...")
    final_metrics = {
        "dataset": dataset,
        "method": method,
        "query_split": query_split,
        "gallery_split": gallery_split,
        "queries_evaluated": N_queries,
        "total_time": time.perf_counter() - t_total,
        "scores": {}
    }

    print()
    for metric_name, values in results.items():
        mean_val = float(np.mean(values))
        final_metrics["scores"][metric_name] = mean_val
        print(f"  {metric_name:<15s}: {mean_val:.4f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\nEvaluation complete -> outputs/{dataset}/{method}/evaluate/metrics.json")
    print(f"Total time: {final_metrics['total_time']:.2f}s")
    return final_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage: Evaluate Retrieval Performance (mAP, P@K, R@K)",
    )
    parser.add_argument(
        "--dataset", required=True,
    )
    parser.add_argument(
        "--method", required=True,
        help="Feature extraction method to evaluate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(dataset=args.dataset, method=args.method)
