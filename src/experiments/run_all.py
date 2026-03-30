"""
Experiment: Run All — Comparative Evaluation
=============================================
Orchestrates extract_features → retrieve → evaluate for multiple methods
on a single dataset, then produces a comparative summary table.

This script does NOT invoke train_model.py — all models must be pre-trained.

Usage:
    python src/experiments/run_all.py --dataset cifar10
    python src/experiments/run_all.py --dataset cifar10 --methods lbp cnn osag --k 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stages.extract_features import run_extract_features, SPLITS_BY_DATASET
from stages.retrieve import run_retrieval
from stages.evaluate import run_evaluation


ALL_METHODS = ["lbp", "color", "nn", "dnn", "cnn", "dsfm", "osag"]


def run_pipeline_for_method(dataset: str, method: str, k: int):
    """Run extract → retrieve → evaluate for a single method."""
    splits = SPLITS_BY_DATASET.get(dataset, ["train", "test"])

    print(f"\n{'#'*60}")
    print(f"# Pipeline: {dataset} / {method}")
    print(f"{'#'*60}")

    # 1. Extract features
    try:
        run_extract_features(dataset=dataset, method=method, splits=splits)
    except Exception as e:
        print(f"[SKIP] Feature extraction failed for {method}: {e}")
        return None

    # 2. Retrieve
    try:
        run_retrieval(dataset=dataset, method=method, k=k)
    except Exception as e:
        print(f"[SKIP] Retrieval failed for {method}: {e}")
        return None

    # 3. Evaluate
    try:
        metrics = run_evaluation(dataset=dataset, method=method)
        return metrics
    except Exception as e:
        print(f"[SKIP] Evaluation failed for {method}: {e}")
        return None


def run_all(dataset: str, methods: list, k: int):
    print(f"\n{'='*60}")
    print(f"Comparative Evaluation: {dataset}")
    print(f"Methods: {methods}")
    print(f"Top-K: {k}")
    print(f"{'='*60}")

    t_total = time.perf_counter()
    all_results = {}

    for method in methods:
        result = run_pipeline_for_method(dataset, method, k)
        if result is not None:
            all_results[method] = result["scores"]

    # Build comparison table
    print(f"\n{'='*60}")
    print(f"COMPARATIVE RESULTS: {dataset}")
    print(f"{'='*60}")

    if not all_results:
        print("[WARN] No methods completed successfully.")
        return

    # Get all metric names from first result
    metric_names = list(next(iter(all_results.values())).keys())

    # Print header
    header = f"{'Metric':<15s}"
    for m in all_results:
        header += f" | {m:>10s}"
    print(header)
    print("-" * len(header))

    # Print rows
    for metric in metric_names:
        row = f"{metric:<15s}"
        for m in all_results:
            val = all_results[m].get(metric, 0.0)
            row += f" | {val:>10.4f}"
        print(row)

    # Save comparison
    out_path = ROOT / "outputs" / dataset / "comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": dataset,
        "k": k,
        "methods": list(all_results.keys()),
        "results": all_results,
        "total_time": time.perf_counter() - t_total,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out_path.relative_to(ROOT)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run full pipeline for all methods")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        help=f"Methods to evaluate (default: {ALL_METHODS})")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(dataset=args.dataset, methods=args.methods, k=args.top_k)
