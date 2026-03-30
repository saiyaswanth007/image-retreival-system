"""
Stage: Retrieval
=================
Reads  : outputs/{dataset}/{method}/features/ (via numpy)
Writes : outputs/{dataset}/{method}/retrieval/
           ├── distances.npy   ← float32 (N_queries, K)
           ├── indices.npy     ← int64   (N_queries, K)
           └── meta.json       ← configuration tracking

Design rules enforced:
  ✅ RULE 1  reads ONLY from features/ output
  ✅ RULE 2  output dir deleted before run
  ✅ RULE 3  method-namespaced: outputs/{dataset}/{method}/retrieval/
  ✅ RULE 4  relative paths via ROOT
  ✅ RULE 5  deterministic — exact FAISS flat L2 search

Usage:
    python src/stages/retrieve.py --dataset cifar10 --method osag
    python src/stages/retrieve.py --dataset cifar10 --method lbp --k 10
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

# Allow imports from src/
sys.path.insert(0, str(ROOT / "src"))

from utils.index import RetrievalIndex


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def load_features(dataset: str, method: str, split: str):
    """Load extracted features and labels."""
    base = ROOT / "outputs" / dataset / method / "features"

    feat_path  = base / f"{split}.npy"
    label_path = base / f"{split}_labels.npy"

    assert feat_path.exists(), (
        f"[FAIL] {feat_path.relative_to(ROOT)} missing.\n"
        f"       Run feature extraction stage first!"
    )
    assert label_path.exists(), f"[FAIL] {label_path.relative_to(ROOT)} missing."

    features = np.load(feat_path)
    labels = np.load(label_path)

    return features, labels


# ─── MAIN STAGE ───────────────────────────────────────────────────────────────

def run_retrieval(
    dataset: str,
    method: str,
    k: int = 5,
    query_split: str = "test",
    gallery_split: str = "train"
):
    out_dir = ROOT / "outputs" / dataset / method / "retrieval"

    # ── RULE 2: delete before run ─────────────────────────────────────────────
    reset_dir(out_dir)

    print(f"\n{'='*58}")
    print(f"Retrieval Stage: {dataset} | method={method}")
    print(f"Splits: queries={query_split} vs gallery={gallery_split}")
    print(f"Top-K: {k}")
    print(f"Output -> outputs/{dataset}/{method}/retrieval/")
    print(f"{'='*58}")

    t_total = time.perf_counter()

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    print(f"\n[1/3] Loading query ({query_split}) and gallery ({gallery_split}) features...")
    q_feats, q_labels = load_features(dataset, method, query_split)
    g_feats, g_labels = load_features(dataset, method, gallery_split)
    
    assert q_feats.shape[1] == g_feats.shape[1], (
        f"[FAIL] Dimension mismatch! Query: {q_feats.shape[1]}, Gallery: {g_feats.shape[1]}"
    )
    feature_dim = q_feats.shape[1]
    
    print(f"  Gallery: {g_feats.shape[0]:,} items | dim={feature_dim}")
    print(f"  Queries: {q_feats.shape[0]:,} items | dim={feature_dim}")

    # ── 2. Initialize and Populate FAISS Index ────────────────────────────────
    print("\n[2/3] Building FAISS Retrieval Index...")
    t_index = time.perf_counter()
    index = RetrievalIndex(feature_dim=feature_dim)
    
    index.add(g_feats)
    
    print(f"  Index mapped {index.total_count:,} gallery vectors. [{(time.perf_counter() - t_index):.2f}s]")

    # ── 3. Execute Top-K Search ───────────────────────────────────────────────
    print(f"\n[3/3] Searching for Top-{k} Matches...")
    t_search = time.perf_counter()
    
    distances, indices = index.search(q_feats, k=k)
    
    print(f"  Search completed. Shape: {distances.shape} [{(time.perf_counter() - t_search):.2f}s]")

    # ── Save outputs ──────────────────────────────────────────────────────────
    np.save(out_dir / "distances.npy", distances)
    np.save(out_dir / "indices.npy", indices)

    meta = {
        "dataset":       dataset,
        "method":        method,
        "k":             k,
        "query_split":   query_split,
        "gallery_split": gallery_split,
        "feature_dim":   feature_dim,
        "gallery_size":  int(g_feats.shape[0]),
        "query_size":    int(q_feats.shape[0]),
        "source":        f"outputs/{dataset}/{method}/features/",
        "total_time":    time.perf_counter() - t_total,
    }
    
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nRetrieval complete.")
    print(f"Saved: distances.npy, indices.npy, meta.json")
    print(f"Total time: {meta['total_time']:.2f}s")
    
    return meta


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage: Retrieval — Compute Top-K ranking via FAISS",
    )
    parser.add_argument(
        "--dataset", required=True,
    )
    parser.add_argument(
        "--method", required=True,
        help="Feature extraction method to evaluate",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=5,
        help="Number of nearest neighbors to retrieve (default: 5)",
    )
    parser.add_argument(
        "--query-split", default="test",
        help="Split to use for querying (default: test)",
    )
    parser.add_argument(
        "--gallery-split", default="train",
        help="Split against which queries are matched (default: train)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    run_retrieval(
        dataset=args.dataset,
        method=args.method,
        k=args.top_k,
        query_split=args.query_split,
        gallery_split=args.gallery_split,
    )
