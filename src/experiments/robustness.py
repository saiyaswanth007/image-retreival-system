"""
Experiment: Robustness to Transformations
==========================================
Applies controlled augmentations to test images, re-extracts features
using a frozen trained model, and measures mAP degradation per augmentation.

Augmentations:
  - rotation (90°, 180°, 270°)
  - horizontal flip
  - Gaussian noise (σ=0.1)
  - brightness shift (+0.2)

Usage:
    python src/experiments/robustness.py --dataset cifar10 --method cnn
    python src/experiments/robustness.py --dataset cifar10 --method osag --k 10
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stages.extract_features import load_preprocessed, denormalize, run_neural
from utils.index import RetrievalIndex
from utils.metrics import precision_at_k, recall_at_k, average_precision


# ─── AUGMENTATIONS ────────────────────────────────────────────────────────────

def augment_rotate90(images: np.ndarray) -> np.ndarray:
    """Rotate all images 90° clockwise. (N, C, H, W)"""
    return np.rot90(images, k=-1, axes=(2, 3)).copy()

def augment_rotate180(images: np.ndarray) -> np.ndarray:
    return np.rot90(images, k=2, axes=(2, 3)).copy()

def augment_rotate270(images: np.ndarray) -> np.ndarray:
    return np.rot90(images, k=-3, axes=(2, 3)).copy()

def augment_hflip(images: np.ndarray) -> np.ndarray:
    """Horizontal flip."""
    return np.flip(images, axis=3).copy()

def augment_noise(images: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Add Gaussian noise."""
    noise = np.random.randn(*images.shape).astype(np.float32) * sigma
    return (images + noise).astype(np.float32)

def augment_brightness(images: np.ndarray, delta: float = 0.2) -> np.ndarray:
    """Shift brightness."""
    return (images + delta).astype(np.float32)


AUGMENTATIONS = {
    "clean":       lambda x: x,
    "rotate_90":   augment_rotate90,
    "rotate_180":  augment_rotate180,
    "rotate_270":  augment_rotate270,
    "hflip":       augment_hflip,
    "noise_0.1":   lambda x: augment_noise(x, 0.1),
    "brightness":  lambda x: augment_brightness(x, 0.2),
}


# ─── EVALUATION HELPER ───────────────────────────────────────────────────────

def evaluate_features(q_feats, q_labels, g_feats, g_labels, k: int):
    """Build FAISS index from gallery, search queries, compute mAP@K."""
    feature_dim = q_feats.shape[1]
    index = RetrievalIndex(feature_dim=feature_dim)
    index.add(g_feats)

    distances, indices = index.search(q_feats, k=k)

    unique_labels, counts = np.unique(g_labels, return_counts=True)
    label_to_count = dict(zip(unique_labels, counts))

    ap_list = []
    p_list = []
    r_list = []
    for i in range(len(q_labels)):
        matches = (g_labels[indices[i]] == q_labels[i])
        total_rel = label_to_count.get(q_labels[i], 0)
        ap_list.append(average_precision(matches, k, total_rel))
        p_list.append(precision_at_k(matches, k))
        r_list.append(recall_at_k(matches, k, total_rel))

    return {
        f"mAP@{k}": float(np.mean(ap_list)),
        f"Precision@{k}": float(np.mean(p_list)),
        f"Recall@{k}": float(np.mean(r_list)),
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run_robustness(dataset: str, method: str, k: int = 5):
    out_dir = ROOT / "outputs" / dataset / method / "robustness"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"\n{'='*60}")
    print(f"Robustness Analysis: {dataset} | method={method}")
    print(f"{'='*60}")

    t_total = time.perf_counter()

    # Load gallery (train) — features already extracted
    gallery_path = ROOT / "outputs" / dataset / method / "features"
    assert (gallery_path / "train.npy").exists(), \
        f"[FAIL] Gallery features missing. Run extract_features first!"

    g_feats = np.load(gallery_path / "train.npy")
    g_labels = np.load(gallery_path / "train_labels.npy")
    print(f"  Gallery: {g_feats.shape[0]:,} vectors, dim={g_feats.shape[1]}")

    # Load raw test images (we will augment these)
    test_images, test_labels, meta = load_preprocessed(dataset, "test")
    test_images = denormalize(test_images, meta)
    print(f"  Queries: {test_images.shape[0]:,} images")

    # Evaluate each augmentation
    results = {}
    for aug_name, aug_fn in AUGMENTATIONS.items():
        print(f"\n  Augmentation: {aug_name}")
        augmented = aug_fn(test_images)

        # Re-extract features using the frozen model
        q_feats = run_neural(augmented, method=method, dataset=dataset)

        # Evaluate
        scores = evaluate_features(q_feats, test_labels, g_feats, g_labels, k)
        results[aug_name] = scores
        print(f"    mAP@{k}: {scores[f'mAP@{k}']:.4f}")

    # Summary table
    clean_map = results["clean"][f"mAP@{k}"]
    print(f"\n{'='*60}")
    print(f"ROBUSTNESS SUMMARY (baseline mAP@{k} = {clean_map:.4f})")
    print(f"{'='*60}")
    print(f"{'Augmentation':<15s} | {'mAP@K':>8s} | {'Delta':>8s}")
    print("-" * 38)
    for aug_name, scores in results.items():
        val = scores[f"mAP@{k}"]
        delta = val - clean_map
        print(f"{aug_name:<15s} | {val:>8.4f} | {delta:>+8.4f}")

    # Save
    summary = {
        "dataset": dataset,
        "method": method,
        "k": k,
        "augmentations": results,
        "total_time": time.perf_counter() - t_total,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved -> outputs/{dataset}/{method}/robustness/results.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Robustness to Transformations Analysis")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("-k", "--top-k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_robustness(dataset=args.dataset, method=args.method, k=args.top_k)
