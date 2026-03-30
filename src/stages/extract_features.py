"""
Stage: Feature Extraction
==========================
Reads  : outputs/{dataset}/preprocessed/  (via data_loader)
Writes : outputs/{dataset}/{method}/features/
           ├── train.npy                  ← float32  (N, D)
           ├── train_labels.npy           ← int64    (N,)
           ├── test.npy / val.npy         ← same
           ├── test_labels.npy / val_labels.npy
           └── meta.json                  ← method params, feature_dim, etc.

Design rules enforced:
  ✅ RULE 1  reads ONLY from preprocessed output (never from data/ directly)
  ✅ RULE 2  output dir deleted before run — no partial state
  ✅ RULE 3  method-namespaced: outputs/{dataset}/{method}/features/
  ✅ RULE 4  relative paths via ROOT
  ✅ RULE 5  deterministic — no randomness

Usage:
    python src/stages/extract_features.py --dataset cifar10 --method lbp
    python src/stages/extract_features.py --dataset cifar10 --method lbp --grid 4
    python src/stages/extract_features.py --dataset mnist   --method lbp --grid 2 --no-uniform
    python src/stages/extract_features.py --dataset flowers102 --method lbp
    python src/stages/extract_features.py --dataset cifar10 --method color
    python src/stages/extract_features.py --dataset cifar10 --method color --grid 4 --hist-bins 32
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

from features.lbp import extract_lbp       # noqa: E402
from features.color import extract_color   # noqa: E402

SPLITS_BY_DATASET = {
    "cifar10":    ["train", "test"],
    "mnist":      ["train", "test"],
    "flowers102": ["train", "val", "test"],
}


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def load_preprocessed(dataset: str, split: str):
    """Load preprocessed arrays and metadata."""
    base = ROOT / "outputs" / dataset / "preprocessed"

    img_path   = base / f"{split}.npy"
    label_path = base / f"{split}_labels.npy"
    meta_path  = base / "meta.json"

    assert img_path.exists(), (
        f"[FAIL] {img_path.relative_to(ROOT)} not found.\n"
        f"       Run: python src/stages/preprocess.py --dataset {dataset}"
    )
    assert label_path.exists(), (
        f"[FAIL] {label_path.relative_to(ROOT)} not found.\n"
        f"       Run: python src/stages/preprocess.py --dataset {dataset}"
    )
    assert meta_path.exists(), (
        f"[FAIL] {meta_path.relative_to(ROOT)} not found.\n"
        f"       Run: python src/stages/preprocess.py --dataset {dataset}"
    )

    images = np.load(img_path)
    labels = np.load(label_path)

    with open(meta_path) as f:
        meta = json.load(f)

    return images, labels, meta


def denormalize(images: np.ndarray, meta: dict) -> np.ndarray:
    """
    Reverse the per-channel normalization applied by preprocess.py.

    Preprocessed data:  x_norm = (x - mean) / std
    De-normalized:      x = x_norm * std + mean

    Returns float32 array with pixel values back in [0..255] range.
    """
    if not meta.get("normalized", False):
        return images   # already raw

    mean = np.array(meta["norm_mean"], dtype=np.float32)   # (C,)
    std  = np.array(meta["norm_std"],  dtype=np.float32)   # (C,)

    # Broadcast: (1, C, 1, 1)
    images = images * std[None, :, None, None] + mean[None, :, None, None]
    return images


# ─── METHOD REGISTRY ──────────────────────────────────────────────────────────

def run_lbp(images: np.ndarray, grid: int, uniform: bool, **_) -> np.ndarray:
    """Extract LBP features from a batch of images."""
    return extract_lbp(images, grid=grid, uniform=uniform)


def run_color(images: np.ndarray, grid: int, hist_bins: int, use_hsv: bool, **_) -> np.ndarray:
    """Extract color features from a batch of images."""
    return extract_color(images, grid=grid, hist_bins=hist_bins, use_hsv=use_hsv)


METHOD_FNS = {
    "lbp":   run_lbp,
    "color": run_color,
}


# ─── MAIN STAGE ───────────────────────────────────────────────────────────────

def run_extract_features(
    dataset: str,
    method: str,
    splits: list[str],
    grid: int = 4,
    uniform: bool = True,
    hist_bins: int = 32,
    use_hsv: bool = True,
):
    out_dir = ROOT / "outputs" / dataset / method / "features"

    # ── RULE 2: delete before run ─────────────────────────────────────────────
    reset_dir(out_dir)

    print(f"\n{'='*58}")
    print(f"Feature Extraction: {dataset} | method={method}")
    print(f"Splits: {splits}")
    print(f"Output -> outputs/{dataset}/{method}/features/")
    print(f"{'='*58}")

    if method == "lbp":
        print(f"  LBP params: grid={grid}  uniform={uniform}")
        n_bins = 59 if uniform else 256
        feature_dim = grid * grid * n_bins
        print(f"  Expected feature dim: {feature_dim}")
    elif method == "color":
        print(f"  Color params: grid={grid}  hist_bins={hist_bins}  hsv={use_hsv}")

    t_total = time.perf_counter()

    for split in splits:
        print(f"\n  Processing split: {split}")

        # ── Load preprocessed data ────────────────────────────────────────────
        t0 = time.perf_counter()
        images, labels, meta = load_preprocessed(dataset, split)
        print(f"    Loaded {images.shape[0]:,} images  shape={images.shape}")

        # ── De-normalize: LBP needs raw pixel values [0..255] ─────────────────
        images = denormalize(images, meta)

        # ── Extract features ──────────────────────────────────────────────────
        method_fn = METHOD_FNS[method]
        features = method_fn(
            images, grid=grid, uniform=uniform,
            hist_bins=hist_bins, use_hsv=use_hsv,
        )

        elapsed = time.perf_counter() - t0
        print(f"    Features: shape={features.shape}  dtype={features.dtype}")
        print(f"    Time: {elapsed:.1f}s")

        # ── Save ──────────────────────────────────────────────────────────────
        np.save(out_dir / f"{split}.npy", features)
        np.save(out_dir / f"{split}_labels.npy", labels)
        print(f"    Saved {split}.npy, {split}_labels.npy")

    # ── Save metadata ─────────────────────────────────────────────────────────
    feat_meta = {
        "dataset":     dataset,
        "method":      method,
        "splits":      splits,
        "feature_dim": int(features.shape[1]),
        "source":      f"outputs/{dataset}/preprocessed/",
    }
    if method == "lbp":
        feat_meta.update({
            "grid":    grid,
            "uniform": uniform,
            "n_bins":  59 if uniform else 256,
        })
    elif method == "color":
        feat_meta.update({
            "grid":      grid,
            "hist_bins": hist_bins,
            "use_hsv":   use_hsv,
        })

    with open(out_dir / "meta.json", "w") as f:
        json.dump(feat_meta, f, indent=2)

    total_time = time.perf_counter() - t_total
    print(f"\nFeature extraction complete -> outputs/{dataset}/{method}/features/")
    print(f"Total time: {total_time:.1f}s")

    return feat_meta


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage: Feature Extraction — extract features from preprocessed data",
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=list(SPLITS_BY_DATASET.keys()),
    )
    parser.add_argument(
        "--method", required=True,
        choices=list(METHOD_FNS.keys()),
        help="Feature extraction method",
    )
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="Splits to process (default: all for dataset)",
    )
    # ── Shared args ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--grid", type=int, default=4,
        help="Spatial grid divisions (default: 4 → 4x4 cells)",
    )
    # ── LBP-specific args ─────────────────────────────────────────────────────
    parser.add_argument(
        "--no-uniform", dest="uniform", action="store_false",
        help="Use full 256-bin LBP instead of uniform 59-bin",
    )
    # ── Color-specific args ───────────────────────────────────────────────────
    parser.add_argument(
        "--hist-bins", type=int, default=32,
        help="Bins per channel for color histograms (default: 32)",
    )
    parser.add_argument(
        "--no-hsv", dest="use_hsv", action="store_false",
        help="Disable HSV color space features",
    )
    parser.set_defaults(uniform=True, use_hsv=True)
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    splits = args.splits or SPLITS_BY_DATASET[args.dataset]
    run_extract_features(
        dataset=args.dataset,
        method=args.method,
        splits=splits,
        grid=args.grid,
        uniform=args.uniform,
        hist_bins=args.hist_bins,
        use_hsv=args.use_hsv,
    )
