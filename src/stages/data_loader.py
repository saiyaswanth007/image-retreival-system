"""
Data Access Layer
==================
Single entry point for all downstream stages.

Usage:
    from src.data_loader import load_data

    images, labels = load_data(dataset="cifar10", split="train")
    images, labels = load_data(dataset="mnist",   split="test")

Returns:
    images : np.ndarray  float32  (N, C, H, W)
    labels : np.ndarray  int64    (N,)

Design rules enforced:
  ✅ RULE 1  reads ONLY from outputs/{dataset}/preprocessed/ — never from data/ directly
  ✅ RULE 4  relative paths only
  ✅ RULE 5  deterministic — no shuffling, no randomness here (that belongs in training)
  ✅ STRICT  fails hard if preprocessed output is missing
"""

import json
from pathlib import Path

import numpy as np

# ─── ROOT ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # src/stages/ → project root


# ─── CORE ─────────────────────────────────────────────────────────────────────

def load_data(
    dataset: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed images and labels for one split.

    Args:
        dataset : "cifar10" | "mnist" | "flowers102"
        split   : "train"   | "val"   | "test"

    Returns:
        images  : float32 array  (N, C, H, W)
        labels  : int64 array    (N,)

    Raises:
        AssertionError if outputs/{dataset}/preprocessed/{split}.npy is missing.
        Run preprocess.py first.
    """
    base = ROOT / "outputs" / dataset / "preprocessed"

    img_path   = base / f"{split}.npy"
    label_path = base / f"{split}_labels.npy"

    # ── Fail hard — no fallback, no silent recompute ──────────────────────────
    assert img_path.exists(), (
        f"[FAIL] {img_path.relative_to(ROOT)} not found.\n"
        f"       Run: python preprocess.py --dataset {dataset}"
    )
    assert label_path.exists(), (
        f"[FAIL] {label_path.relative_to(ROOT)} not found.\n"
        f"       Run: python preprocess.py --dataset {dataset}"
    )

    images = np.load(img_path)
    labels = np.load(label_path)

    assert images.ndim == 4,       f"[FAIL] Expected NCHW array, got shape {images.shape}"
    assert len(images) == len(labels), "[FAIL] Image/label count mismatch"

    return images, labels


# ─── EXTRAS ───────────────────────────────────────────────────────────────────

def load_class_names(dataset: str) -> dict[int, str]:
    """Return {label_int: class_name} from the preprocessed metadata."""
    path = ROOT / "outputs" / dataset / "preprocessed" / "class_names.json"
    assert path.exists(), (
        f"[FAIL] {path.relative_to(ROOT)} not found. Run preprocess.py first."
    )
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_meta(dataset: str) -> dict:
    """Return preprocessing metadata (size, norm stats, layout, etc.)."""
    path = ROOT / "outputs" / dataset / "preprocessed" / "meta.json"
    assert path.exists(), (
        f"[FAIL] {path.relative_to(ROOT)} not found. Run preprocess.py first."
    )
    with open(path) as f:
        return json.load(f)