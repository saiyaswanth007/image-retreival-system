"""
Stage: Preprocess
==================
Reads  : data/{dataset}/{split}/              (ImageFolder structure)
Writes : outputs/{dataset}/preprocessed/
           ├── train.npy                      ← float32, shape (N, C, H, W)
           ├── train_labels.npy               ← int64,   shape (N,)
           ├── test.npy / val.npy             ← same
           ├── test_labels.npy / val_labels.npy
           ├── class_names.json               ← index → class name
           └── meta.json                      ← resize, norm, channel, split sizes

Design rules enforced:
  ✅ RULE 1  reads ONLY from data/ (raw source, never a method output)
  ✅ RULE 2  output dir deleted before run — no partial state
  ✅ RULE 4  relative paths only
  ✅ RULE 5  deterministic — fixed order, no shuffling (shuffling is a training concern)
  ✅ STRICT  fails hard if data/{dataset} missing

Processing applied:
  1. Resize          → target_size (default 64×64), Lanczos resampling
  2. Channel align   → MNIST L/grayscale → RGB (3-channel) so all datasets are uniform
  3. Normalize       → per-channel mean/std computed from train split
                       (test/val use train statistics — no leakage)
  4. Output dtype    → float32, layout NCHW (PyTorch convention)

Usage:
    python preprocess.py --dataset cifar10
    python preprocess.py --dataset mnist  --size 64
    python preprocess.py --dataset flowers102 --size 128
    python preprocess.py --dataset cifar10 --size 32 --no-normalize
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ─── PATHS ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # src/stages/ → project root

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


def load_split(split_dir: Path, size: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Walk an ImageFolder-structured directory, resize every image, return:
        images : float32 array  (N, 3, H, W)  — channel-first, pixels in [0, 255]
        labels : int64 array    (N,)
        classes: list[str]      sorted class names (index = label integer)
    """
    class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    assert class_dirs, f"[FAIL] No class sub-folders found in {split_dir}"

    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
    classes      = [d.name for d in class_dirs]

    images_list = []
    labels_list = []

    for class_dir in tqdm(class_dirs, desc=f"    {split_dir.name}", ncols=80):
        idx   = class_to_idx[class_dir.name]
        paths = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))

        for p in paths:
            with Image.open(p) as img:
                # ── Channel alignment: everything → RGB ────────────────────
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # ── Resize with Lanczos (best quality downscaler) ──────────
                if img.size != (size, size):
                    img = img.resize((size, size), Image.LANCZOS)

                arr = np.array(img, dtype=np.float32)   # H×W×C  [0..255]

            images_list.append(arr)
            labels_list.append(idx)

    images = np.stack(images_list, axis=0)          # N×H×W×C
    images = images.transpose(0, 3, 1, 2)           # N×C×H×W  (PyTorch NCHW)
    labels = np.array(labels_list, dtype=np.int64)

    return images, labels, classes


def compute_channel_stats(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std over the TRAIN split only.
    images: float32  (N, C, H, W)  values in [0..255]
    Returns mean, std — both shape (C,), values in [0..255] scale.
    """
    # Reshape to (C, N*H*W) for vectorised stats
    c = images.shape[1]
    flat = images.transpose(1, 0, 2, 3).reshape(c, -1)   # C × (N*H*W)
    mean = flat.mean(axis=1)
    std  = flat.std(axis=1)
    std  = np.where(std < 1e-6, 1.0, std)   # guard zero-std channels
    return mean, std


def normalize(images: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Standardize: (x - mean) / std
    mean/std shape: (C,)  → broadcast over (N, C, H, W)
    """
    return (images - mean[None, :, None, None]) / std[None, :, None, None]


# ─── MAIN STAGE ───────────────────────────────────────────────────────────────

def run_preprocess(dataset: str, splits: list[str], size: int, do_normalize: bool):
    data_dir = ROOT / "data" / dataset

    # ── STRICT: fail hard if input missing ────────────────────────────────────
    assert data_dir.exists(), (
        f"[FAIL] data/{dataset}/ not found. Run download_datasets.py first."
    )

    out_dir = ROOT / "outputs" / dataset / "preprocessed"

    # ── RULE 2: delete output dir before run ──────────────────────────────────
    reset_dir(out_dir)

    print(f"\n{'='*55}")
    print(f"Preprocess: {dataset}  |  size={size}×{size}  normalize={do_normalize}")
    print(f"Splits: {splits}")
    print(f"Output → outputs/{dataset}/preprocessed/")
    print(f"{'='*55}")

    saved_classes = None
    mean = std = None

    for split in splits:
        split_dir = data_dir / split
        assert split_dir.exists(), f"[FAIL] data/{dataset}/{split}/ missing."

        print(f"\n  Processing split: {split}")
        images, labels, classes = load_split(split_dir, size)

        # Validate class consistency across splits
        if saved_classes is None:
            saved_classes = classes
        else:
            assert classes == saved_classes, (
                f"[FAIL] Class mismatch between splits: {classes} vs {saved_classes}"
            )

        # ── Compute normalization stats from TRAIN only (no leakage) ──────────
        if do_normalize:
            if split == "train":
                print("    Computing channel mean/std from train split ...")
                mean, std = compute_channel_stats(images)
                print(f"    mean={np.round(mean, 2)}  std={np.round(std, 2)}")
            else:
                # non-train splits: reuse train stats
                if mean is None:
                    # edge case: user runs val/test without train in splits
                    print("    ⚠️  Train split not in run — computing stats from this split.")
                    mean, std = compute_channel_stats(images)

            images = normalize(images, mean, std)

        # ── Save arrays ───────────────────────────────────────────────────────
        img_path   = out_dir / f"{split}.npy"
        label_path = out_dir / f"{split}_labels.npy"

        np.save(img_path,   images)
        np.save(label_path, labels)

        print(f"    Saved {split}.npy        shape={images.shape}  dtype={images.dtype}")
        print(f"    Saved {split}_labels.npy shape={labels.shape}")

    # ── Save class names ───────────────────────────────────────────────────────
    class_map = {i: name for i, name in enumerate(saved_classes)}
    with open(out_dir / "class_names.json", "w") as f:
        json.dump(class_map, f, indent=2)

    # ── Save metadata ──────────────────────────────────────────────────────────
    meta = {
        "dataset":     dataset,
        "splits":      splits,
        "size":        size,
        "channels":    3,
        "layout":      "NCHW",
        "dtype":       "float32",
        "normalized":  do_normalize,
        "norm_mean":   mean.tolist() if mean is not None else None,
        "norm_std":    std.tolist()  if std  is not None else None,
        "n_classes":   len(saved_classes),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Preprocessing complete → outputs/{dataset}/preprocessed/")
    print(f"   meta.json written — downstream stages must read this for norm stats.")

    return meta


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 5 — Preprocess: resize, channel-align, normalize → .npy"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=list(SPLITS_BY_DATASET.keys()),
    )
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="Splits to process (default: all for dataset)"
    )
    parser.add_argument(
        "--size", type=int, default=64,
        help="Target image size in pixels, square (default: 64)"
    )
    parser.add_argument(
        "--no-normalize", dest="normalize", action="store_false",
        help="Skip mean/std normalization (useful for LBP / color feature stages)"
    )
    parser.set_defaults(normalize=True)
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    splits = args.splits or SPLITS_BY_DATASET[args.dataset]
    run_preprocess(args.dataset, splits, args.size, args.normalize)