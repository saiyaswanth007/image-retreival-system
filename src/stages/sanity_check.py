"""
Stage: Data Sanity Check
=========================
Reads  : data/{dataset}/{split}/          (ImageFolder structure)
Writes : outputs/{dataset}/sanity/
           ├── report.json               ← full stats (class counts, resolutions)
           ├── class_balance.png         ← bar chart per split
           ├── resolution_scatter.png    ← W×H scatter coloured by split
           └── samples/
               └── {split}_grid.png     ← 5×5 random sample grid per split

Design rules enforced:
  ✅ RULE 1  reads ONLY from data/  (raw source, never another stage's output)
  ✅ RULE 2  output dir deleted before run — no partial state
  ✅ RULE 4  relative paths only
  ✅ RULE 5  deterministic — seed fixed, no hidden state
  ✅ STRICT  fails hard if data/{dataset} missing

Usage:
    python sanity_check.py --dataset cifar10
    python sanity_check.py --dataset mnist
    python sanity_check.py --dataset flowers102
    python sanity_check.py --dataset cifar10 --splits train test
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def collect_image_paths(split_dir: Path) -> dict[str, list[Path]]:
    """Return {class_name: [path, ...]} from an ImageFolder-structured directory."""
    result = {}
    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir():
            imgs = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
            if imgs:
                result[class_dir.name] = imgs
    return result


# ─── CHECK 1: CLASS BALANCE ───────────────────────────────────────────────────

def check_class_balance(split_data: dict[str, dict]) -> dict:
    """
    Returns per-split class counts and imbalance ratio.
    Imbalance ratio = max_class_count / min_class_count  (1.0 = perfectly balanced)
    """
    stats = {}
    for split, class_map in split_data.items():
        counts = {cls: len(paths) for cls, paths in class_map.items()}
        total  = sum(counts.values())
        c_max  = max(counts.values())
        c_min  = min(counts.values())
        stats[split] = {
            "total":          total,
            "n_classes":      len(counts),
            "counts":         counts,
            "imbalance_ratio": round(c_max / c_min, 4) if c_min > 0 else float("inf"),
            "max_class":      max(counts, key=counts.get),
            "min_class":      min(counts, key=counts.get),
        }
    return stats


def plot_class_balance(balance_stats: dict, out_dir: Path, dataset: str):
    splits = list(balance_stats.keys())
    n      = len(splits)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    fig.suptitle(f"{dataset} — Class Distribution", fontsize=14, fontweight="bold")

    for ax, split in zip(axes[0], splits):
        counts = balance_stats[split]["counts"]
        classes = list(counts.keys())
        values  = list(counts.values())

        # Colour-code: green = balanced, red = outlier (>2× mean)
        mean_c = np.mean(values)
        colors = ["#e74c3c" if v > 2 * mean_c or v < 0.5 * mean_c else "#2ecc71"
                  for v in values]

        ax.bar(range(len(classes)), values, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_title(
            f"{split}  |  {balance_stats[split]['total']:,} images  "
            f"|  imbalance ratio: {balance_stats[split]['imbalance_ratio']:.2f}",
            fontsize=10
        )
        ax.set_ylabel("Image count")
        ax.set_xlabel("Class")

    plt.tight_layout()
    plt.savefig(out_dir / "class_balance.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─── CHECK 2: RESOLUTION ─────────────────────────────────────────────────────

def check_resolutions(split_data: dict[str, dict], sample_limit: int = 500) -> dict:
    """
    Sample up to `sample_limit` images per split, record (W, H).
    Returns resolution stats per split.
    """
    stats = {}
    for split, class_map in split_data.items():
        all_paths = [p for paths in class_map.values() for p in paths]
        sampled   = random.sample(all_paths, min(sample_limit, len(all_paths)))

        wh = []
        for p in tqdm(sampled, desc=f"  Resolutions [{split}]", ncols=80, leave=False):
            with Image.open(p) as img:
                wh.append(img.size)   # (W, H)

        widths  = [w for w, _ in wh]
        heights = [h for _, h in wh]

        stats[split] = {
            "sampled":    len(wh),
            "unique_resolutions": len(set(wh)),
            "w": {"min": min(widths),  "max": max(widths),  "mean": round(np.mean(widths), 1)},
            "h": {"min": min(heights), "max": max(heights), "mean": round(np.mean(heights), 1)},
            "_wh_sample": wh,   # kept for plotting, stripped before JSON save
        }
    return stats


def plot_resolutions(res_stats: dict, out_dir: Path, dataset: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"{dataset} — Resolution Scatter (sampled)", fontsize=13, fontweight="bold")

    colors = ["#3498db", "#e67e22", "#9b59b6"]
    for (split, stats), color in zip(res_stats.items(), colors):
        wh = stats["_wh_sample"]
        ws = [w for w, _ in wh]
        hs = [h for _, h in wh]
        ax.scatter(ws, hs, alpha=0.35, s=12, label=split, color=color)

    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "resolution_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─── CHECK 3: SAMPLE GRID ────────────────────────────────────────────────────

def plot_sample_grid(split_data: dict[str, dict], out_dir: Path, dataset: str,
                     grid_rows: int = 5, grid_cols: int = 5):
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    for split, class_map in split_data.items():
        all_paths = [p for paths in class_map.values() for p in paths]
        chosen    = random.sample(all_paths, min(grid_rows * grid_cols, len(all_paths)))

        fig = plt.figure(figsize=(grid_cols * 1.8, grid_rows * 1.8))
        fig.suptitle(f"{dataset} / {split} — random samples", fontsize=11, fontweight="bold")
        gs  = gridspec.GridSpec(grid_rows, grid_cols, figure=fig,
                                hspace=0.05, wspace=0.05)

        for i, path in enumerate(chosen):
            ax = fig.add_subplot(gs[i // grid_cols, i % grid_cols])
            with Image.open(path) as img:
                ax.imshow(img, cmap="gray" if img.mode == "L" else None)
            ax.axis("off")
            ax.set_title(path.parent.name, fontsize=5, pad=1)

        plt.savefig(samples_dir / f"{split}_grid.png", dpi=150, bbox_inches="tight")
        plt.close()


# ─── MAIN STAGE ───────────────────────────────────────────────────────────────

def run_sanity_check(dataset: str, splits: list[str]):
    data_dir = ROOT / "data" / dataset

    # ── STRICT: fail hard if data missing ─────────────────────────────────────
    assert data_dir.exists(), (
        f"[FAIL] data/{dataset}/ not found. Run download_datasets.py first."
    )

    out_dir = ROOT / "outputs" / dataset / "sanity"

    # ── RULE 2: delete before run ──────────────────────────────────────────────
    reset_dir(out_dir)

    print(f"\n{'='*55}")
    print(f"Sanity Check: {dataset}  |  splits: {splits}")
    print(f"Output → outputs/{dataset}/sanity/")
    print(f"{'='*55}")

    # ── Collect all image paths ────────────────────────────────────────────────
    split_data: dict[str, dict] = {}
    for split in splits:
        split_dir = data_dir / split
        assert split_dir.exists(), f"[FAIL] data/{dataset}/{split}/ missing."
        split_data[split] = collect_image_paths(split_dir)
        n = sum(len(v) for v in split_data[split].values())
        print(f"  Found {n:,} images in {split}  ({len(split_data[split])} classes)")

    # ── CHECK 1: Class balance ─────────────────────────────────────────────────
    print("\n[1/3] Class balance ...")
    balance = check_class_balance(split_data)
    plot_class_balance(balance, out_dir, dataset)

    for split, s in balance.items():
        ratio = s["imbalance_ratio"]
        flag  = "⚠️  IMBALANCED" if ratio > 5.0 else "✓"
        print(f"  {split}: imbalance ratio = {ratio:.2f}  {flag}")
        if ratio > 5.0:
            print(f"         max class: {s['max_class']}  min class: {s['min_class']}")

    # ── CHECK 2: Resolutions ───────────────────────────────────────────────────
    print("\n[2/3] Resolution scan ...")
    res_stats = check_resolutions(split_data)
    plot_resolutions(res_stats, out_dir, dataset)

    all_resolutions = set()
    for split, s in res_stats.items():
        all_resolutions.add(s["unique_resolutions"])
        print(f"  {split}: W=[{s['w']['min']}..{s['w']['max']}] "
              f"H=[{s['h']['min']}..{s['h']['max']}]  "
              f"unique resolutions={s['unique_resolutions']}")

    if max(all_resolutions) > 1:
        print("  ⚠️  Variable resolutions detected → resize in preprocess.py is REQUIRED")
    else:
        print("  ✓  Fixed resolution — resize still recommended for cross-dataset consistency")

    # ── CHECK 3: Sample grids ──────────────────────────────────────────────────
    print("\n[3/3] Generating sample grids ...")
    plot_sample_grid(split_data, out_dir, dataset)
    print(f"  ✓  Saved to outputs/{dataset}/sanity/samples/")

    # ── Save JSON report ───────────────────────────────────────────────────────
    # Strip non-serialisable _wh_sample before saving
    report_res = {
        split: {k: v for k, v in s.items() if k != "_wh_sample"}
        for split, s in res_stats.items()
    }
    report = {
        "dataset":    dataset,
        "splits":     splits,
        "balance":    balance,
        "resolution": report_res,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Sanity check complete → outputs/{dataset}/sanity/report.json")
    return report


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 4 — Data sanity check (class balance, resolution, sample grids)"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=list(SPLITS_BY_DATASET.keys()),
        help="Dataset to inspect"
    )
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="Which splits to inspect (default: all splits for the dataset)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sample grids (default: 42)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    splits = args.splits or SPLITS_BY_DATASET[args.dataset]
    run_sanity_check(args.dataset, splits)