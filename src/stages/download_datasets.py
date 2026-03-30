"""
Dataset Download Script (Accelerated)
======================================
Downloads CIFAR-10, MNIST, and Oxford Flowers-102 into data/ directory.
Follows design philosophy: relative paths only, deterministic, no hidden state.

Speed-ups applied:
  1. Parallel image saving via ThreadPoolExecutor (I/O-bound bottleneck)
  2. Batch pre-fetch: entire dataset loaded into RAM before writing
  3. All three datasets downloaded concurrently (ProcessPoolExecutor)
  4. Workers auto-scaled to min(32, cpu_count + 4)

Usage:
    python download_datasets.py                         # all three, auto workers
    python download_datasets.py --datasets cifar10 mnist
    python download_datasets.py --workers 16
    python download_datasets.py --sequential            # disable dataset-level parallelism
    python download_datasets.py --subset 0.1            # 10% of each split, class-balanced
    python download_datasets.py --subset 500            # exactly 500 images per split

Directory structure after download:
    data/
    ├── cifar10/    train/ test/
    ├── mnist/      train/ test/
    └── flowers102/ train/ val/  test/
"""

import argparse
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10, MNIST, Flowers102
from tqdm import tqdm


# ─── ROOT ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[2]   # src/stages/ → project root
DATA_DIR = ROOT / "data"

# Workers: cap at 32 so we don't thrash on HDDs
NUM_WORKERS = min(32, (os.cpu_count() or 4) + 4)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def sample_subset(dataset, subset) -> list:
    """
    Return a class-balanced subset of (img, label) indices.

    Args:
        dataset: torchvision dataset with a .targets attribute
        subset : float  -> fraction of each class to keep (e.g. 0.1 = 10%)
                 int    -> total images to keep (distributed evenly across classes)
                 None   -> return all indices unchanged

    Balancing strategy: sample proportionally per class so label distribution
    is preserved — important for fair Precision/Recall evaluation.
    """
    if subset is None:
        return list(range(len(dataset)))

    import random, math
    from collections import defaultdict

    # Build per-class index lists
    targets = (
        dataset.targets if hasattr(dataset, "targets")
        else [label for _, label in dataset]   # fallback for Flowers102
    )
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[int(label)].append(idx)

    n_classes = len(class_indices)

    if isinstance(subset, float):
        # fraction mode: keep `subset` fraction of each class
        assert 0.0 < subset <= 1.0, "--subset float must be in (0, 1]"
        per_class = {
            c: max(1, math.floor(len(idxs) * subset))
            for c, idxs in class_indices.items()
        }
    else:
        # count mode: distribute `subset` images evenly across classes
        assert subset > 0, "--subset int must be > 0"
        base  = max(1, subset // n_classes)
        extra = max(0, subset - base * n_classes)
        per_class = {c: base + (1 if i < extra else 0)
                     for i, c in enumerate(sorted(class_indices))}

    sampled = []
    for c, idxs in class_indices.items():
        k = min(per_class[c], len(idxs))
        sampled.extend(random.sample(idxs, k))

    random.shuffle(sampled)
    return sampled


def _save_one(args):
    """Top-level function (picklable) for ProcessPoolExecutor fallback."""
    img, class_dir, filename = args
    Path(class_dir).mkdir(parents=True, exist_ok=True)
    img.save(Path(class_dir) / filename)


def save_split_parallel(dataset, split_dir: Path, label_to_name: dict, workers: int,
                        indices: list = None):
    """
    Pre-load (img, label) pairs (optionally a subset), then flush to disk in parallel threads.
    ThreadPoolExecutor is ideal here: GIL is released during PIL PNG writes.
    """
    indices = indices if indices is not None else list(range(len(dataset)))
    print(f"    Loading {len(indices):,} / {len(dataset):,} items into RAM ...", flush=True)

    tasks = []
    for save_idx, ds_idx in enumerate(indices):
        img, label = dataset[ds_idx]
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        class_name = label_to_name[int(label)]
        class_dir  = split_dir / class_name
        tasks.append((img, str(class_dir), f"{save_idx:06d}.png"))

    print(f"    Writing {len(tasks):,} PNGs with {workers} threads ...", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_save_one, t) for t in tasks]
        for _ in tqdm(as_completed(futs), total=len(futs), desc=f"    {split_dir.name}", ncols=80):
            pass


# ─── DOWNLOADERS (each returns elapsed seconds) ───────────────────────────────

def download_cifar10(data_dir: Path, workers: int, subset=None) -> float:
    t0 = time.perf_counter()
    print("\n[CIFAR-10]  32x32 | 10 classes | 60k images")

    raw_dir   = data_dir / "cifar10" / "_raw"
    train_dir = data_dir / "cifar10" / "train"
    test_dir  = data_dir / "cifar10" / "test"

    reset_dir(raw_dir); reset_dir(train_dir); reset_dir(test_dir)

    train_set = CIFAR10(root=raw_dir, train=True,  download=True, transform=None)
    test_set  = CIFAR10(root=raw_dir, train=False, download=True, transform=None)

    label_to_name = {i: name for i, name in enumerate(train_set.classes)}

    print("  Train split:")
    save_split_parallel(train_set, train_dir, label_to_name, workers,
                        indices=sample_subset(train_set, subset))
    print("  Test split:")
    save_split_parallel(test_set,  test_dir,  label_to_name, workers,
                        indices=sample_subset(test_set,  subset))

    shutil.rmtree(raw_dir)
    elapsed = time.perf_counter() - t0
    print(f"  CIFAR-10 done in {elapsed:.1f}s")
    return elapsed


def download_mnist(data_dir: Path, workers: int, subset=None) -> float:
    t0 = time.perf_counter()
    print("\n[MNIST]  28x28 grayscale | 10 classes | 70k images")

    raw_dir   = data_dir / "mnist" / "_raw"
    train_dir = data_dir / "mnist" / "train"
    test_dir  = data_dir / "mnist" / "test"

    reset_dir(raw_dir); reset_dir(train_dir); reset_dir(test_dir)

    train_set = MNIST(root=raw_dir, train=True,  download=True, transform=None)
    test_set  = MNIST(root=raw_dir, train=False, download=True, transform=None)

    label_to_name = {i: str(i) for i in range(10)}

    print("  Train split:")
    save_split_parallel(train_set, train_dir, label_to_name, workers,
                        indices=sample_subset(train_set, subset))
    print("  Test split:")
    save_split_parallel(test_set,  test_dir,  label_to_name, workers,
                        indices=sample_subset(test_set,  subset))

    shutil.rmtree(raw_dir)
    elapsed = time.perf_counter() - t0
    print(f"  MNIST done in {elapsed:.1f}s")
    return elapsed


def download_flowers102(data_dir: Path, workers: int, subset=None) -> float:
    t0 = time.perf_counter()
    print("\n[Flowers-102]  variable size | 102 fine-grained classes | ~8k images")

    raw_dir   = data_dir / "flowers102" / "_raw"
    train_dir = data_dir / "flowers102" / "train"
    val_dir   = data_dir / "flowers102" / "val"
    test_dir  = data_dir / "flowers102" / "test"

    for d in [raw_dir, train_dir, val_dir, test_dir]:
        reset_dir(d)

    label_to_name = {i: f"class_{i+1:03d}" for i in range(102)}

    for split, out_dir in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        print(f"  {split.capitalize()} split:")
        ds = Flowers102(root=raw_dir, split=split, download=True, transform=None)
        # Build targets list for Flowers102 (no .targets attribute)
        ds.targets = [label for _, label in ds]
        save_split_parallel(ds, out_dir, label_to_name, workers,
                            indices=sample_subset(ds, subset))

    shutil.rmtree(raw_dir)
    elapsed = time.perf_counter() - t0
    print(f"  Flowers-102 done in {elapsed:.1f}s")
    return elapsed


# ─── DATASET-LEVEL PARALLELISM ────────────────────────────────────────────────

def _worker_cifar10(args):
    data_dir, workers, subset = args
    return ("cifar10", download_cifar10(Path(data_dir), workers, subset))

def _worker_mnist(args):
    data_dir, workers, subset = args
    return ("mnist", download_mnist(Path(data_dir), workers, subset))

def _worker_flowers102(args):
    data_dir, workers, subset = args
    return ("flowers102", download_flowers102(Path(data_dir), workers, subset))

DATASET_FNS = {
    "cifar10":    _worker_cifar10,
    "mnist":      _worker_mnist,
    "flowers102": _worker_flowers102,
}


# ─── SUMMARY ──────────────────────────────────────────────────────────────────

def print_summary(data_dir: Path, timings: dict):
    print("\n" + "=" * 58)
    print("Download complete")
    print("=" * 58)

    structure = {
        "cifar10":    ["train", "test"],
        "mnist":      ["train", "test"],
        "flowers102": ["train", "val", "test"],
    }

    for dataset, splits in structure.items():
        ds_dir = data_dir / dataset
        if not ds_dir.exists():
            continue
        for split in splits:
            split_dir = ds_dir / split
            if split_dir.exists():
                n_classes = sum(1 for p in split_dir.iterdir() if p.is_dir())
                n_images  = sum(1 for _ in split_dir.rglob("*.png"))
                t = f"  ({timings[dataset]:.1f}s)" if split == splits[-1] else ""
                print(f"  {dataset}/{split:<6}  {n_images:>6,} images  {n_classes:>3} classes{t}")

    print("=" * 58)
    total = sum(timings.values())
    print(f"Total wall time: {total:.1f}s  (datasets may have overlapped)")
    print()
    print("All splits are ImageFolder-compatible:")
    print("  from torchvision.datasets import ImageFolder")
    print("  ds = ImageFolder('data/cifar10/train')")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download datasets for image retrieval assignment (accelerated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["cifar10", "mnist", "flowers102"],
        default=["cifar10", "mnist", "flowers102"],
    )
    parser.add_argument(
        "--root", type=str, default=str(DATA_DIR),
        help="Data root directory (default: ./data)"
    )
    parser.add_argument(
        "--workers", type=int, default=NUM_WORKERS,
        help=f"Threads per split for image writing (default: {NUM_WORKERS})"
    )
    parser.add_argument(
        "--subset", type=str, default=None,
        metavar="N",
        help=(
            "Save only a subset of each split, class-balanced. "
            "Float → fraction per class  (e.g. 0.1 = 10%%). "
            "Int   → total images        (e.g. 500, spread evenly across classes). "
            "Omit for full dataset."
        )
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Disable dataset-level parallelism (useful on HDDs or low-RAM machines)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args     = parse_args()
    data_dir = Path(args.root)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Parse --subset: "0.1" → float, "500" → int, None → None
    subset = None
    if args.subset is not None:
        try:
            subset = int(args.subset)
        except ValueError:
            subset = float(args.subset)

    print(f"Target  : {data_dir}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Subset  : {subset if subset is not None else 'full'}")
    print(f"Threads per split: {args.workers}")
    print(f"Dataset parallelism: {'off (--sequential)' if args.sequential else 'on'}")

    wall_start = time.perf_counter()
    timings    = {}

    if args.sequential or len(args.datasets) == 1:
        for name in args.datasets:
            _, elapsed = DATASET_FNS[name]((str(data_dir), args.workers, subset))
            timings[name] = elapsed
    else:
        print("\nRunning datasets in parallel (use --sequential to disable)\n")
        with ProcessPoolExecutor(max_workers=len(args.datasets)) as pool:
            futs = {
                pool.submit(DATASET_FNS[name], (str(data_dir), args.workers, subset)): name
                for name in args.datasets
            }
            for fut in as_completed(futs):
                name, elapsed = fut.result()
                timings[name] = elapsed

    print(f"\nTotal wall time: {time.perf_counter() - wall_start:.1f}s")
    print_summary(data_dir, timings)