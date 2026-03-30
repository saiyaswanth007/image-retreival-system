"""
Stage: Train Model
===================
Reads  : outputs/{dataset}/preprocessed/  (via data_loader)
Writes : outputs/{dataset}/{method}/model/
           ├── model.pt                   ← PyTorch state_dict
           └── metrics.json               ← Training history

Design rules enforced:
  ✅ RULE 1  reads ONLY from preprocessed output
  ✅ RULE 2  output dir deleted before run
  ✅ RULE 3  method-namespaced: outputs/{dataset}/{method}/model/
  ✅ RULE 4  relative paths via ROOT
  ✅ RULE 5  deterministic — fixed random seeds

Usage:
    python src/stages/train_model.py --dataset cifar10 --method nn --epochs 20
    python src/stages/train_model.py --dataset cifar10 --method dnn --lr 1e-3
    python src/stages/train_model.py --dataset mnist   --method cnn
"""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─── PATHS ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # src/stages/ → project root

# Allow imports from src/
sys.path.insert(0, str(ROOT / "src"))

from models.nn import ShallowNN
from models.dnn import DeepNN
from models.cnn import SimpleCNN

SPLITS_BY_DATASET = {
    "cifar10":    ["train", "test"],
    "mnist":      ["train", "test"],
    "flowers102": ["train", "val", "test"],
}

MODEL_REGISTRY = {
    "nn":  ShallowNN,
    "dnn": DeepNN,
    "cnn": SimpleCNN,
}


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def set_seed(seed: int = 42):
    """Rule 5: Determinism"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Slower but reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_preprocessed(dataset: str, split: str):
    """Load preprocessed arrays and metadata."""
    base = ROOT / "outputs" / dataset / "preprocessed"

    img_path   = base / f"{split}.npy"
    label_path = base / f"{split}_labels.npy"
    meta_path  = base / "meta.json"

    assert img_path.exists(), f"[FAIL] {img_path.relative_to(ROOT)} missing."
    assert label_path.exists(), f"[FAIL] {label_path.relative_to(ROOT)} missing."
    assert meta_path.exists(), f"[FAIL] {meta_path.relative_to(ROOT)} missing."

    images = np.load(img_path)
    labels = np.load(label_path)

    with open(meta_path) as f:
        meta = json.load(f)

    return images, labels, meta


# ─── MAIN STAGE ───────────────────────────────────────────────────────────────

def run_train_model(
    dataset: str,
    method: str,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
):
    out_dir = ROOT / "outputs" / dataset / method / "model"

    # ── RULE 2: delete before run ─────────────────────────────────────────────
    reset_dir(out_dir)

    print(f"\n{'='*58}")
    print(f"Train Model: {dataset} | method={method}")
    print(f"Params: epochs={epochs} batch={batch_size} lr={lr}")
    print(f"Output -> outputs/{dataset}/{method}/model/")
    print(f"{'='*58}")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    print("\n[1/3] Loading preprocessed data...")
    images_np, labels_np, meta = load_preprocessed(dataset, "train")

    N, C, H, W = images_np.shape
    num_classes = meta["n_classes"]
    print(f"  Train split: {N:,} images | {num_classes} classes | shape={images_np.shape}")

    # Create PyTorch DataLoader
    # We do NOT denormalize here — Neural Networks prefer zero-mean standardized inputs!
    x_tensor = torch.from_numpy(images_np).float()
    y_tensor = torch.from_numpy(labels_np).long()

    train_ds = TensorDataset(x_tensor, y_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # ── 2. Initialize Model ───────────────────────────────────────────────────
    print("\n[2/3] Initializing model...")
    ModelClass = MODEL_REGISTRY[method]
    
    if method in ("nn", "dnn"):
        model_kwargs = {"input_dim": C * H * W, "emb_dim": 128, "num_classes": num_classes}
    elif method == "cnn":
        model_kwargs = {"in_channels": C, "emb_dim": 128, "num_classes": num_classes}
    
    model = ModelClass(**model_kwargs).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {ModelClass.__name__}")
    print(f"  Parameters: {n_params:,}")

    # ── 3. Train Loop ─────────────────────────────────────────────────────────
    print(f"\n[3/3] Training for {epochs} epochs...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"loss": [], "accuracy": [], "epoch_times": []}
    t_total = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t_epoch = time.perf_counter()
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            _, predicted = logits.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        epoch_time = time.perf_counter() - t_epoch

        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)
        history["epoch_times"].append(epoch_time)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:>3}/{epochs}] | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"Acc: {epoch_acc:>5.2f}% | "
                  f"Time: {epoch_time:.1f}s")

    # ── Save outputs ──────────────────────────────────────────────────────────
    model_path = out_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n  Saved model weights -> {model_path.name}")

    metrics = {
        "dataset":      dataset,
        "method":       method,
        "epochs":       epochs,
        "batch_size":   batch_size,
        "lr":          lr,
        "weight_decay": weight_decay,
        "n_params":     n_params,
        "final_loss":   history["loss"][-1],
        "final_acc":    history["accuracy"][-1],
        "total_time":   time.perf_counter() - t_total,
        "history":      history,
    }
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete.")
    print(f"Total time: {metrics['total_time']:.1f}s")
    
    return metrics


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage: Train Model — learn feature representations",
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=list(SPLITS_BY_DATASET.keys()),
    )
    parser.add_argument(
        "--method", required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to train",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for determinism (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Enable TF32 for faster training on Ampere+ GPUs (no-op on older)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    run_train_model(
        dataset=args.dataset,
        method=args.method,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
