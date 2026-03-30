"""
Experiment: Cross-Dataset Zero-Shot Generalization
====================================================
Loads a model trained on a SOURCE dataset, extracts features on a TARGET
dataset using the frozen source model, then runs retrieval + evaluation.

This tests the design document hypothesis:
  "OSAG will maintain robust performance via adaptive gating, while CNN
   will experience catastrophic mAP collapse on out-of-distribution data."

Usage:
    python src/experiments/cross_dataset.py --source cifar10 --target cifar10 --method cnn
    python src/experiments/cross_dataset.py --source cifar10 --target eurosat --method osag
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stages.extract_features import load_preprocessed, denormalize, SPLITS_BY_DATASET
from utils.index import RetrievalIndex
from utils.metrics import precision_at_k, recall_at_k, average_precision


def load_source_model(source_dataset: str, method: str):
    """Load a model trained on the source dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = ROOT / "outputs" / source_dataset / method / "model" / "model.pt"
    assert model_path.exists(), (
        f"[FAIL] Source model not found: {model_path}\n"
        f"       Train it first: python src/stages/train_model.py --dataset {source_dataset} --method {method}"
    )

    # Determine source dataset properties for model construction
    source_meta_path = ROOT / "outputs" / source_dataset / "preprocessed" / "meta.json"
    with open(source_meta_path) as f:
        source_meta = json.load(f)

    C = source_meta["n_channels"]
    num_classes = source_meta["n_classes"]

    if method == "nn":
        from models.nn import ShallowNN
        H, W = source_meta["image_size"]
        model = ShallowNN(input_dim=C*H*W, emb_dim=128, num_classes=num_classes)
    elif method == "dnn":
        from models.dnn import DeepNN
        H, W = source_meta["image_size"]
        model = DeepNN(input_dim=C*H*W, emb_dim=128, num_classes=num_classes)
    elif method == "cnn":
        from models.cnn import SimpleCNN
        model = SimpleCNN(in_channels=C, emb_dim=128, num_classes=num_classes)
    elif method == "dsfm":
        from features.dsfm import DSFM
        model = DSFM(in_channels=C, emb_dim=128, num_classes=num_classes)
    elif method == "osag":
        from models.osag import OSAG
        model = OSAG(in_channels=C, emb_dim=128, num_classes=num_classes)
    else:
        raise ValueError(f"[FAIL] Method '{method}' not supported for cross-dataset eval")

    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model = model.to(device)
    model.eval()
    return model, device


def extract_with_model(model, device, images: np.ndarray, method: str) -> np.ndarray:
    """Extract features from images using a frozen model."""
    features_list = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.tensor(images[i:i+batch_size], dtype=torch.float32, device=device)
            feats = model(batch, return_features=True)
            # OSAG returns (features, ortho_loss) during training, but model.eval() returns features only
            if isinstance(feats, tuple):
                feats = feats[0]
            features_list.append(feats.cpu().numpy())
    return np.concatenate(features_list, axis=0)


def evaluate_features(q_feats, q_labels, g_feats, g_labels, k: int):
    """Build FAISS index, search, and compute metrics."""
    index = RetrievalIndex(feature_dim=q_feats.shape[1])
    index.add(g_feats)
    distances, indices = index.search(q_feats, k=k)

    unique_labels, counts = np.unique(g_labels, return_counts=True)
    label_to_count = dict(zip(unique_labels, counts))

    ap_list, p_list, r_list = [], [], []
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


def run_cross_dataset(source: str, target: str, method: str, k: int = 5):
    out_dir = ROOT / "outputs" / "cross_dataset" / f"{source}_to_{target}" / method
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"\n{'='*60}")
    print(f"Cross-Dataset Evaluation")
    print(f"Source: {source} | Target: {target} | Method: {method}")
    print(f"{'='*60}")

    t_total = time.perf_counter()

    # 1. Load source model (frozen)
    print("\n[1/4] Loading source model...")
    model, device = load_source_model(source, method)
    print(f"  Loaded model trained on: {source}")

    # 2. Load target dataset
    print("\n[2/4] Loading target dataset...")
    splits = SPLITS_BY_DATASET.get(target, ["train", "test"])
    gallery_split = "train"
    query_split = splits[-1]  # test or val

    g_images, g_labels, g_meta = load_preprocessed(target, gallery_split)
    q_images, q_labels, q_meta = load_preprocessed(target, query_split)
    g_images = denormalize(g_images, g_meta)
    q_images = denormalize(q_images, q_meta)
    print(f"  Gallery ({gallery_split}): {g_images.shape[0]:,} images")
    print(f"  Queries ({query_split}): {q_images.shape[0]:,} images")

    # 3. Extract features using frozen source model
    print("\n[3/4] Extracting features with frozen source model...")
    g_feats = extract_with_model(model, device, g_images, method)
    q_feats = extract_with_model(model, device, q_images, method)
    print(f"  Gallery features: {g_feats.shape}")
    print(f"  Query features: {q_feats.shape}")

    # 4. Evaluate
    print(f"\n[4/4] Evaluating Top-{k} retrieval...")
    scores = evaluate_features(q_feats, q_labels, g_feats, g_labels, k)

    print(f"\n  Results:")
    for metric, val in scores.items():
        print(f"    {metric}: {val:.4f}")

    # Save
    result = {
        "source_dataset": source,
        "target_dataset": target,
        "method": method,
        "k": k,
        "scores": scores,
        "total_time": time.perf_counter() - t_total,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved -> outputs/cross_dataset/{source}_to_{target}/{method}/metrics.json")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-Dataset Zero-Shot Generalization")
    parser.add_argument("--source", required=True, help="Dataset the model was trained on")
    parser.add_argument("--target", required=True, help="Dataset to evaluate on")
    parser.add_argument("--method", required=True)
    parser.add_argument("-k", "--top-k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cross_dataset(source=args.source, target=args.target, method=args.method, k=args.top_k)
