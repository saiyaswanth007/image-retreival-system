"""
Task 7: Interactive Retrieval Dashboard
========================================
Upload a query image, select a method and Top-K,
view retrieved images with similarity scores and metrics.

Reads ONLY from previously computed pipeline outputs:
  - Trained models   : outputs/{dataset}/{method}/model/model.pt
  - Gallery features : outputs/{dataset}/{method}/features/train.npy
  - Gallery labels   : outputs/{dataset}/{method}/features/train_labels.npy
  - Gallery images   : outputs/{dataset}/preprocessed/train.npy
  - Class names      : outputs/{dataset}/preprocessed/class_names.json

Design rules enforced:
  ✅ RULE 1  reads ONLY from previous pipeline outputs
  ✅ RULE 4  relative paths via ROOT
  ✅ No state mutation — never writes to outputs/

Usage:
    python src/dashboard.py
    python src/dashboard.py --port 7861
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ─── CONFIGURATION ───────────────────────────────────────────────────────────

DATASETS = ["cifar10", "mnist"]
METHODS = ["lbp", "color", "nn", "dnn", "cnn", "dsfm", "osag"]
NEURAL_METHODS = {"nn", "dnn", "cnn", "dsfm", "osag"}
IMAGE_SIZE = 64  # Preprocessed image size


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_gallery(dataset: str, method: str):
    """Load precomputed gallery features, labels, and raw images."""
    feat_path  = ROOT / "outputs" / dataset / method / "features" / "train.npy"
    label_path = ROOT / "outputs" / dataset / method / "features" / "train_labels.npy"
    img_path   = ROOT / "outputs" / dataset / "preprocessed" / "train.npy"
    meta_path  = ROOT / "outputs" / dataset / "preprocessed" / "meta.json"
    class_path = ROOT / "outputs" / dataset / "preprocessed" / "class_names.json"

    assert feat_path.exists(), f"Gallery features missing for {dataset}/{method}. Run pipeline first!"
    assert img_path.exists(), f"Preprocessed images missing for {dataset}."

    gallery_feats  = np.load(feat_path)
    gallery_labels = np.load(label_path)
    gallery_images = np.load(img_path)

    with open(meta_path) as f:
        meta = json.load(f)

    class_names = {}
    if class_path.exists():
        with open(class_path) as f:
            class_names = json.load(f)

    return gallery_feats, gallery_labels, gallery_images, meta, class_names


def denormalize_images(images: np.ndarray, meta: dict) -> np.ndarray:
    """Reverse normalization for display."""
    if not meta.get("normalized", False):
        return images
    mean = np.array(meta["norm_mean"], dtype=np.float32)
    std  = np.array(meta["norm_std"],  dtype=np.float32)
    images = images * std[None, :, None, None] + mean[None, :, None, None]
    return np.clip(images, 0, 255).astype(np.uint8)


def nchw_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a single (C, H, W) array to a PIL Image."""
    arr = arr.transpose(1, 2, 0)  # CHW -> HWC
    if arr.shape[2] == 1:
        arr = np.squeeze(arr, axis=2)
    return Image.fromarray(arr.astype(np.uint8))


def extract_query_features(query_pil: Image.Image, dataset: str, method: str, meta: dict):
    """Extract features from a single query image."""
    import torch

    # Preprocess the query image to match pipeline format
    query = query_pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    arr = np.array(query, dtype=np.float32)  # H, W, C
    arr = arr.transpose(2, 0, 1)              # C, H, W

    # Normalize with train statistics
    if meta.get("normalized", False):
        mean = np.array(meta["norm_mean"], dtype=np.float32)
        std  = np.array(meta["norm_std"],  dtype=np.float32)
        arr = (arr - mean[:, None, None]) / std[:, None, None]

    if method in NEURAL_METHODS:
        # Use trained model
        device = torch.device("cpu")
        model_path = ROOT / "outputs" / dataset / method / "model" / "model.pt"
        assert model_path.exists(), f"Model not found for {dataset}/{method}"

        C = 3
        num_classes = meta.get("n_classes", 10)

        if method == "nn":
            from models.nn import ShallowNN
            model = ShallowNN(input_dim=C*IMAGE_SIZE*IMAGE_SIZE, emb_dim=128, num_classes=num_classes)
        elif method == "dnn":
            from models.dnn import DeepNN
            model = DeepNN(input_dim=C*IMAGE_SIZE*IMAGE_SIZE, emb_dim=128, num_classes=num_classes)
        elif method == "cnn":
            from models.cnn import SimpleCNN
            model = SimpleCNN(in_channels=C, emb_dim=128, num_classes=num_classes)
        elif method == "dsfm":
            from features.dsfm import DSFM
            model = DSFM(in_channels=C, emb_dim=128, num_classes=num_classes)
        elif method == "osag":
            from models.osag import OSAG
            model = OSAG(in_channels=C, emb_dim=128, num_classes=num_classes)

        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()

        with torch.no_grad():
            batch = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
            feats = model(batch, return_features=True)
            if isinstance(feats, tuple):
                feats = feats[0]
            return feats.numpy().squeeze(0)

    elif method == "lbp":
        from features.lbp import extract_lbp
        # LBP needs denormalized images
        if meta.get("normalized", False):
            mean = np.array(meta["norm_mean"], dtype=np.float32)
            std  = np.array(meta["norm_std"],  dtype=np.float32)
            raw = arr * std[:, None, None] + mean[:, None, None]
        else:
            raw = arr
        feats = extract_lbp(raw[np.newaxis], grid=4, uniform=True)
        return feats.squeeze(0)

    elif method == "color":
        from features.color import extract_color
        if meta.get("normalized", False):
            mean = np.array(meta["norm_mean"], dtype=np.float32)
            std  = np.array(meta["norm_std"],  dtype=np.float32)
            raw = arr * std[:, None, None] + mean[:, None, None]
        else:
            raw = arr
        feats = extract_color(raw[np.newaxis], grid=4, hist_bins=32, use_hsv=True)
        return feats.squeeze(0)


def retrieve(query_feats: np.ndarray, gallery_feats: np.ndarray, k: int):
    """Euclidean nearest-neighbor search."""
    from utils.index import RetrievalIndex
    index = RetrievalIndex(feature_dim=query_feats.shape[0])
    index.add(gallery_feats)
    distances, indices = index.search(query_feats[np.newaxis], k=k)
    return distances.squeeze(0), indices.squeeze(0)


# ─── GRADIO APP ───────────────────────────────────────────────────────────────

def build_app():
    import gradio as gr

    def do_retrieval(query_image, dataset, method, top_k):
        """Main retrieval callback."""
        if query_image is None:
            return [], "Please upload a query image."

        try:
            # Load gallery
            g_feats, g_labels, g_images, meta, class_names = load_gallery(dataset, method)
            g_images_display = denormalize_images(g_images, meta)

            # Extract query features
            query_pil = Image.fromarray(query_image)
            q_feats = extract_query_features(query_pil, dataset, method, meta)

            # Retrieve
            distances, indices = retrieve(q_feats, g_feats, top_k)

            # Build result gallery
            results = []
            report_lines = [
                f"**Query**: uploaded image",
                f"**Dataset**: {dataset} | **Method**: {method} | **Top-K**: {top_k}",
                "",
                "| Rank | Class | Distance |",
                "|------|-------|----------|",
            ]

            for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
                label = int(g_labels[idx])
                class_name = class_names.get(str(label), str(label))

                img_pil = nchw_to_pil(g_images_display[idx])
                img_pil = img_pil.resize((128, 128), Image.NEAREST)
                caption = f"#{rank} {class_name} (d={dist:.2f})"
                results.append((img_pil, caption))

                report_lines.append(f"| {rank} | {class_name} | {dist:.4f} |")

            # Compute Precision@K (if all same class as rank-1)
            top1_label = int(g_labels[indices[0]])
            matches = sum(1 for idx in indices if int(g_labels[idx]) == top1_label)
            precision = matches / top_k

            report_lines.extend([
                "",
                f"**Precision@{top_k}** (vs. top-1 class): {precision:.2%}",
                f"**Nearest distance**: {distances[0]:.4f}",
                f"**Farthest distance**: {distances[-1]:.4f}",
            ])

            report = "\n".join(report_lines)
            return results, report

        except Exception as e:
            return [], f"**Error**: {str(e)}"

    # ── Build Interface ───────────────────────────────────────────────────────
    with gr.Blocks(
        title="Image Retrieval Dashboard",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# 🔍 Image Retrieval Dashboard\n"
            "Upload a query image, select a dataset and method, and retrieve the Top-K most similar images."
        )

        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Image(label="Query Image", type="numpy")
                dataset_dd = gr.Dropdown(choices=DATASETS, value="cifar10", label="Dataset")
                method_dd = gr.Dropdown(choices=METHODS, value="cnn", label="Method")
                topk_slider = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Top-K")
                run_btn = gr.Button("Retrieve", variant="primary")

            with gr.Column(scale=2):
                gallery_output = gr.Gallery(label="Retrieved Images", columns=5, height="auto")
                report_output = gr.Markdown(label="Metrics Report")

        run_btn.click(
            fn=do_retrieval,
            inputs=[query_input, dataset_dd, method_dd, topk_slider],
            outputs=[gallery_output, report_output],
        )

    return app


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Task 7: Interactive Retrieval Dashboard")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = build_app()
    app.launch(server_port=args.port, share=args.share)
