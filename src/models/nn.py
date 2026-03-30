"""
Shallow Neural Network (NN)
============================
Task 2: "Retrieval using NN"

A simple 1-hidden-layer MLP designed to extract a dense bottleneck
feature (embedding) for image retrieval.

Architecture:
    Input: Flattened image (C * W * H)
    Layer 1: Linear -> ReLU (Hidden Dimension)
    Layer 2: Linear -> Embedding (emb_dim)  ← Features extracted here
    Classifier: Linear (num_classes)
"""

import torch
import torch.nn as nn


class ShallowNN(nn.Module):
    def __init__(
        self,
        input_dim: int = 3 * 64 * 64,  # default for 64x64 RGB
        hidden_dim: int = 1024,
        emb_dim: int = 128,            # feature output dimension
        num_classes: int = 10,         # CIFAR-10 default
    ):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim

        # Feature Extractor
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            # Bottleneck / Embedding space
            nn.Linear(hidden_dim, emb_dim),
        )

        # Classification Head (used only during training)
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) images
            return_features: If True, returns the `emb_dim` vector suitable
                             for L2/Cosine retrieval (bypasses classifier).
        """
        # (B, emb_dim)
        features = self.encoder(x)

        if return_features:
            return features

        # (B, num_classes)
        logits = self.classifier(features)
        return logits
