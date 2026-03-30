"""
Deep Neural Network (DNN)
==========================
Task 2: "Retrieval using Deep NN"

A deeper Multi-Layer Perceptron (3+ hidden layers) designed to learn
hierarchical dense embeddings for retrieval.

Architecture:
    Input: Flattened image (C * H * W)
    Layer 1: Linear(2048) -> ReLU
    Layer 2: Linear(1024) -> ReLU
    Layer 3: Linear(512) -> ReLU
    Layer 4: Linear(emb_dim)        ← Features extracted here
    Classifier: Linear(num_classes)
"""

import torch
import torch.nn as nn


class DeepNN(nn.Module):
    def __init__(
        self,
        input_dim: int = 3 * 64 * 64,
        emb_dim: int = 128,            # feature output dimension
        num_classes: int = 10,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim

        # Deeper Feature Extractor
        self.encoder = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            # Bottleneck / Embedding space
            nn.Linear(512, emb_dim),
        )

        # Classification Head
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
                             for distance-based retrieval.
        """
        features = self.encoder(x)

        if return_features:
            return features

        logits = self.classifier(features)
        return logits
