"""
Convolutional Neural Network (CNN)
===================================
Task 3: "CNN-based Retrieval"

A custom VGG-style CNN designed to learn translation-invariant spatial 
features for image retrieval.

Architecture:
    Block 1: Conv(32) -> Conv(32) -> MaxPool -> Dropout
    Block 2: Conv(64) -> Conv(64) -> MaxPool -> Dropout
    Block 3: Conv(128) -> Conv(128) -> MaxPool
    GlobalAvgPool: Spatial invariant pooling
    Linear(emb_dim)                 ← Features extracted here
    Classifier: Linear(num_classes)
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        emb_dim: int = 128,            # feature output dimension
        num_classes: int = 10,
    ):
        super().__init__()
        
        self.emb_dim = emb_dim

        # Spatial Feature Extractor
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32 (or 28x28 -> 14x14)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.2),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.3),

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Global pooling collapses spatial dimensions (B, 128, H, W) -> (B, 128, 1, 1)
        # Making the network resilient to varying input sizes WxH
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Bottleneck / Embedding space
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, emb_dim)
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
                             for semantic image retrieval.
        """
        spatial_feats = self.features(x)
        pooled = self.pool(spatial_feats)
        
        # (B, emb_dim)
        embeddings = self.embedding(pooled)

        if return_features:
            return embeddings

        # (B, num_classes)
        logits = self.classifier(embeddings)
        return logits
