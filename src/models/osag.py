"""
Orthogonal Subspace Adaptive Gating (OSAG)
===========================================
Task 5: Hybrid Retrieval Architecture

Fuses Semantic Context (CNN) and High-Frequency Structure (DSFM).
Applies Gram-Schmidt Orthogonal Subspace Projection to ensure features
are geometrically independent, followed by dynamic contextual gating.
"""

import torch
import torch.nn as nn

from models.cnn import SimpleCNN
from features.dsfm import DSFM

class OSAG(nn.Module):
    def __init__(self, in_channels: int = 3, emb_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.emb_dim = emb_dim
        
        # Branch 1: Semantic Context (CNN)
        self.cnn = SimpleCNN(in_channels=in_channels, emb_dim=emb_dim, num_classes=num_classes)
        
        # Branch 2: Topological/Frequency Context (DSFM)
        self.dsfm = DSFM(in_channels=in_channels, emb_dim=emb_dim, num_classes=num_classes)
        
        # Dynamically learned projection gating mechanism for explicit convex combination
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.Sigmoid()
        )
        
        
        # Optional classification head for API compatibility
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # 1. Extract base embeddings
        f_cnn = self.cnn(x, return_features=True)    # (B, 128)
        f_dsfm = self.dsfm(x, return_features=True)  # (B, 128)
        
        # 2. Orthogonal Subspace Projection (Gram-Schmidt)
        # Project f_dsfm onto the orthogonal complement of f_cnn
        # f_perp = f_dsfm - proj_{f_cnn}(f_dsfm)
        dot_cf = (f_cnn * f_dsfm).sum(dim=1, keepdim=True) # (B, 1)
        dot_cc = (f_cnn * f_cnn).sum(dim=1, keepdim=True)  # (B, 1)
        
        f_perp = f_dsfm - (dot_cf / (dot_cc + 1e-8)) * f_cnn  # (B, 128)
        
        # 3. Dynamic Contextual Gating Fusion (Convex Combination)
        concat_features = torch.cat([f_cnn, f_perp], dim=1)  # (B, 256)
        
        # Calculate context-aware attention weights vector w -> (B, 128)
        w = self.gate(concat_features)
        
        # Explicit convex combination: f_hybrid = w * f_cnn + (1 - w) * f_perp
        f_hybrid = w * f_cnn + (1.0 - w) * f_perp  # (B, 128)
        
        if return_features:
            if self.training:
                # Penalty to prevent Gram-Schmidt from aggressively stripping features.
                # Enforces natural decorrelation between CNN and DSFM latent spaces.
                cos_sim = torch.nn.functional.cosine_similarity(f_cnn, f_dsfm, dim=1)
                ortho_loss = torch.abs(cos_sim).mean()
                return f_hybrid, ortho_loss
            return f_hybrid
            
        return self.classifier(f_hybrid)
