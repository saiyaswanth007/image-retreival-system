"""
Differentiable Spatial-Frequency Morphology (DSFM)
===================================================
Task 4: "Novel Method Implementation"

A completely novel neural feature extractor that unifies:
  1. Learnable Gabor Frequency manifold
  2. Probabilistic SoftMorph structure manifold

These are unified via a topology-guided cross-attention operation,
yielding a dense invariant embedding space optimized for 
cross-domain retrieval (e.g., EuroSAT generalization).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableGaborConv2d(nn.Module):
    r"""
    Part 1: Learnable Gabor Frequency Extraction.
    Dynamically generates convolutional filters from mathematical Gabor 
    equations whose parameters (\theta, \lambda, \sigma, \gamma, \psi)
    are learned via backpropagation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 11, stride: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

        # Compress input channels down to 1 (grayscale) to apply canonical Gabor
        self.rgb_to_gray = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

        # Gabor mathematical parameters
        self.theta = nn.Parameter(torch.rand(out_channels) * math.pi)
        self.sigma = nn.Parameter(torch.rand(out_channels) * 2.0 + 1.0)
        self.Lambda = nn.Parameter(torch.rand(out_channels) * 5.0 + 3.0)
        self.psi = nn.Parameter(torch.rand(out_channels) * math.pi)
        self.gamma = nn.Parameter(torch.rand(out_channels) * 0.5 + 0.5)

        # Static coordinate grid
        range_val = kernel_size // 2
        y, x = torch.meshgrid(
            torch.arange(-range_val, range_val + 1, dtype=torch.float32),
            torch.arange(-range_val, range_val + 1, dtype=torch.float32),
            indexing='ij'
        )
        self.register_buffer('y', y.clone())
        self.register_buffer('x', x.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = self.rgb_to_gray(x)  # (B, 1, H, W)
        
        # Broadcastable shapes: (out_channels, 1, 1)
        theta = self.theta.view(-1, 1, 1)
        sigma = self.sigma.view(-1, 1, 1)
        Lambda = self.Lambda.view(-1, 1, 1)
        psi = self.psi.view(-1, 1, 1)
        gamma = self.gamma.view(-1, 1, 1)

        x_grid = self.x.view(1, self.kernel_size, self.kernel_size)
        y_grid = self.y.view(1, self.kernel_size, self.kernel_size)

        # Rotation
        x_prime = x_grid * torch.cos(theta) + y_grid * torch.sin(theta)
        y_prime = -x_grid * torch.sin(theta) + y_grid * torch.cos(theta)

        # Gabor definition: Envelope * Carrier
        envelope = torch.exp(-(x_prime**2 + (gamma**2) * (y_prime**2)) / (2 * sigma**2 + 1e-6))
        carrier = torch.cos(2 * math.pi * x_prime / (Lambda + 1e-6) + psi)

        # Generated kernel: (out_channels, 1, K, K)
        gabor_kernels = (envelope * carrier).unsqueeze(1) 

        # Standard conv2d utilizing the dynamically generated kernels
        return F.conv2d(gray, gabor_kernels, stride=self.stride, padding=self.padding)


class SoftMorphErosion(nn.Module):
    """Differentiable Erosion via Fuzzy Logic Product Norm"""
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.ones(1, 1, kernel_size, kernel_size), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # \prod Y_i = \exp( \sum \log Y_i )
        log_x = torch.log(x.clamp(min=1e-6))
        sum_log = F.conv2d(log_x, self.weight.to(x.device), padding=self.padding)
        return torch.exp(sum_log)


class SoftMorphDilation(nn.Module):
    """Differentiable Dilation via Fuzzy Logic Product Norm"""
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.ones(1, 1, kernel_size, kernel_size), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1 - \prod (1 - Y_i)
        inv_x = 1.0 - x
        log_inv_x = torch.log(inv_x.clamp(min=1e-6))
        sum_log = F.conv2d(log_inv_x, self.weight.to(x.device), padding=self.padding)
        return 1.0 - torch.exp(sum_log)


class SoftMorph(nn.Module):
    """
    Part 2: Differentiable Probabilistic Morphology.
    Computes invariant topological skeletons.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # Produce a bounded pseudo-binary activation map
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.erode = SoftMorphErosion(kernel_size=3)
        self.dilate = SoftMorphDilation(kernel_size=3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bottleneck(x)  # (B, 1, H, W)
        
        # Soft Opening
        eroded = self.erode(y)
        opened = self.dilate(eroded)
        
        # Skeletonization extraction metric: Y - Open(Y)
        skeleton = F.relu(y - opened)
        return skeleton


class DSFM(nn.Module):
    """
    Main Architect: Differentiable Spatial-Frequency Morphology
    """
    def __init__(
        self, 
        in_channels: int = 3, 
        emb_dim: int = 128, 
        num_classes: int = 10, 
        num_gabor_filters: int = 64
    ):
        super().__init__()
        self.emb_dim = emb_dim
        
        self.gabor_extractor = LearnableGaborConv2d(
            in_channels=in_channels, 
            out_channels=num_gabor_filters, 
            kernel_size=11
        )
        
        self.soft_morph = SoftMorph(in_channels=in_channels)
        
        # Unification bottleneck
        self.embedding = nn.Sequential(
            nn.Linear(num_gabor_filters, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True)
        )
        
        # Supervised Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 1. High-Frequency Manifold
        f_freq = self.gabor_extractor(x)  # (B, 64, H, W)
        
        # 2. Structural Topology Manifold
        f_struct = self.soft_morph(x)     # (B, 1, H, W)
        
        # 3. Topology-Guided Unification (Depth-wise Spatial Attention)
        # Sample high-frequency texture *only* along significant structural skeletons
        spatial_weights = F.softmax(f_struct.view(B, 1, -1), dim=-1) # (B, 1, H*W)
        f_freq_flat = f_freq.view(B, -1, H * W)                      # (B, 64, H*W)
        
        # (B, 64, H*W) x (B, H*W, 1) -> (B, 64, 1) -> (B, 64)
        pooled = torch.bmm(f_freq_flat, spatial_weights.transpose(1, 2)).squeeze(-1)
        
        # 4. Dense Retrieval Embedding
        embeddings = self.embedding(pooled) # (B, 128)
        
        if return_features:
            return embeddings
            
        logits = self.classifier(embeddings)
        return logits
