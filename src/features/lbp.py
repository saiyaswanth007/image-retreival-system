"""
LBP Feature Extractor (Pure NumPy)
===================================
Extracts Local Binary Pattern histograms from images.

Algorithm:
  1. RGB → grayscale (weighted luminance)
  2. 3×3 neighbourhood: compare 8 neighbours with center pixel
  3. Encode comparisons as 8-bit binary → decimal (0–255)
  4. Split image into grid×grid spatial regions
  5. Histogram per region (uniform-LBP: 59 bins, or full: 256 bins)
  6. Concatenate + L2-normalize → final feature vector

Public API:
    extract_lbp(images, grid=4, uniform=True) → np.ndarray  (N, D)
"""

import numpy as np


# ─── UNIFORM LBP LOOKUP TABLE ────────────────────────────────────────────────

def _build_uniform_lut() -> np.ndarray:
    """
    Build a 256-entry lookup table mapping each LBP code (0–255) to a
    uniform-LBP bin index.

    A pattern is "uniform" if it has at most 2 bitwise 0→1 or 1→0
    transitions when read as a circular 8-bit string.
    There are exactly 58 uniform patterns; all non-uniform map to bin 58.

    Returns:
        lut : np.ndarray  uint8  (256,)   values in [0..58]
    """
    lut = np.zeros(256, dtype=np.uint8)
    uniform_idx = 0

    for code in range(256):
        # Count circular bit transitions
        bits = format(code, "08b")
        transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))

        if transitions <= 2:
            lut[code] = uniform_idx
            uniform_idx += 1
        else:
            lut[code] = 58  # non-uniform bin

    return lut


# Pre-compute once at import time
_UNIFORM_LUT = _build_uniform_lut()

# Bins: 59 uniform (0..58) or 256 full
N_BINS_UNIFORM = 59
N_BINS_FULL = 256


# ─── CORE: COMPUTE LBP MAP ───────────────────────────────────────────────────

def _rgb_to_gray(images: np.ndarray) -> np.ndarray:
    """
    Convert NCHW RGB float32 → NHW grayscale uint8.

    Uses standard luminance weights: 0.2989 R + 0.5870 G + 0.1140 B
    Input values should be in [0..255] range (de-normalized).
    """
    # images: (N, 3, H, W)  →  weighted sum over channel axis
    r, g, b = images[:, 0], images[:, 1], images[:, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b           # (N, H, W)
    return np.clip(gray, 0, 255).astype(np.uint8)


def _compute_lbp_maps(gray: np.ndarray) -> np.ndarray:
    """
    Vectorised LBP over a batch of grayscale images.

    For each pixel (y, x), compare 8 neighbours with center:
        bit_k = 1 if neighbour_k >= center else 0
    Encode as:  LBP = sum(bit_k * 2^k)

    Neighbour ordering (clockwise from top-left):
        0  1  2
        7  C  3
        6  5  4

    Args:
        gray : uint8 array (N, H, W)

    Returns:
        lbp_maps : uint8 array (N, H-2, W-2)   — border pixels excluded
    """
    g = gray.astype(np.int16)   # promote to avoid uint8 underflow

    # 8 shifted views (all same shape: N, H-2, W-2)
    # Offsets: (dy, dx) for each neighbour in clockwise order
    offsets = [
        (-1, -1), (-1,  0), (-1, +1),
        ( 0, +1),
        (+1, +1), (+1,  0), (+1, -1),
        ( 0, -1),
    ]

    center = g[:, 1:-1, 1:-1]   # (N, H-2, W-2)

    lbp = np.zeros_like(center, dtype=np.uint8)
    for k, (dy, dx) in enumerate(offsets):
        neighbour = g[:, 1+dy : g.shape[1]-1+dy,
                        1+dx : g.shape[2]-1+dx]
        lbp += ((neighbour >= center).astype(np.uint8)) << k

    return lbp


# ─── CORE: HISTOGRAM FEATURES ────────────────────────────────────────────────

def _grid_histograms(
    lbp_maps: np.ndarray,
    grid: int,
    uniform: bool,
) -> np.ndarray:
    """
    Compute grid-based LBP histograms for a batch.

    1. Divide each LBP map into grid×grid spatial cells
    2. Compute histogram of LBP codes in each cell
    3. Concatenate all cell histograms
    4. L2-normalize the full vector

    Args:
        lbp_maps : uint8  (N, H, W)
        grid     : spatial divisions (e.g. 4 → 4×4 = 16 cells)
        uniform  : use uniform-LBP (59 bins) or full (256 bins)

    Returns:
        features : float32  (N, grid*grid*n_bins)
    """
    if uniform:
        # Remap codes via lookup table
        lbp_maps = _UNIFORM_LUT[lbp_maps]
        n_bins = N_BINS_UNIFORM
    else:
        n_bins = N_BINS_FULL

    N, H, W = lbp_maps.shape
    cell_h = H // grid
    cell_w = W // grid

    features = np.zeros((N, grid * grid * n_bins), dtype=np.float32)

    cell_idx = 0
    for gy in range(grid):
        for gx in range(grid):
            # Extract cell region
            y0, y1 = gy * cell_h, (gy + 1) * cell_h
            x0, x1 = gx * cell_w, (gx + 1) * cell_w
            cell = lbp_maps[:, y0:y1, x0:x1].reshape(N, -1)   # (N, cell_h*cell_w)

            # Histogram via bincount per image
            offset = cell_idx * n_bins
            for i in range(N):
                counts = np.bincount(cell[i], minlength=n_bins)[:n_bins]
                features[i, offset : offset + n_bins] = counts.astype(np.float32)

            cell_idx += 1

    # L2-normalize each feature vector
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms < 1e-7, 1.0, norms)
    features /= norms

    return features


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def extract_lbp(
    images: np.ndarray,
    grid: int = 4,
    uniform: bool = True,
) -> np.ndarray:
    """
    Extract LBP histogram features from a batch of images.

    Args:
        images  : float32  (N, C, H, W)  — pixel values in [0..255] range
                  (de-normalize before calling if data was standardized)
        grid    : number of spatial divisions per axis (default 4 → 4×4 cells)
        uniform : use uniform-LBP (59 bins, more robust) or full (256 bins)

    Returns:
        features : float32  (N, D)  where D = grid * grid * n_bins
                   L2-normalized
    """
    assert images.ndim == 4, f"Expected NCHW, got shape {images.shape}"
    assert images.shape[1] in (1, 3), f"Expected 1 or 3 channels, got {images.shape[1]}"

    # ── RGB → grayscale ───────────────────────────────────────────────────────
    if images.shape[1] == 3:
        gray = _rgb_to_gray(images)
    else:
        gray = np.clip(images[:, 0], 0, 255).astype(np.uint8)

    # ── Compute LBP maps ──────────────────────────────────────────────────────
    lbp_maps = _compute_lbp_maps(gray)

    # ── Grid histograms ───────────────────────────────────────────────────────
    features = _grid_histograms(lbp_maps, grid, uniform)

    return features
